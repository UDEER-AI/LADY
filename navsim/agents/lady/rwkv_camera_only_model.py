from typing import Dict, Optional, List
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler

from fla.models.utils import Cache

from navsim.agents.lady.rwkv_config import RWKVConfig
from navsim.agents.lady.rwkv_camera_only_backbone import RWKVBackbone
from navsim.agents.lady.rwkv_features import BoundingBox2DIndex
from navsim.agents.rwkv_block.lady_block_fla import RWKV7_DoubleSelfAttnBlock
from navsim.agents.diffusion_block.blocks import linear_relu_ln, gen_sineembed_for_position, bias_init_with_prob, SinusoidalPosEmb, GridSampleCrossBEVAttention
from navsim.agents.diffusion_block.multimodel_loss import py_sigmoid_focal_loss
from navsim.common.enums import StateSE2Index


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim:int, bias=None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        if bias is not None:
            bias_init = bias_init_with_prob(bias)
            nn.init.constant_(self.mlp[-1].bias, bias_init)

    def forward(
        self, inputs
    ) -> torch.Tensor:
        x = self.mlp(inputs)
        return x


class RWKVModel(nn.Module):
    """Torch module for RWKV."""

    def __init__(self, config: RWKVConfig):
        """
        Initializes RWKV torch module.
        :param config: global config dataclass of RWKV.
        """

        super().__init__()

        self._config = config
        self._backbone = RWKVBackbone(config)

        self.bev_feature_len = config.img_vert_anchors * config.img_horz_anchors
        self._bev_feature_embedding = nn.Embedding(self.bev_feature_len, config.rwkv_d_model)
        self._status_embedding = nn.Embedding(1, config.rwkv_d_model)
        self._query_embedding = nn.Embedding(config.num_bounding_boxes, config.rwkv_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.rwkv_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.rwkv_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._rwkv_decoder = nn.ModuleList(
            [
                RWKV7_DoubleSelfAttnBlock(
                    hidden_size=config.rwkv_d_model,
                    norm_first=config.rwkv_d_norm_first,
                    num_heads=config.rwkv_d_num_head,
                    decay_low_rank_dim=config.decay_low_rank_dim,
                    gate_low_rank_dim=config.gate_low_rank_dim,
                    a_low_rank_dim=config.a_low_rank_dim,
                    v_low_rank_dim=config.v_low_rank_dim,
                    fuse_norm=config.fuse_norm,
                    n_layer=config.rwkv_d_num_layers,
                    layer_idx=i
                )
                for i in range(config.rwkv_d_num_layers)
            ]
        )
    
        self.fusion_drop = nn.Dropout(config.rwkv_fusion_drop)
        self.state_drop = nn.Dropout(config.rwkv_state_drop)
        self.ln_out = nn.LayerNorm(config.rwkv_d_model)

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.rwkv_d_ffn,
            d_model=config.rwkv_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            config=config
        )

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(config.rwkv_d_model, 1, 1, config.rwkv_d_model + 64),
        )

    def forward(
            self, 
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            past_features: Optional[List[Cache]] = [None, None, None, None],
            past_query: Optional[Cache] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
        ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"] # (B, _)

        valid_frame_len = None
        if 'valid_frame_len' in features:
            valid_frame_len: torch.Tensor = features["valid_frame_len"].view(-1, )

        bs, seq_len, C, H, W = camera_feature.shape
        camera_feature = camera_feature.view(bs * seq_len, C, H, W) # (seq_len * B, _, _, _)

        # get multi frame bev features
        bev_feature_upscale, bev_feature, _, past_features = self._backbone(camera_feature, past_features, use_cache, valid_frame_len)

        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]

        # get the augmented multi frame bev_feature
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        expanded_bs, T, C = bev_feature.shape
        
        # embed the keyval
        keyval_bev = bev_feature + self._bev_feature_embedding.weight[None, ...]
        keyval_bev = keyval_bev.contiguous().view(bs, -1, C) 
        keyval_bev = self.fusion_drop(keyval_bev)

        status_encoding = self._status_encoding(status_feature)
        keyval_status = status_encoding[:, None] + self._status_embedding.weight[None, ...]
        keyval_status = self.state_drop(keyval_status)

        keyval = torch.cat([keyval_bev, keyval_status], dim=1)

        # get the query
        agents_query = self._query_embedding.weight[None, ...].repeat(bs, 1, 1)
        query_v_first = None
        cross_query_v_first = None
        for block in self._rwkv_decoder:
            agents_query, _, past_query, past_key_values, query_v_first, cross_query_v_first = block(
                agents_query, keyval, 
                query_v_first, cross_query_v_first,
                past_query=past_query, 
                past_key_values=past_key_values, 
                use_cache=use_cache
            )
        agents_query = self.ln_out(agents_query)

        # obtain the cross bev feature
        concat_cross_bev = bev_feature.permute(0, 2, 1).contiguous().view(-1, C, concat_cross_bev_shape[0], concat_cross_bev_shape[1])  # [bs * seq_len, C, H, W]

        # upsample to the same shape as bev_feature_upscale
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)

        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, bev_feature_upscale], dim=1)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))  # [bs * seq_len, T_expand, C]
        expanded_bs, T, C = cross_bev_feature.shape
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(expanded_bs, C, bev_spatial_shape[0], bev_spatial_shape[1])

        # down sample the cross bev feature
        cross_bev_feature = F.interpolate(
            cross_bev_feature, 
            size=(self._config.bev_fuse_height, self._config.bev_fuse_width), 
            mode='bilinear', align_corners=False
        )
        cross_bev_feature = cross_bev_feature.flatten(-2, -1).permute(0, 2, 1).contiguous().view(bs, -1, C)

        # get the last frame of bev_feature_upscale 
        expanded_bs, C, H, W = bev_feature_upscale.shape
        bev_feature_upscale = bev_feature_upscale.view(bs, -1, C, H, W)
        bev_feature_upscale = bev_feature_upscale[:, -1]
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        trajectory = self._trajectory_head(
            agents_query, cross_bev_feature, targets=targets, global_img=None
        )
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output
    

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=8, if_zeroinit_reg=True, use_double_scorer=False):
        super().__init__()  

        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.use_double_scorer = use_double_scorer

        self.plan_score_branch = MLP(embed_dims + ego_fut_ts * 3, embed_dims, 6, bias=0.01)
        if use_double_scorer:
            self.plan_score_branch2 = MLP(embed_dims + ego_fut_ts * 3, embed_dims, 6, bias=0.5)
        
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )

        self.agent_col_branch = MLP(embed_dims + ego_fut_ts * 3, embed_dims, 2 * 40 * 9, bias=0.01)
        self.area_branch = MLP(embed_dims + ego_fut_ts * 3, embed_dims, 40 * 2, bias=0.01)

        self.if_zeroinit_reg = False
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

    def forward(self, traj_feature):
        bs, ego_fut_mode, _ = traj_feature.shape

        # get the denoised trajectories and classification scores
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

        traj_delta_clone = traj_delta.clone().detach()
        concat_traj_feature = torch.cat([traj_feature, traj_delta_clone], dim=-1)

        plan_score = self.plan_score_branch(concat_traj_feature)
        plan_score2 = None
        pred_agent_col = self.agent_col_branch(concat_traj_feature).reshape(bs, ego_fut_mode, self.ego_fut_ts, -1, 2, 9)
        pred_area = self.area_branch(concat_traj_feature)

        if self.use_double_scorer:
            plan_score2 = self.plan_score_branch2(concat_traj_feature)

        return plan_reg, plan_score, plan_score2, pred_agent_col, pred_area


class ModulationLayer(nn.Module):
    def __init__(self, embed_dims: int, condition_dims: int):
        super().__init__()

        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(self, traj_feature, time_embed, global_cond=None, global_img=None):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
    
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # encode the diffusion timestep information
        traj_feature = traj_feature * (1 + scale) + shift

        return traj_feature


class RWKV_DiffusionDeCoderLayer(nn.Module):
    def __init__(self, idx, num_poses, config: RWKVConfig) -> None:
        super().__init__()
        self.config = config

        self.cross_bev_attention = RWKV7_DoubleSelfAttnBlock(
            hidden_size=config.rwkv_d_model,
            norm_first=config.rwkv_d_norm_first,
            num_heads=config.rwkv_d_num_head,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            fuse_norm=config.fuse_norm,
            n_layer=config.diffusion_decoder_layer_nums * 2,
            layer_idx=idx * 2
        )
        
        self.cross_agent_attention = RWKV7_DoubleSelfAttnBlock(
            hidden_size=config.rwkv_d_model,
            norm_first=config.rwkv_d_norm_first,
            num_heads=config.rwkv_d_num_head,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            fuse_norm=config.fuse_norm,
            n_layer=config.diffusion_decoder_layer_nums * 2,
            layer_idx=idx * 2 + 1
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.rwkv_d_model, config.rwkv_d_ffn),
            nn.ReLU(),
            nn.Linear(config.rwkv_d_ffn, config.rwkv_d_model)
        )

        self.norm1 = nn.LayerNorm(config.rwkv_d_model)
        self.norm2 = nn.LayerNorm(config.rwkv_d_model)
        self.norm3 = nn.LayerNorm(config.rwkv_d_model)
        
        self.time_modulation = ModulationLayer(config.rwkv_d_model, config.rwkv_d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.rwkv_d_model,
            ego_fut_ts=num_poses,
            use_double_scorer=config.use_double_scorer,
        )

    def forward(
            self, traj_feature, noisy_traj_points, bev_feature, 
            agents_query, time_embed, v_first=None, global_img=None
        ):
        # agent cross attention
        traj_feature, _, _, _, v_first, _ = self.cross_bev_attention(traj_feature, bev_feature, v_first)
        traj_feature = self.norm1(traj_feature)

        # bev cross attention
        traj_feature, _, _, _, v_first, _ = self.cross_agent_attention(traj_feature, agents_query, v_first)
        traj_feature = self.norm2(traj_feature)

        # feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))

        # modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)
        
        # predict the offset & heading
        poses_reg, poses_score, poses_score2, pred_agent_col, pred_area = self.task_decoder(traj_feature) # [bs, 20, 8, 3]; [bs, 20]
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_score, poses_score2, pred_agent_col, pred_area, v_first


class RWKV_DiffusionDecoder(nn.Module):
    def __init__(self, num_poses, config: RWKVConfig) -> None:
        super().__init__()
        # torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        self.layers = nn.ModuleList(
            [
                RWKV_DiffusionDeCoderLayer(
                    idx=i,
                    num_poses=num_poses,
                    config=config
                ) for i in range(config.diffusion_decoder_layer_nums)
            ]
        )

    def forward(
        self, traj_feature, noisy_traj_points, bev_feature, 
        agents_query, time_embed, global_img=None
    ):  
        poses_reg_list = []
        poses_score_list = []
        poses_score_list2 = []
        traj_points = noisy_traj_points
        pred_agent_col = None
        pred_area = None

        v_first = None
        for layer in self.layers:
            poses_reg, poses_score, poses_score2, pred_agent_col, pred_area, v_first = layer(
                traj_feature, traj_points, bev_feature, 
                agents_query, time_embed, v_first, global_img
            )
            poses_reg_list.append(poses_reg)
            poses_score_list.append(poses_score)
            if poses_score2 is not None:
                poses_score_list2.append(poses_score2)
            traj_points = poses_reg[...,:2].clone().detach()

        return poses_reg_list, poses_score_list, poses_score_list2, pred_agent_col, pred_area

    
class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, config: RWKVConfig):
        """
        Initializes trajectory head.
        Args:
            num_poses: number of (x, y, θ) poses to predict
            d_ffn: dimensionality of feed-forward network
            d_model: input dimensionality
        """
        super().__init__()

        self._num_poses = num_poses
        self._d_model = config.rwkv_d_model
        self.diff_loss_weight = 2.0

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        
        # shape [20, 8, 2]
        plan_anchor = np.load(config.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) 
        self.plan_anchor_encoder = MLP(512, self._d_model, self._d_model)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self._d_model),
            nn.Linear(self._d_model, self._d_model * 4),
            nn.Mish(),
            nn.Linear(self._d_model * 4, self._d_model),
        )

        self.diff_decoder = RWKV_DiffusionDecoder(num_poses, config)
        
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2 * (odo_info_fut_x + 1.2) / 56.9 - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + 20) / 46 - 1
        odo_info_fut_head = 2 * (odo_info_fut_head + 2) / 3.9 - 1
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1) / 2 * 3.9 - 2
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def forward(self, agents_query, bev_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        """ Torch module forward pass """
        if self.training:
            return self.forward_train(agents_query, bev_feature, targets, global_img)
        else:
            return self.forward_test(agents_query, bev_feature, global_img)
        
    def forward_train(self, agents_query, bev_feature, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        """ Torch module forward pass during training """
        bs = agents_query.shape[0]
        device = agents_query.device
        
        # add the truncated noise to the plan anchor, [bs, 20, 8, 2]
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        odo_info_fut = self.norm_odo(plan_anchor)
        noise = torch.randn(odo_info_fut.shape, device=device)
        
        timesteps = torch.randint(0, 50, (bs,), device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        # proj noisy_traj_points to the query
        ego_fut_mode = noisy_traj_points.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64) # [bs, ego_fut_mode, 8, 64]
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        
        # embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)
      
        # denoise the trajectory, get the regression trajectories and pdm scores
        poses_reg_list, poses_score_list, poses_score_list2, pred_agent_col, pred_area = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, 
            agents_query, time_embed, global_img
        )

        trajectory_candidates = []
        for poses_reg in poses_reg_list:
            _, _, ts, d = poses_reg.shape
            target_traj = targets["trajectory"]
            dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1) # [bs, 20, 8]
            dist = dist.mean(dim=-1) # [bs, 20]
            mode_idx = torch.argmin(dist, dim=-1) # [bs,]
            mode_idx = mode_idx[..., None, None, None].repeat(1, 1, ts, d) # [bs, 1, 8, 3]
            best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1) # [bs, 8, 3]
            trajectory_candidates.append(best_reg)

        pdm_score = torch.sigmoid(poses_score_list[-1])[:,:,-1]
        if len(poses_score_list2) > 0 and poses_score_list2[-1] is not None:
            pdm_score += torch.sigmoid(poses_score_list2[-1])[:,:,-1]
            pdm_score /= 2.0
        best_pdm_score = pdm_score.amax(dim=-1)
        average_pdm_score = pdm_score.mean(dim=-1)

        mode_idx = pdm_score.argmax(dim=-1) # (bs, )
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3) # (bs, 1, 8, 3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1) # (bs, 8, 3)

        return {"trajectory": best_reg, "best_trajectory": trajectory_candidates, "denoised_trajectories": poses_reg_list[-1], "score_logit": poses_score_list[-1], "score_logit2": poses_score_list2[-1] if poses_score_list2 else None, "best_pdm_score": best_pdm_score, "average_pdm_score": average_pdm_score, "pred_agent_col": pred_agent_col, "pred_area": pred_area}
    
    def forward_test(self, agents_query, bev_feature, global_img) -> Dict[str, torch.Tensor]:
        """ Torch module forward pass during inference """
        bs = agents_query.shape[0]
        device = agents_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        
        step_num = 2
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        norm_traj_points = self.norm_odo(plan_anchor)
        
        noise = torch.randn(norm_traj_points.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        noisy_norm_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=norm_traj_points, noise=noise, timesteps=trunc_timesteps)
        
        ego_fut_mode = noisy_norm_traj_points.shape[1]     
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(noisy_norm_traj_points, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)
            
            # proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU, so try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=noisy_norm_traj_points.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(noisy_norm_traj_points.device)
            
            # embed the timesteps
            timesteps = timesteps.expand(noisy_norm_traj_points.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            # denoise the trajectory
            poses_reg_list, poses_score_list, poses_score_list2, pred_agent_col, pred_area = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, 
                agents_query, time_embed, global_img
            )
            poses_reg = poses_reg_list[-1]
            poses_score = poses_score_list[-1]
            poses_score2 = poses_score_list2[-1] if poses_score_list2 else None

            # refine the denoised trajectories
            # pdms_score = poses_score[..., 5]
            # if poses_score2 is not None:
            #     pdms_score = (pdms_score + poses_score2[..., 5]) / 2.0

            # bs, traj_num, ts, d = poses_reg.shape
            # top_k = np.ceil(traj_num * 0.2).astype(int)

            # topk_values, topk_indices = torch.topk(pdms_score, k=top_k, dim=1, largest=True)
            # bottomk_values, bottomk_indices = torch.topk(pdms_score, k=top_k, dim=1, largest=False)

            # topk_indices_exp = topk_indices.view(bs, top_k, 1, 1).expand(-1, -1, ts, d)
            # topk_traj = torch.gather(poses_reg, dim=1, index=topk_indices_exp)
            
            # bottomk_indices_exp = bottomk_indices.view(bs, top_k, 1, 1).expand(-1, -1, ts, d)
            # poses_reg_repl = poses_reg.clone()
            # poses_reg_repl.scatter_(dim=1, index=bottomk_indices_exp, src=topk_traj)
            # poses_reg = poses_reg_repl
                       
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            
            noisy_norm_traj_points = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=noisy_norm_traj_points
            ).prev_sample

        pdm_score = torch.sigmoid(poses_score)[:,:,-1]
        if poses_score2 is not None:
            pdm_score += torch.sigmoid(poses_score2)[:,:,-1]
            pdm_score /= 2.0
        best_pdm_score = pdm_score.amax(dim=-1)
        average_pdm_score = pdm_score.mean(dim=-1)
            
        mode_idx = pdm_score.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)

        return {"trajectory": best_reg, "denoised_trajectories": poses_reg, "score_logit": poses_score, "score_logit2": poses_score2, "best_pdm_score": best_pdm_score, "average_pdm_score": average_pdm_score, "pred_agent_col": pred_agent_col, "pred_area": pred_area}


if  __name__ == "__main__":
    import os
    cfg = RWKVConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 10
    camera_data = torch.rand((seq_len, 3, 256, 1024)).to(device)
    status_data = torch.rand((seq_len, 8)).to(device)

    total_features = {
        "camera_feature": camera_data,
        "status_feature": status_data[-1]
    }
    total_features = {k: v.unsqueeze(0) for k, v in total_features.items()}

    model = RWKVModel(cfg).to(device)
    model.eval()

    model.forward(total_features, targets=None)
