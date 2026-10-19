import os
from typing import Any, List, Dict, Optional, Union
from pathlib import Path

import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from deepspeed.ops.adam import FusedAdam

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.lady.rwkv_config import RWKVConfig
from navsim.common.dataclasses import AgentInput, Scene, Trajectory

model_type = os.getenv("MODEL_TYPE", "default")
if model_type == "default":
    from navsim.agents.lady.rwkv_fla_diffusion_model import RWKVModel
    print("Using Default Diffusion model")
elif model_type == "camera_only":
    from navsim.agents.lady.rwkv_camera_only_model import RWKVModel
    print("Using Camera Only model")
elif model_type == "stand_cross":
    from navsim.agents.lady.rwkv_fla_standard_crossattn import RWKVModel
    print("Using Standard Cross Attention model")
else:
    raise ValueError("Model type must be default or camera_only or stand_cross")

from navsim.agents.lady.rwkv_callback import RWKVCallback
from navsim.agents.lady.rwkv_loss import RWKV_loss
from navsim.agents.lady.rwkv_features import RWKVFeatureBuilder, RWKVTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.score_module.compute_navsim_score import get_scores
from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
from nuplan.planning.utils.multithreading.worker_utils import worker_map


class LADYAgent(AbstractAgent):
    """LADY agent interface using the released RWKV-7 backend."""

    def __init__(
        self,
        config: RWKVConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes LADY agent.
        :param config: global config of the released LADY backend
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._rwkv_model = RWKVModel(config)

        metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache_training_new"))
        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        self.get_scores = get_scores

        if config.use_ray:
            self.worker = RayDistributedNoTorch(threads_per_node=16)
            self.worker_map = worker_map

        if config.use_gt_eval:
            self.requires_scene = True

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        self._rwkv_model.to(device="cuda")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[i for i in range(self._config.cam_seq_len)])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [RWKVTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [RWKVFeatureBuilder(config=self._config)]

    # def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
    #     """
    #     Computes the ego vehicle trajectory.
    #     :param current_input: Dataclass with agent inputs.
    #     :return: Trajectory representing the predicted ego's position in future
    #     """
    #     self.eval()
    #     features: Dict[str, torch.Tensor] = {}
    #     targets: Dict[str, torch.Tensor] = {}
    #     # build features
    #     for builder in self.get_feature_builders():
    #         features.update(builder.compute_features(agent_input))
        
    #     for builder in self.get_target_builders():
    #         targets.update(builder.compute_targets(scene))

    #     # add batch dimension
    #     features = {k: v.unsqueeze(0) for k, v in features.items()}
    #     for k, v in targets.items():
    #         if isinstance(v, torch.Tensor):
    #             targets[k] = v.unsqueeze(0)

    #     # forward pass
    #     with torch.no_grad():
    #         predictions = self.forward(features, targets)
    #         poses = predictions["trajectory"].squeeze(0).cpu().numpy()

    #     # extract trajectory
    #     return Trajectory(poses)

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        for k, v in features.items():
            features[k] = v.cuda()

        if targets is not None:
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.cuda()
        
        return self._rwkv_model(features, targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if isinstance(predictions["denoised_trajectories"], list):
            target_scores_list = []
            for pred in predictions["denoised_trajectories"]:
                final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
                    targets, pred)
                target_scores_list.append(target_scores)
            return RWKV_loss(targets, predictions, target_scores_list, final_scores, best_scores, gt_states, gt_valid, gt_ego_areas, self._config)
        else:
            final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
                targets, predictions["denoised_trajectories"])

            best_traj = predictions["trajectory"].unsqueeze(1)
            pred_final_pdms_scores = self.compute_pdms_best_traj(targets, best_traj)

            return RWKV_loss(targets, predictions, pred_final_pdms_scores, target_scores, final_scores, best_scores, gt_states, gt_valid, gt_ego_areas, self._config)

    def compute_score(self, targets, proposals):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        proposals=proposals.detach()
        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": False
            }
            for token, poses in zip(targets["token"], proposals.float().cpu().numpy())
        ]

        if self._config.use_ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

        key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

        all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

        return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_pdms_best_traj(self, targets, proposals):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        proposals = proposals.detach()
        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": False
            }
            for token, poses in zip(targets["token"], proposals.float().cpu().numpy())
        ]

        if self._config.use_ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]

        return final_scores

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        optimizer = torch.optim.Adam(self._rwkv_model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_epoch"
            }
        }

        # lr_decay = set()
        # lr_1x = set()
        # lr_2x = set()

        # for n, p in self.named_parameters():
        #     condition = (
        #         (".rwkvs" in n) or
        #         ("_rwkv_decoder." in n) or
        #         (".cross_agent_attention" in n) or
        #         (".cross_ego_attention" in n)
        #     )
        #     if "attn.w0" in n:
        #         lr_2x.add(n)
        #     elif (
        #         (len(p.squeeze().shape) >= 2)
        #         and (self._config.weight_decay > 0)
        #         and (".weight" in n)
        #         and condition
        #     ):
        #         lr_decay.add(n)
        #     else:
        #         lr_1x.add(n)

        # lr_decay = sorted(list(lr_decay))
        # lr_1x = sorted(list(lr_1x))
        # lr_2x = sorted(list(lr_2x))

        # param_dict = {n: p for n, p in self.named_parameters()}

        # optim_groups = [
        #     {
        #         "params": [param_dict[n] for n in lr_1x],
        #         "weight_decay": 0.0,
        #         "my_lr_scale": 1.0,
        #     },
        #     {
        #         "params": [param_dict[n] for n in lr_2x],
        #         "weight_decay": 0.0,
        #         "my_lr_scale": 2.0,
        #     },
        # ]

        # if self._config.weight_decay > 0:
        #     optim_groups += [
        #         {
        #             "params": [param_dict[n] for n in lr_decay],
        #             "weight_decay": self._config.weight_decay,
        #             "my_lr_scale": 1.0,
        #         }
        #     ]
        #     return FusedAdam(
        #         optim_groups,
        #         lr=self._lr,
        #         betas=self._config.betas,
        #         eps=self._config.adam_eps,
        #         bias_correction=True,
        #         adam_w_mode=True,
        #         amsgrad=False,
        #     )
        # else:
        #     return FusedAdam(
        #         optim_groups,
        #         lr=self._lr,
        #         betas=self._config.betas,
        #         eps=self._config.adam_eps,
        #         bias_correction=True,
        #         adam_w_mode=False,
        #         weight_decay=0,
        #         amsgrad=False,
        #     )

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        best_model_checkpoint = ModelCheckpoint(
            monitor='val/loss_epoch',               
            mode='min',                   
            save_top_k=1,                      
            filename='best_{epoch}',  
            save_weights_only=False            
        )
        every_5_epoch_checkpoint = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=5,
            filename='{epoch}_{step}',
            save_weights_only=False
        )

        return [RWKVCallback(self._config), best_model_checkpoint, every_5_epoch_checkpoint]
    
    def params_test1(self):
        for n, p in self.named_parameters():
            if "attn.w0" in n:
                print(f"2x lr: {n}")

    def params_test2(self):
        for n, p in self.named_parameters():
            if (len(p.squeeze().shape) >= 2) and (self._config.weight_decay > 0) and (".weight" in n):
                print(f"decay: {n}")

    def params_test3(self):
        for n, p in self.named_parameters():
            condition = (
                (".rwkvs" in n) or
                ("_rwkv_decoder." in n) or
                (".cross_agent_attention" in n) or
                (".cross_ego_attention" in n)
            )
            if (len(p.squeeze().shape) >= 2) and (self._config.weight_decay > 0) and (".weight" in n) and condition:
                print(f"decay: {n}")

    def params_test4(self):
        lr_decay = []
        lr_2x = []

        for n, p in self.named_parameters():
            condition = (
                (".rwkvs" in n) or
                ("_rwkv_decoder." in n) or
                (".cross_agent_attention" in n) or
                (".cross_ego_attention" in n)
            )
            if "attn.w0" in n:
                lr_2x.append(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (self._config.weight_decay > 0)
                and (".weight" in n)
                and condition
            ):
                lr_decay.append(n)

        print("lr_decay:", lr_decay)
        print('----------')
        print("lr_2x:", lr_2x)


if __name__ == "__main__":
    # Example usage
    config = RWKVConfig()
    agent = LADYAgent(config=config, lr=0.001)

    # agent.params_test1()
    print("----")
    agent.params_test4()
