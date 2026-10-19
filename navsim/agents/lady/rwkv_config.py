from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import os
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class RWKVConfig:
    """LADY multi-frame configuration for the default RWKV-7 implementation."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    base_path: str = os.getenv('NAVSIM_DEVKIT_ROOT', str(Path(__file__).resolve().parents[3]))
    use_100_anchor = os.getenv('USE_100_ANCHOR', 'true').lower() == 'true'
    if use_100_anchor:
        plan_anchor_path: str = os.path.join(base_path, "plan_anchors_100.npy")
        print("Using 100 anchors")
    else:
        plan_anchor_path: str = os.path.join(base_path, "plan_anchors.npy")
        print("Using 20 anchors")

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    cam_seq_len: int = 10
    lidar_seq_len: int = 10
    model_seq_len: int = 10

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    bev_fuse_height: int = 16
    bev_fuse_width: int = 16

    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True
    
    # diffusion decoder
    diffusion_decoder_layer_nums: int = 2

    # group learning rate 
    betas = (0.9, 0.99)
    weight_decay: float = 0.01
    adam_eps: float = 1e-18

    # rwkv decoder
    rwkv_d_model: int = 256
    rwkv_d_ffn: int = 1024
    rwkv_d_hidden_ratio: int = 4.0
    rwkv_d_num_layers: int = 3
    rwkv_d_num_head: int = 4
    rwkv_d_norm_first: bool = True
    rwkv_d_time_pdrop: float = 0.0
    rwkv_d_channel_pdrop: float = 0.0
    rwkv_fusion_drop: float = 0.05
    rwkv_state_drop: float = 0.1
    tf_dropout: float = 0.0

    # rwkv encoder
    fuse_norm: bool = True
    block_exp = 4
    hidden_size: int = 2048
    num_heads: int = 4
    norm_first: bool = True
    decay_low_rank_dim: int = 64
    gate_low_rank_dim: int = 128
    a_low_rank_dim: int = 64
    v_low_rank_dim: int = 32
    num_layers = 2  # Number of rwkv layers of each GPT used in the vision backbone
    embd_pdrop = 0.1
    time_pdrop = 0.0
    channel_pdrop = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_sub_score_weight = 5.0
    trajectory_final_score_weight = 5.0
    trajectory_reg_weight = 8.0

    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    agent_ce_weight: float = 1.0
    agent_l1_weight: float = 0.1
    area_weight: float = 2.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    use_ray = True
    use_gt_eval = False
    use_double_scorer = False

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
