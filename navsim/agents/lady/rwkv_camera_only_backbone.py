"""
Implements the RWKV7 vision backbone.
"""

from typing import Optional

import timm
import torch
import torch.nn.functional as F
from torch import nn

from fla.models.utils import Cache

from navsim.agents.rwkv_block.lady_block_fla import RWKV7Block
from navsim.agents.lady.rwkv_config import RWKVConfig


class RWKVBackbone(nn.Module):
    """Multi-scale Fusion RWKV7 Model for image feature fusion."""

    def __init__(self, config: RWKVConfig):

        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        if config.use_ground_plane:
            in_channels = 2 * config.cam_seq_len
        else:
            in_channels = 1

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.adptive_num_heads = [4, 4, 8, 8]
        self.rwkvs = nn.ModuleList(
            [
                GPT_RWKV(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    num_heads=self.adptive_num_heads[i],
                    config=config,
                )
                for i in range(4)
            ]
        )

        # FPN fusion
        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # top down
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.camera_height // self.config.bev_down_sample_factor,
                    self.config.camera_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(self.image_encoder.feature_info.info[start_index + 3]["num_chs"], channel, (1, 1))

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, past_features=[None, None, None, None], use_cache=False, frame=None):
        """
        Process Image feature using rwkvs
        Args:
            image_list (list): list of input images
            past_features (Optional[List[Cache]]): Cache for past features
            use_cache (Optional[bool]): Whether to use cache for past features
        """
        image_features = image

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())

        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

        # Loop through the 4 blocks of the network.
        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            image_features, past_features[i] = self.fuse_features(image_features, i, past_features[i], use_cache, frame)

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features
         
        features = self.top_down(image_features)
    
        return features, image_features, image_feature_grid, past_features

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, layer_idx, past_feature=None, use_cache=False, frame=None):
        """
        Perform a feature fusion block using a RWKV module.
        :param image_features: Features from the image branch
        :param layer_idx: RWKV layer index.
        :return: image_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        image_features_layer, past_feature = self.rwkvs[layer_idx](image_embd_layer, past_feature, use_cache, frame)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer

        return image_features, past_feature


class GPT_RWKV(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""
    def __init__(self, n_embd, num_heads, config: RWKVConfig):
        super().__init__()

        self.n_embd = n_embd
        self.config = config
        self.seq_len = config.cam_seq_len

        self.img_feature_size = config.img_vert_anchors * config.img_horz_anchors
        self.pos_emb = nn.Parameter(torch.zeros(1, self.img_feature_size, self.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        # RWKV Blocks
        self.blocks = nn.ModuleList(
            [ 
                RWKV7Block(
                    hidden_size=n_embd,
                    norm_first=config.norm_first,
                    num_heads=num_heads,
                    decay_low_rank_dim=config.decay_low_rank_dim,
                    gate_low_rank_dim=config.gate_low_rank_dim,
                    a_low_rank_dim=config.a_low_rank_dim,
                    v_low_rank_dim=config.v_low_rank_dim,
                    fuse_norm=config.fuse_norm,
                    n_layer=config.num_layers,
                    layer_idx=layer_idx
                ) 
                for layer_idx in range(config.num_layers)
            ]
        )
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(
            self, image_tensor, past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False, 
            frame: Optional[torch.Tensor] = None
        ):
        """
        Args:
            image_tensor (tensor): B * seq_len, C, H, W
            past_key_values (Optional[Cache]): cache for rwkv
            use_cache (Optional[bool]): whether to use cache for rwkv
        """
        expanded_batch_size = image_tensor.shape[0]
        batch_size = expanded_batch_size // self.seq_len

        img_h, img_w = image_tensor.shape[2:4]
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(expanded_batch_size, -1, self.n_embd)

        x = self.drop(self.pos_emb + image_tensor) # (B * seq_len, T, C)
        x = x.view(batch_size, -1, self.n_embd)
        
        v_first = None
        attention_mask = None
        if frame is not None:
            feature_length = self.img_feature_size

            # padding useless frames
            start_indices = (self.seq_len - frame) * feature_length
            start_indices = start_indices.unsqueeze(1) 
            range_tensor = torch.arange(feature_length * self.seq_len, device=x.device).repeat(batch_size, 1)
            attention_mask = (range_tensor >= start_indices).int()
        
        for block in self.blocks:
            x, _, past_key_values, v_first = block(x, attention_mask, v_first, past_key_values, use_cache)
        x = self.ln_f(x)   

        # restore the original shape
        x = x.view(expanded_batch_size, -1, self.n_embd)
        image_tensor_out = x.view(expanded_batch_size, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()
        
        return image_tensor_out, past_key_values
