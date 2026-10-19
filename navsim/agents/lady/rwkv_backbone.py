"""
Implements the RWKV7 vision backbone.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.rwkv_block.lady_block import RWKV7Block
from navsim.agents.lady.rwkv_config import RWKVConfig


class RWKVBackbone(nn.Module):
    """Multi-scale Fusion RWKV7 Model for image + LiDAR feature fusion."""

    def __init__(self, config: RWKVConfig):

        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = 1

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    (1, in_channels, config.lidar_resolution_width, config.lidar_resolution_height),
                    requires_grad=True,
                )
            )

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        lidar_time_frames = [config.lidar_seq_len for i in range(4)]

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.rwkvs = nn.ModuleList(
            [
                GPT_RWKV(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    config=config,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
        # Typical encoders down-sample by a factor of 32
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        if self.config.transformer_decoder_join:
            self.num_features = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
        else:
            if self.config.add_features:
                self.lidar_to_img_features_end = nn.Linear(
                    self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"],
                )
                # Number of features the encoder produces.
                self.num_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
            else:
                # Number of features the encoder produces.
                self.num_features = (
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
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
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"], channel, (1, 1))

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, lidar):
        """
        Image + LiDAR feature fusion using rwkvs
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        image_features, lidar_features = image, lidar

        if self.config.latent:
            batch_size = lidar.shape[0]
            lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        # Loop through the 4 blocks of the network.
        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            x4 = lidar_features

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features

        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features = self.global_pool_img(image_features)
            image_features = torch.flatten(image_features, 1)
            lidar_features = self.global_pool_lidar(lidar_features)
            lidar_features = torch.flatten(lidar_features, 1)

            if self.config.add_features:
                lidar_features = self.lidar_to_img_features_end(lidar_features)
                fused_features = image_features + lidar_features
            else:
                fused_features = torch.cat((image_features, lidar_features), dim=1)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(x4)
        else:
            features = None

        return features, fused_features, image_feature_grid

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

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """
        Perform a feature fusion block using a RWKV module.
        :param image_features: Features from the image branch
        :param lidar_features: Features from the LiDAR branch
        :param layer_idx: RWKV layer index.
        :return: image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.rwkvs[layer_idx](image_embd_layer, lidar_embd_layer)
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT_RWKV(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""

    def __init__(self, n_embd, config: RWKVConfig, lidar_time_frames):
        super().__init__()

        self.n_embd = n_embd
        self.config = config

        self.seq_len = config.cam_seq_len
        self.lidar_seq_len = config.lidar_seq_len
        self.lidar_time_frames = lidar_time_frames

        self.img_feature_size = config.img_vert_anchors * config.img_horz_anchors
        self.lidar_feature_size = config.lidar_vert_anchors * config.lidar_horz_anchors

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, self.img_feature_size + self.lidar_feature_size, self.n_embd))

        #  embedding parameter (learnable), sequence length
        self.time_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        # RWKV Blocks
        self.blocks = nn.ModuleList(
            [ 
                RWKV7Block(
                    hidden_size=n_embd,
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    norm_first=config.norm_first,
                    time_pdrop=config.time_pdrop,
                    channel_pdrop=config.channel_pdrop,
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

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B * seq_len, C, H, W
            lidar_tensor (tensor): B * seq_len, C, H, W
        """
        expanded_batch_size = image_tensor.shape[0]
        batch_size = expanded_batch_size // self.seq_len
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(expanded_batch_size, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(expanded_batch_size, -1, self.n_embd)
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1).contiguous()
        token_embeddings = token_embeddings.view(batch_size, -1, self.n_embd)

        embedding = self.pos_emb.unsqueeze(1) + self.time_emb.unsqueeze(2)
        embedding = embedding.view(1, -1, self.n_embd)

        x = self.drop(embedding + token_embeddings) # (B, seq_len * T, C), T = img_ver * img_horz + lidar_ver * lidar_horz
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_f(x)   

        # restore the original shape
        x = x.view(expanded_batch_size, -1, self.n_embd)
        image_tensor_out = (
            x[:, :self.img_feature_size, :]
            .view(expanded_batch_size, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        lidar_tensor_out = (
            x[:, self.img_feature_size :, :]
            .view(expanded_batch_size, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out
