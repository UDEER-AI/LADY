from typing import Optional, Tuple

import torch
import torch.nn as nn  

from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward
from fla.modules import LayerNorm
from fla.models.utils import Cache

# from navsim.agents.rwkv_block.lady_attention_fla import RWKV7Attention, RWKV7_CrossAttention
from navsim.agents.rwkv_block.lady_attention_fla_initial import RWKV7Attention, RWKV7_CrossAttention


class RWKV7Block(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
            norm_bias: bool = True,
            norm_eps: float = 1e-5,
            attn_mode: str = "chunk",
            num_heads: int = 4,
            head_dim: int = None,
            decay_low_rank_dim: int = 64,
            gate_low_rank_dim: int = 128,
            a_low_rank_dim: int = 64,
            v_low_rank_dim: int = 32,
            fuse_norm: bool = True,
            hidden_ratio: Optional[int] = 4.0,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "sqrelu",
            n_layer: int = 3,
            layer_idx: int = None,
        ) -> None:
        super().__init__()    
        self.fuse_norm = fuse_norm

        if norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
                hidden_size,
                bias=norm_bias,
                eps=norm_eps
            )
        
        self.attn_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.ffn_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.attn = RWKV7Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            n_layer=n_layer,
            layer_idx=layer_idx,
            fuse_norm=fuse_norm
        )

        self.ffn = RWKV7FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=n_layer
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        v_first: torch.Tensor = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states

        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, past_key_values, v_first = self.attn(
            hidden_states=hidden_states,
            v_first=v_first,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        if self.fuse_norm:
            hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
        
        hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, state=past_key_values)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, v_first)

        return outputs
    

class RWKV7_DoubleSelfAttnBlock(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
            norm_bias: bool = True,
            norm_eps: float = 1e-5,
            attn_mode: str = "chunk",
            num_heads: int = 4,
            head_dim: int = None,
            decay_low_rank_dim: int = 64,
            gate_low_rank_dim: int = 128,
            a_low_rank_dim: int = 64,
            v_low_rank_dim: int = 32,
            fuse_norm: bool = True,
            hidden_ratio: Optional[int] = 4.0,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "sqrelu",
            n_layer: int = 3,
            layer_idx: int = None
        ) -> None:
        super().__init__()    
        self.fuse_norm = fuse_norm

        if norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
                hidden_size,
                bias=norm_bias,
                eps=norm_eps
            )

        self.qkv_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.ffn_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.query_attn = RWKV7Block(
            hidden_size=hidden_size,
            norm_first=norm_first,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            attn_mode=attn_mode,
            num_heads=num_heads,
            head_dim=head_dim,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            fuse_norm=fuse_norm,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            n_layer= n_layer * 2,
            layer_idx=layer_idx * 2
        )

        self.qkv_self_attn = RWKV7Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            n_layer=n_layer * 2,
            layer_idx=layer_idx * 2 + 1,
            fuse_norm=fuse_norm
        )

        self.ffn = RWKV7FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx * 2 + 1,
            num_hidden_layers=n_layer * 2
        )

    def forward(
        self, query, keyval, query_v_first=None, qkv_v_first=None, frames=None,
        past_query=None, past_key_values=None, use_cache=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Cache], Optional[torch.Tensor], Optional[torch.Tensor]]:
        query, self_attentions, past_query, query_v_first = self.query_attn(
            hidden_states=query, 
            v_first=query_v_first, 
            past_key_values=past_query,
            use_cache=use_cache
        )
        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query

        B, query_len, _ = query.size()
        if frames is not None:
            keyval_padding = torch.zeros_like(query, device=query.device)
            qkv = torch.cat([keyval, keyval_padding], dim=1)

            # put query in the right place
            frame_indices = frames * 64 + 1
            qkv[torch.arange(B).unsqueeze(1), frame_indices.unsqueeze(1) + torch.arange(query_len).cuda().unsqueeze(0), :] = residual

            qkv_attn, cross_attentions, past_key_values, query_v_first = self.qkv_self_attn(
                hidden_states=self.qkv_norm(qkv),
                v_first=query_v_first,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            query_attn = qkv_attn[torch.arange(B).unsqueeze(1), frame_indices.unsqueeze(1) + torch.arange(query_len).cuda().unsqueeze(0), :]
        else:
            qkv = torch.cat([keyval, residual], dim=1)
            qkv_attn, cross_attentions, past_key_values, query_v_first = self.qkv_self_attn(
                hidden_states=self.qkv_norm(qkv),
                v_first=query_v_first,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            query_attn = qkv_attn[:, -query_len:, :]

        if self.fuse_norm:
            hidden_states, residual = self.ffn_norm(query_attn, residual, True)
        else:
            hidden_states = query_attn + residual
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)

        hidden_states, past_query = self.ffn(hidden_states, state=past_query)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, cross_attentions, past_query, past_key_values, query_v_first, qkv_v_first)

        return outputs


class RWKV7_CustomCrossAttnBlock(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
            norm_bias: bool = True,
            norm_eps: float = 1e-5,
            attn_mode: str = "chunk",
            num_heads: int = 4,
            head_dim: int = None,
            decay_low_rank_dim: int = 64,
            gate_low_rank_dim: int = 128,
            a_low_rank_dim: int = 64,
            v_low_rank_dim: int = 32,
            fuse_norm: bool = True,
            hidden_ratio: Optional[int] = 4.0,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "sqrelu",
            n_layer: int = 3,
            layer_idx: int = None
        ) -> None:
        super().__init__()    
        self.fuse_norm = fuse_norm
        self.layer_idx = layer_idx

        if norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
                hidden_size,
                bias=norm_bias,
                eps=norm_eps
            )

        self.query_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.kv_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.ffn_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.query_attn = RWKV7Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            n_layer=n_layer,
            layer_idx=layer_idx,
            fuse_norm=fuse_norm
        )

        self.kv_attn = RWKV7Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            n_layer=n_layer,
            layer_idx=layer_idx,
            fuse_norm=fuse_norm
        )

        self.ffn = RWKV7FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=n_layer
        )

    def forward(self, query, keyval, query_v_first=None, kv_v_first=None, frames=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # derive the recurrent state of the keyval
        use_cache = True
        past_key_values = Cache()

        _, _, past_key_values, kv_v_first = self.kv_attn(
            hidden_states=self.kv_norm(keyval),
            v_first=kv_v_first,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        if len(past_key_values) <= self.layer_idx:
            last_state = past_key_values[len(past_key_values) - 1]
            recurrent_state = last_state['recurrent_state']
            conv_state = last_state['conv_state']

            for i in range(len(past_key_values), self.layer_idx + 1):
                past_key_values.update(
                    recurrent_state=recurrent_state,
                    conv_state=conv_state,
                    layer_idx=i,
                    offset=keyval.size(1)
                )
    
        # use the recurrent state as the initial state for each query
        query_len = query.size(1)
        query_attn = torch.zeros_like(query, device=query.device)

        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query
        query = self.query_norm(residual)
        
        if query_v_first is None:
            query_v_first = torch.zeros_like(query, device=query.device)
            for i in range(query_len):
                 query_attn[:, i:i+1, :], _, past_key_values, query_v_first[:, i:i+1, :] = self.query_attn(
                    hidden_states=query[:, i:i+1, :],
                    v_first=None, 
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    update_cache=False
                )
        else:
            for i in range(query_len):
                 query_attn[:, i:i+1, :], _, past_key_values, query_v_first[:, i:i+1, :] = self.query_attn(
                    hidden_states=query[:, i:i+1, :],
                    v_first=query_v_first[:, i:i+1, :], 
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    update_cache=False
                )

        if self.fuse_norm:
            hidden_states, residual = self.ffn_norm(query_attn, residual, True)
        else:
            hidden_states = query_attn + residual
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)

        hidden_states, _ = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, query_v_first, kv_v_first


class RWKV7_CrossAttBlock(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
            norm_bias: bool = True,
            norm_eps: float = 1e-5,
            attn_mode: str = "chunk",
            num_heads: int = 4,
            head_dim: int = None,
            decay_low_rank_dim: int = 64,
            gate_low_rank_dim: int = 128,
            a_low_rank_dim: int = 64,
            v_low_rank_dim: int = 32,
            fuse_norm: bool = True,
            hidden_ratio: Optional[int] = 4.0,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "sqrelu",
            n_layer: int = 3,
            layer_idx: int = None,
        ) -> None:
        super().__init__()    
        self.fuse_norm = fuse_norm

        if norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
                hidden_size,
                bias=norm_bias,
                eps=norm_eps
            )
        
        self.query_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.keyval_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.ffn_norm = (LayerNorm if fuse_norm else nn.LayerNorm)(
            hidden_size,
            bias=norm_bias,
            eps=norm_eps
        )

        self.query_attn = RWKV7Block(
            hidden_size=hidden_size,
            norm_first=norm_first,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            attn_mode=attn_mode,
            num_heads=num_heads,
            head_dim=head_dim,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            fuse_norm=fuse_norm,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            n_layer=n_layer,
            layer_idx=layer_idx
        )

        self.cross_attn = RWKV7_CrossAttention(
            mode=attn_mode,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            n_layer=n_layer,
            layer_idx=layer_idx,
            fuse_norm=fuse_norm
        )

        self.ffn = RWKV7FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx
        )

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        query_v_first: torch.Tensor,
        cross_query_v_first: torch.Tensor,
        frames: int = None,
        past_query: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Cache], Optional[torch.Tensor], Optional[torch.Tensor]]:
        query, _, past_query, query_v_first = self.query_attn(
            hidden_states=query, 
            v_first=query_v_first, 
            past_key_values=past_query,
            use_cache=use_cache
        )
        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query

        hidden_states, cross_attentions, past_key_values, cross_query_v_first = self.cross_attn(
            query=self.query_norm(residual),
            keyval=self.keyval_norm(keyval),
            v_first=cross_query_v_first,
            frames=frames,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        if self.fuse_norm:
            hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
        
        hidden_states, past_query = self.ffn(hidden_states, state=past_query)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, cross_attentions, past_query, past_key_values, query_v_first, cross_query_v_first)

        return outputs
    

def self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.ModuleList(
        [
            RWKV7Block(
                hidden_size=256,
                num_heads=4,
                layer_idx=i
            )
            for i in range(2)
        ]
    )

    model.to(device)
    B = 16
    seq_len = 31

    keyval = torch.randn(B, seq_len, 256).to(device)
    v_first = None

    for layer in model:
        keyval, _, _, v_first = layer(keyval, v_first)

    print("Success!")
    print("Output Shape: ", keyval.shape)
    print("v_first Shape: ", v_first.shape)

    # add backward
    target = torch.randn_like(keyval).to(device)
    loss_fn = nn.MSELoss()
    loss = loss_fn(keyval, target)
    model.zero_grad()
    loss.backward()

def cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layer = 3
    model = nn.ModuleList([
        RWKV7_DoubleSelfAttnBlock(
            hidden_size=256,
            num_heads=4,
            layer_idx=i,
            n_layer=n_layer
        )
        for i in range(n_layer)
    ])
    model.to(device)
    B = 4
    que_seq_len = 31
    seq_len = 641

    query = torch.randn(B, que_seq_len, 256).to(device)
    keyval = torch.randn(B, seq_len, 256).to(device)

    query_v_first = None
    cross_query_v_first= None
    for layer in model:
        query, _, _, _, query_v_first, cross_query_v_first = layer(
            query, keyval, query_v_first, cross_query_v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("query_v_first Shape: ", query_v_first.shape)
    if cross_query_v_first is not None:
        print("cross_query_v_first Shape: ", cross_query_v_first.shape)

def custom_cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_layer = 3
    # model = nn.ModuleList([
    #     RWKV7_CustomCrossAttnBlock(
    #         hidden_size=256,
    #         num_heads=4,
    #         layer_idx=i,
    #         n_layer=n_layer
    #     )
    #     for i in range(n_layer)
    # ])
    model1 = RWKV7_CustomCrossAttnBlock(
        hidden_size=256,
        num_heads=4,
        layer_idx=0,
        n_layer=n_layer
    )
    model1.to(device)

    model2 = RWKV7_CustomCrossAttnBlock(
        hidden_size=256,
        num_heads=4,
        layer_idx=1,
        n_layer=n_layer
    )
    model2.to(device)

    B = 4
    que_seq_len = 20
    seq_len = 30

    query = torch.randn(B, que_seq_len, 256).to(device)
    keyval = torch.randn(B, seq_len, 256).to(device)

    query_v_first = None
    kv_v_first = None

    query, query_v_first, kv_v_first = model1(
        query, keyval, query_v_first, kv_v_first
    )

    query, query_v_first, kv_v_first = model2(
        query, keyval, query_v_first, kv_v_first
    )

    print("Success!")
    print("Output Shape: ", query.shape)
    print("query_v_first Shape: ", query_v_first.shape)
    print("kv_v_first Shape: ", kv_v_first.shape)


if __name__ == '__main__':
    # self_attn_test()
    # cross_attn_test()
    custom_cross_attn_test()
