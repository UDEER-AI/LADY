# this is a old version of the rwkv crossattention block, which is now replaced by the rwkv6_block.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import LayerNorm
from fla.models.utils import Cache
from fla.modules.activations import ACT2FN
from fla.layers.rwkv6 import RWKV6Attention
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from fla.models.rwkv6.modeling_rwkv6 import RWKV6FeedForward

from navsim.agents.rwkv_block.rwkv6_attention import RWKV6_CrossAttention


class RWKV6_CrossAttBlock(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
            norm_bias: bool = True,
            norm_eps: float = 1e-5,
            attn_mode: str = "chunk",
            expand_k: int = 1,
            expand_v: int = 1,
            num_heads: int = 4,
            proj_low_rank_dim: int = 32,
            gate_low_rank_dim: int = 64,
            fuse_norm: bool = True,
            hidden_ratio: Optional[int] = 3.5,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "sqrelu",
            layer_idx: int = 0
        ):
        super().__init__()
        if norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)

        self.query_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)
        self.keyval_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)        
        self.ffn_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)

        self.self_attn = RWKV6Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            fuse_norm=fuse_norm,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx
        )

        self.cross_attn = RWKV6_CrossAttention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx
        )

        self.ffn = RWKV6FeedForward(
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
        attention_mask: Optional[torch.Tensor] = None,
        past_query: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query
        query = self.query_norm(residual)
        keyval = self.keyval_norm(keyval)

        query, attentions, past_query = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            past_key_values=past_query,
            use_cache=use_cache
        )
        query, attentions, past_key_values = self.cross_attn(
            query=query,
            keyval=keyval,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        query, residual = self.ffn_norm(query, residual, True)
        query, past_query = self.ffn(query, attention_mask, past_query)
        query = residual + query

        outputs = (query, attentions, past_query, past_key_values)

        return outputs

    def init_state(self, **kwargs) -> Tuple[torch.Tensor]:
        state = []
        if callable(getattr(self.attn, 'init_state', None)):
            state += self.attn.init_state(**kwargs)
        if callable(getattr(self.ffn, 'init_state', None)):
            state += self.ffn.init_state(**kwargs)
        return state


def rwkv_block_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval):
    """ rwkv block test """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RWKV6_CrossAttBlock()
    model.to(device)
    torch.manual_seed(42)

    query = torch.randn(batch_size, seq_len_query, hidden_dim).to(device)
    keyval = torch.randn(batch_size, seq_len_keyval, hidden_dim).to(device)

    output, _, _, _ = model(query, keyval) 
    print("Output Shape: ", output.shape)
    print(output[0, 0, :5])
    print('yep')


if __name__ == "__main__":
    batch_size = 10
    hidden_dim = 256
    seq_len_query = 31
    seq_len_keyval = 64

    rwkv_block_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval)

