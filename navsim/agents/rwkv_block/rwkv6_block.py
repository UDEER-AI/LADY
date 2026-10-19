from typing import Optional, Tuple

import torch
import torch.nn as nn   

from fla.layers.rwkv6 import RWKV6Attention
from fla.models.rwkv6.modeling_rwkv6 import RWKV6FeedForward
from fla.models.utils import Cache

from navsim.agents.rwkv.rwkv_cross_attention import RWKV6_CrossAttention


class RWKV6Block(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
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
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = None,
        ) -> None:
        super().__init__()

        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
            
        self.attn = RWKV6Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            layer_idx=layer_idx
        )

        self.ffn = RWKV6FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx
        )

        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if hasattr(self, "pre_norm"):
            hidden_states = self.pre_norm(hidden_states)

        attn_output, attentions, past_key_values = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        hidden_states = self.time_drop(hidden_states + attn_output)

        hidden_states_ffn, past_key_values = self.ffn(
            self.ffn_norm(hidden_states), attention_mask, past_key_values)
        hidden_states = self.channel_drop(hidden_states + hidden_states_ffn)
        outputs = (hidden_states, attentions, past_key_values)

        return outputs
    

class RWKV6_CrossAttBlock(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            norm_first: bool = True,
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
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = 0
        ) -> None:
        super().__init__()

        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)
        
        self.query_norm = nn.LayerNorm(hidden_size)
        self.keyval_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.self_attn = RWKV6Block(
            hidden_size=hidden_size,
            norm_first=norm_first,
            norm_eps=norm_eps,
            attn_mode=attn_mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            fuse_norm=fuse_norm,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            time_pdrop=time_pdrop,
            channel_pdrop=channel_pdrop,
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
            fuse_norm=fuse_norm,
            layer_idx=layer_idx
        )
  
        self.ffn = RWKV6FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx
        )

        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_query: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        query, attentions, past_query = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            past_key_values=past_query,
            use_cache=use_cache
        )

        if hasattr(self, "pre_norm"):
            query = self.pre_norm(query)

        query_attn, attentions, past_key_values = self.cross_attn(
            query=self.query_norm(query),
            keyval=self.keyval_norm(keyval),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        query = self.time_drop(query + query_attn)

        query_ffn, past_key_values = self.ffn(
            self.ffn_norm(query), attention_mask, past_key_values)
        query = self.channel_drop(query + query_ffn)
        outputs = (query, attentions, past_query, past_key_values)
        return outputs
    

def rwkv_block_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval):
    """ rwkv block test """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RWKV6_CrossAttBlock()
    model.to(device)

    query = torch.randn(batch_size, seq_len_query, hidden_dim).to(device)
    keyval = torch.randn(batch_size, seq_len_keyval, hidden_dim).to(device)

    output, _, _, _ = model(query, keyval) 
    print("Output Shape: ", output.shape)
    print('yep')


if __name__ == "__main__":
    batch_size = 10
    hidden_dim = 256
    seq_len_query = 31
    seq_len_keyval = 2048

    rwkv_block_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval)