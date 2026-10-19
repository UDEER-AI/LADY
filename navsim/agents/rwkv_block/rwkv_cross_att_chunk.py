from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import GroupNorm, LayerNorm
from fla.models.utils import Cache
from fla.modules.activations import ACT2FN
from fla.layers.rwkv6 import LerpLinear, DDLerpLinear

from navsim.agents.rwkv_block.rwkv_chunk import rwkv_chunk

class RWKV6_CrossAttention_Chunk(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 256,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.kv_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 2),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 2, hidden_size, bias=False)
        )
        self.kv_bias = nn.Parameter(torch.zeros(2, hidden_size))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim))

        self.w_mu = nn.Parameter(torch.zeros(hidden_size))
        self.w_mu_proj_a = nn.Linear(hidden_size, proj_low_rank_dim, bias=False)
        self.w_mu_proj_b = nn.Linear(proj_low_rank_dim, hidden_size)

        self.w_proj_a = nn.Linear(hidden_size, gate_low_rank_dim, bias=False)
        self.w_proj_b = nn.Linear(gate_low_rank_dim, self.key_dim)

        self.g_norm = GroupNorm(self.num_heads, self.value_dim, elementwise_affine=elementwise_affine, bias=True, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_fn = ACT2FN[gate_fn]

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        batch_size, que_seq_len, hidden_size = query.shape
        batch_size, seq_len, hidden_size = keyval.shape

        hidden_states = keyval
        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        # obtain k, v based on RWKV6
        delta = shifted - hidden_states
        kv = self.kv_proj[0](hidden_states, delta).view(batch_size, seq_len, -1, self.proj_low_rank_dim)
        kv = torch.einsum('b l n r, h n r-> b l n h', self.kv_proj[1](kv), self.kv_proj[2].weight.view(hidden_size, 2, -1))
        k, v = kv.add_(self.kv_bias).unbind(-2)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)

        # obtain r, g like Transformer
        r = self.r_proj(query)
        g = self.g_proj(query)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, k, v))
        u = self.bonus

        recurrent_state = last_state[1] if use_cache else None
        if self.mode == 'chunk':
            o = rwkv_chunk(query, hidden_states, self.w_mu, self.w_mu_proj_a.weight, 
                           self.w_mu_proj_b.weight, self.w_mu_proj_b.bias, 
                           self.w_proj_a.weight, self.w_proj_b.weight, self.w_proj_b.bias, 
                           k, v, u, scale=1., initial_state=recurrent_state,
                           training=False)
        else:
            raise NotImplementedError(f"Not supported mode `{self.mode}`.")

        if past_key_values is not None:
            past_key_values.update((hidden_states[:, -1], o), self.layer_idx, r.shape[2])
        
        o = torch.matmul(r.unsqueeze(-2), o).squeeze(-2)
        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)')) * self.gate_fn(g)
        o = self.o_proj(o)

        return o, None, past_key_values
    

def rwkv_chunk_test(num_heads, batch_size, hidden_dim, seq_len_query, seq_len_keyval, proj_low_rank_dim, gate_low_rank_dim):
    """ only check the function rwkv_chunk """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w_mu = nn.Parameter(torch.zeros(hidden_dim).to(device))
    w_mu_proj_a = nn.Linear(hidden_dim, proj_low_rank_dim, bias=False).to(device)
    w_mu_proj_b = nn.Linear(proj_low_rank_dim, hidden_dim).to(device)

    w_proj_a = nn.Linear(hidden_dim, gate_low_rank_dim, bias=False).to(device)
    w_proj_b = nn.Linear(gate_low_rank_dim, hidden_dim).to(device)

    query = torch.randn((batch_size, seq_len_query, hidden_dim), device=device)
    keyval = torch.randn((batch_size, seq_len_keyval, hidden_dim), device=device)

    k = torch.randn((batch_size, num_heads, seq_len_keyval, hidden_dim // num_heads), device=device)
    v = torch.randn((batch_size, num_heads, seq_len_keyval, hidden_dim // num_heads), device=device)
    u = nn.Parameter(torch.zeros(num_heads, hidden_dim // num_heads).to(device))

    o = rwkv_chunk(query, keyval, w_mu, w_mu_proj_a.weight, w_mu_proj_b.weight, 
                   w_mu_proj_b.bias, w_proj_a.weight, w_proj_b.weight, w_proj_b.bias, k, v, u)
    
    print('o shape:', o.shape)
    print('yep')

def rwkv_cross_att_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval):
    """ rwkv cross attention test """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    rwkv_att = RWKV6_CrossAttention_Chunk()
    rwkv_att.to(device)

    query = torch.randn(batch_size, seq_len_query, hidden_dim).to(device)
    keyval = torch.randn(batch_size, seq_len_keyval, hidden_dim).to(device)
    o, _, _ = rwkv_att(query, keyval)

    print('o shape:', o.shape)
    print('yep')
    

if __name__ == '__main__':
    batch_size = 10
    num_heads = 4
    hidden_dim = 256

    seq_len_query = 31
    seq_len_keyval = 2048

    proj_low_rank_dim = 32
    gate_low_rank_dim = 64

    # rwkv_chunk_test(num_heads, batch_size, hidden_dim, seq_len_query, seq_len_keyval, proj_low_rank_dim, gate_low_rank_dim)
    rwkv_cross_att_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval)
