from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import GroupNorm
from fla.models.utils import Cache
from fla.modules.activations import ACT2FN
from fla.layers.rwkv6 import LerpLinear, DDLerpLinear
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6


class RWKV6_CrossAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 256,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
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

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        self.rwg_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 3),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 3, hidden_size, bias=False)
        )
        self.rwg_bias = nn.Parameter(torch.zeros(3, hidden_size))

        self.kv_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 2),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 2, hidden_size, bias=False)
        )
        self.kv_bias = nn.Parameter(torch.zeros(2, hidden_size))

        self.r_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim))

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

        # expand the query and the hidden_states
        expanded_query = query.unsqueeze(2).repeat(1, 1, seq_len, 1)
        expanded_query = expanded_query.reshape(-1, seq_len, hidden_size)

        hidden_states = keyval.unsqueeze(1).repeat(1, que_seq_len, 1, 1)
        hidden_states = hidden_states.reshape(-1, seq_len, hidden_size)
        expanded_batch_size = hidden_states.shape[0]

        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode
        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        delta_cross = expanded_query - hidden_states
        delta = shifted - hidden_states

        rwg = self.rwg_proj[0](hidden_states, delta_cross).view(expanded_batch_size, seq_len, -1, self.proj_low_rank_dim)
        rwg = torch.einsum('b l n r, h n r-> b l n h', self.rwg_proj[1](rwg), self.rwg_proj[2].weight.view(hidden_size, 3, -1))
        r, w, g = rwg.add_(self.rwg_bias).unbind(-2)

        kv = self.kv_proj[0](hidden_states, delta).view(expanded_batch_size, seq_len, -1, self.proj_low_rank_dim)
        kv = torch.einsum('b l n r, h n r-> b l n h', self.kv_proj[1](kv), self.kv_proj[2].weight.view(hidden_size, 2, -1))
        k, v = kv.add_(self.kv_bias).unbind(-2)

        r = self.r_proj(hidden_states, r, delta_cross)
        w = self.w_proj(hidden_states, w, delta_cross)
        g = self.g_proj(hidden_states, g, delta_cross)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus

        recurrent_state = last_state[1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_rwkv6(r, k, v, w, u,
                                                       scale=1.,
                                                       initial_state=recurrent_state,
                                                       output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_rwkv6(r, k, v, w, u,
                                             scale=1.,
                                             initial_state=recurrent_state,
                                             output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update((hidden_states[:, -1], recurrent_state), self.layer_idx, r.shape[2])

        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)')) * self.gate_fn(g)
        o = o[:, -1, :]
        o = o.view(batch_size, que_seq_len, hidden_size)
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size
    

def chunk_rwkv6_test(num_heads, batch_size, hidden_dim, seq_len_query, seq_len_keyval, proj_low_rank_dim, gate_low_rank_dim):
    """ only check the function chunk_rwkv6 """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query  = torch.randn(batch_size * seq_len_query, seq_len_keyval, hidden_dim).to(device)
    keyval = torch.randn(batch_size * seq_len_query, seq_len_keyval, hidden_dim).to(device)
    delta = query - keyval

    w_mu_proj = nn.Sequential(
        LerpLinear(hidden_dim, proj_low_rank_dim),
        nn.Tanh(),
        nn.Linear(proj_low_rank_dim, hidden_dim)
    ).to(device)
    w_proj = DDLerpLinear(hidden_dim, hidden_dim, low_rank_dim=gate_low_rank_dim).to(device)

    w_mu = w_mu_proj[2](w_mu_proj[1](w_mu_proj[0](keyval, delta))) 
    w = w_proj(keyval, w_mu, delta)
    w, = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=num_heads), (w,))
    w = -torch.exp(w).to(device)

    r = torch.randn(batch_size * seq_len_query, num_heads, seq_len_keyval, hidden_dim // num_heads).to(device)
    k = torch.randn(batch_size * seq_len_query, num_heads, seq_len_keyval, hidden_dim // num_heads).to(device)
    v = torch.randn(batch_size * seq_len_query, num_heads, seq_len_keyval, hidden_dim // num_heads).to(device)
    u = nn.Parameter(torch.zeros(num_heads, hidden_dim // num_heads).to(device))
    o, recurrent_state = chunk_rwkv6(r, k, v, w, u, scale=1., initial_state=None, output_final_state=False)

    print('o shape: ', o.shape)
    print('yep')

def rwkv_cross_att_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval):
    """ cross attention test """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query = torch.randn(batch_size, seq_len_query, hidden_dim).to(device)
    keyval = torch.randn(batch_size, seq_len_keyval, hidden_dim).to(device)
    model = RWKV6_CrossAttention()
    model.to(device)

    o, _, _ = model(query, keyval)
    print("Output Shape: ", o.shape)
    print('yep')


if __name__ == '__main__':
    num_heads = 4
    batch_size = 10
    hidden_dim = 256
    seq_len_query = 31
    seq_len_keyval = 2048

    proj_low_rank_dim = 32
    gate_low_rank_dim = 64

    # chunk_rwkv6_test(
    #     num_heads, batch_size, hidden_dim, 
    #     seq_len_query, seq_len_keyval, 
    #     proj_low_rank_dim, gate_low_rank_dim
    # )
    rwkv_cross_att_test(batch_size, hidden_dim, seq_len_query, seq_len_keyval)
