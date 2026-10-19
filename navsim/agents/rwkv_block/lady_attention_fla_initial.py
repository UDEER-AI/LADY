from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
import inspect

from fla.modules import GroupNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7

if TYPE_CHECKING:
    from fla.models.utils import Cache


class RWKV7Attention(nn.Module):
    def __init__(
            self,
            mode: str = 'chunk',
            hidden_size: int = 256,
            head_dim: Optional[int] = None,
            num_heads: Optional[int] = 4,
            decay_low_rank_dim: int = 64,
            gate_low_rank_dim: int = 128,
            a_low_rank_dim: int = 64,
            v_low_rank_dim: int = 32,
            elementwise_affine: Optional[bool] = True,
            norm_eps: float = 1e-5,
            n_layer: int = 3,
            layer_idx: int = None,
            fuse_norm: bool = False,
            value_dim: int = None,
            **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx
        self.n_layer = n_layer
        self.fuse_norm = fuse_norm

        with torch.no_grad():
            if self.n_layer == 1:
                ratio_0_to_1 = 0.0
            else:
                ratio_0_to_1 = layer_idx / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.hidden_size)
            for i in range(self.hidden_size):
                ddd[0, 0, i] = i / self.hidden_size

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
                
            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round((1.8 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(self.hidden_size, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, self.hidden_size), 0.1))
            decay_speed = torch.ones(self.hidden_size)
            for n in range(self.hidden_size):
                decay_speed[n] = -7 + 5 * (n / (self.hidden_size - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, self.hidden_size) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(self.hidden_size, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, self.hidden_size), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            if self.layer_idx != 0:
                self.v1 = nn.Parameter(torch.zeros(self.hidden_size, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, self.hidden_size), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, self.hidden_size) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6 * (self.hidden_size ** 0.8)) / 32) * 32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(self.hidden_size, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, self.hidden_size), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, self.hidden_size) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, self.hidden_size))
            self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            # self.time_shift = ShortConvolution(
            #     hidden_size=self.value_dim,
            #     kernel_size=4,
            #     bias=False,
            #     activation='silu'
            # )
            self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

            if self.fuse_norm:
                self.g_norm = GroupNorm(
                    num_groups=self.num_heads,
                    hidden_size=self.value_dim,
                    elementwise_affine=elementwise_affine,
                    eps=self.head_dim * norm_eps,
                    bias=True,
                )
            else:
                self.g_norm = nn.GroupNorm(
                    num_groups=self.num_heads,
                    num_channels=self.value_dim,
                    eps=self.head_dim * norm_eps,
                    affine=elementwise_affine
                )

            self._init_weights()

    def _init_weights(self):
        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.r_proj.weight.data.uniform_(-0.5 / (self.hidden_size ** 0.5), 0.5 / (self.hidden_size ** 0.5))
        self.k_proj.weight.data.uniform_(-0.05 / (self.hidden_size ** 0.5), 0.05 / (self.hidden_size ** 0.5))
        self.v_proj.weight.data.uniform_(-0.5 / (self.hidden_size ** 0.5), 0.5 / (self.hidden_size ** 0.5))
        self.o_proj.weight.data.zero_()

    def rwkv7_compatible_call(self, rwkv7_fn, *, r, w, k, v, a, b, scale,
                              initial_state, output_final_state, cu_seqlens):
        sig = inspect.signature(rwkv7_fn)
        param_names = sig.parameters.keys()

        kwargs = dict(
            r=r, k=k, v=v, a=a, b=b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        # 兼容 w vs log_w
        if "w" in param_names:
            kwargs["w"] = w
        elif "log_w" in param_names:
            kwargs["log_w"] = w
        else:
            raise TypeError("rwkv7_fn 必须包含参数 'w' 或 'log_w'")
        return rwkv7_fn(**kwargs)

    def forward(
        self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        v_first: torch.Tensor = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        update_cache: Optional[bool] = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)
            hidden_states = hidden_states.mul(am)

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
            # launching the triton kernel for just one token will actually be slower
            # mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
            mode = 'fused_recurrent'

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state['conv_state'].unsqueeze(1)
        else:
            # shifted, _ = self.time_shift(hidden_states)
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state['conv_state'][0]

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states
        xr = hidden_states + delta * self.x_r
        xw = hidden_states + delta * self.x_w
        xk = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        xa = hidden_states + delta * self.x_a
        xg = hidden_states + delta * self.x_g

        r = self.r_proj(xr)
        w = -math.exp(-0.5) * (self.w0 + torch.tanh(xw @ self.w1) @ self.w2).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0 and v_first is None:
            v_first = v
        else:
            _, v_first_seq_len, _ = v_first.shape
            if v_first_seq_len < seq_len:
                kv_len = seq_len - v_first_seq_len
                v[:, kv_len:] = v[:, kv_len:] + (v_first - v[:, kv_len:]) * torch.sigmoid(self.v0 + (xv[:, kv_len:] @ self.v1) @ self.v2) 
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) 

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) 
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
        k = k.addcmul(k * (a - 1), self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * am

        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        r, w, k, v, kk, a = map(lambda x: x.to(torch.bfloat16), (r, w, k, v, kk, a))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)

        o, recurrent_state = self.rwkv7_compatible_call(
            rwkv7_fn=rwkv7_fn,
            r=r,
            w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None and update_cache:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = self.o_proj(o * g)

        return o, None, past_key_values, v_first


class RWKV7_CrossAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 256,
        head_dim: Optional[int] = None,
        num_heads: Optional[int] = 4,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 32,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        n_layer: int = 3,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        **kwargs
    ) -> RWKV7_CrossAttention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx
        self.n_layer = n_layer
        self.fuse_norm = fuse_norm

        with torch.no_grad():
            if self.n_layer == 1:
                ratio_0_to_1 = 0.0
            else:
                ratio_0_to_1 = layer_idx / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.hidden_size)
            for i in range(self.hidden_size):
                ddd[0, 0, i] = i / self.hidden_size

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
                
            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round((1.8 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(self.hidden_size, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, self.hidden_size), 0.1))
            decay_speed = torch.ones(self.hidden_size)
            for n in range(self.hidden_size):
                decay_speed[n] = -7 + 5 * (n / (self.hidden_size - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, self.hidden_size) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(self.hidden_size, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, self.hidden_size), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3 * (self.hidden_size ** 0.5)) / 32) * 32)) # suggestion
            if self.layer_idx != 0:
                self.v1 = nn.Parameter(torch.zeros(self.hidden_size, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, self.hidden_size), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, self.hidden_size) + 1.0)
        
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6 * (self.hidden_size ** 0.8)) / 32) * 32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(self.hidden_size, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, self.hidden_size), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, self.hidden_size) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, self.hidden_size))
            self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

            if self.fuse_norm:
                self.g_norm = GroupNorm(
                    num_groups=self.num_heads,
                    hidden_size=self.value_dim,
                    elementwise_affine=elementwise_affine,
                    eps=self.head_dim * norm_eps,
                    bias=True,
                )
            else:
                self.g_norm = nn.GroupNorm(
                    num_groups=self.num_heads,
                    num_channels=self.value_dim,
                    eps=self.head_dim * norm_eps,
                    affine=elementwise_affine
                )

            self._init_weights()

    def _init_weights(self):
        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.r_proj.weight.data.uniform_(-0.5 / (self.hidden_size ** 0.5), 0.5 / (self.hidden_size ** 0.5))
        self.k_proj.weight.data.uniform_(-0.05 / (self.hidden_size ** 0.5), 0.05 / (self.hidden_size ** 0.5))
        self.v_proj.weight.data.uniform_(-0.5 / (self.hidden_size ** 0.5), 0.5 / (self.hidden_size ** 0.5))
        self.o_proj.weight.data.zero_()

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        v_first: torch.Tensor = None,
        frames=None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]:
        batch_size, query_len, query_hidden_size = query.shape
        batch_size, keyval_len, keyval_hidden_size = keyval.shape
        assert query_hidden_size == keyval_hidden_size

        # expand the query and the keyval
        expanded_query = query.unsqueeze(2).repeat(1, 1, keyval_len, 1)
        expanded_query = expanded_query.reshape(-1, keyval_len, query_hidden_size)

        expanded_keyval = keyval.unsqueeze(1).repeat(1, query_len, 1, 1)
        expanded_keyval = expanded_keyval.reshape(-1, keyval_len, keyval_hidden_size)
        expanded_batch_size, _, _ = expanded_keyval.shape

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
            # launching the triton kernel for just one token will actually be slower
            # mode = 'fused_recurrent' if keyval_len <= 64 else self.mode
            mode = 'fused_recurrent'

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        hidden_states = expanded_keyval
        xx_cross = expanded_query - hidden_states
        xx_keyval = self.time_shift(hidden_states) - hidden_states

        xr = hidden_states + xx_cross * self.x_r
        xw = hidden_states + xx_cross * self.x_w
        xa = hidden_states + xx_cross * self.x_a
        xg = hidden_states + xx_cross * self.x_g
        xk = hidden_states + xx_keyval * self.x_k
        xv = hidden_states + xx_keyval * self.x_v
    
        r = self.r_proj(xr)
        w = -math.exp(-0.5) * (self.w0 + torch.tanh(xw @ self.w1) @ self.w2).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) 

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) 
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
        k = k.addcmul(k * (a - 1), self.k_a)

        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        r, w, k, v, kk, a = map(lambda x: x.to(torch.bfloat16), (r, w, k, v, kk, a))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)
        o, recurrent_state = rwkv7_fn(
            r=r,
            w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(expanded_batch_size, keyval_len, -1)

        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(expanded_batch_size, keyval_len, -1)
        o = o * g
        if frames is not None:
            frame_indices = frames * 64  # each frame has 64 feature length
            frame_indices = frame_indices.repeat_interleave(query_len)
            o = o[torch.arange(expanded_batch_size), frame_indices]
        else:
            o = o[:, -1, :]

        o = o.view(batch_size, query_len, query_hidden_size)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first


def rwkv_self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rwkv = RWKV7Attention(
        hidden_size=256,
        num_heads=4,
        layer_idx=0,
        fuse_norm=True
    )
    rwkv.to(device)

    B = 4
    seq_len = 31

    query = torch.randn(B, seq_len, 256).to(device)

    v_first = None
    query, _, past_key_values, v_first = rwkv(query, v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("v_first Shape: ", v_first.shape)

def rwkv_cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rwkv = RWKV7_CrossAttention(
        hidden_size=256,
        num_heads=4,
        layer_idx=0,
        fuse_norm=True
    )
    rwkv.to(device)

    B = 4
    que_seq_len = 31
    seq_len = 641

    query = torch.randn(B, que_seq_len, 256).to(device)
    keyval = torch.randn(B, seq_len, 256).to(device)

    v_first = None
    query, _, past_key_values, v_first= rwkv(query, keyval, v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("v_first Shape: ", v_first.shape)


if __name__ == '__main__':
    # rwkv_self_attn_test()
    rwkv_cross_attn_test()
