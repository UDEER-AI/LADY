from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
import triton
import triton.language as tl    

from fla.utils import device_capacity, detect_tf32
from fla.utils import contiguous, autocast_custom_fwd, autocast_custom_bwd


_detect_use_tf32 = None


def rwkv_chunk(
    query: torch.Tensor,
    keyval: torch.Tensor,
    w_mu: torch.Tensor,
    w_mu_proj_a: torch.Tensor,
    w_mu_proj_b: torch.Tensor,
    w_mu_bias: torch.Tensor,
    w_proj_a: torch.Tensor,
    w_proj_b: torch.Tensor,
    w_bias: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    checkpoint_level: Optional[int] = 0,
    training: bool = True,
    use_tf32: Optional[bool] = None,
    chunk_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        query (torch.Tensor):
            query of shape `(B, T1, hidden_dim)`. 
        keyval (torch.Tensor):
            keyvals of shape `(B, T2, hidden_dim)`.
        w_mu (torch.Tensor):
            default w_mu of shape '(hidden_dim, )'.
        w_mu_proj_a (torch.Tensor):
            weight matrix of shape `(proj_low_rank_dim, hidden_dim)`.
        w_mu_proj_b (torch.Tensor):
            weight matrix of shape `(hidden_dim, proj_low_rank_dim)`.
        w_mu_bias (torch.Tensor):
            bias of shape `(hidden_dim, )`.
        w_proj_a (torch.Tensor):
            weight matrix of shape `(gate_low_rank_dim, hidden_dim)`.
        w_proj_b (torch.Tensor):
            weight matrix of shape `(H * K, gate_low_rank_dim)`.
        w_bias (torch.Tensor):
            bias of shape `(H * K, )`.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        u (torch.Tensor):
            bonus of shape `(H, K)` or `(B, H, K)` for each head.
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `0`:
            - Level `0`: store forward hidden states for backprop.
            - Level `1`: recompute the forward hidden states during backward.
    """
    global _detect_use_tf32
    assert checkpoint_level in [0, 1]
    scale = k.shape[-1] ** -0.5 if scale == -1.0 else scale
    u_2d = True if u.dim() == 2 else False
    if use_tf32 is None and _detect_use_tf32 is None:
        _detect_use_tf32 = detect_tf32()
    else:
        _detect_use_tf32 = use_tf32
    o = ChunkRWKV_Function.apply(query, keyval, w_mu, w_mu_proj_a, w_mu_proj_b, 
                                 w_mu_bias, w_proj_a, w_proj_b, w_bias, k, v, u, 
                                 scale, initial_state, checkpoint_level, u_2d, training, _detect_use_tf32, chunk_size)
    return o


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_fwd_kernal_intra(
    query, keyval,
    w_mu, w_mu_proj_a, w_mu_proj_b, w_mu_bias,
    w_proj_a, w_proj_b, w_bias,
    k, v, g, h,
    hidden_dim: tl.constexpr,
    proj_low_rank_dim: tl.constexpr,
    gate_low_rank_dim: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_v, i_k, i_bh_t2 = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bh = i_bh_t2 // NT
    i_t2 = i_bh_t2 % NT
    i_b = i_bh // H
    i_h = i_bh % H

    p_w_mu = tl.make_block_ptr(w_mu, (hidden_dim,), (1,), (0,), (hidden_dim,), (0,))
    p_w_mu_proj_a = tl.make_block_ptr(w_mu_proj_a, (proj_low_rank_dim, hidden_dim), (hidden_dim, 1), (0, 0), (proj_low_rank_dim, hidden_dim), (1, 0))
    p_w_mu_proj_b = tl.make_block_ptr(w_mu_proj_b, (hidden_dim, proj_low_rank_dim), (proj_low_rank_dim, 1), (0, 0), (hidden_dim, proj_low_rank_dim), (1, 0))
    p_w_mu_bias = tl.make_block_ptr(w_mu_bias, (hidden_dim,), (1,), (0,), (hidden_dim,), (0,))
    p_w_proj_a = tl.make_block_ptr(w_proj_a, (gate_low_rank_dim, hidden_dim), (hidden_dim, 1), (0, 0), (gate_low_rank_dim, hidden_dim), (1, 0))
    p_w_proj_b = tl.make_block_ptr(w_proj_b, (H * K, gate_low_rank_dim), (gate_low_rank_dim, 1), (i_h * K + i_k * BK, 0), (BK, gate_low_rank_dim), (1, 0))
    p_w_bias = tl.make_block_ptr(w_bias, (H * K,), (1,), (i_h * K + i_k * BK,), (BK,), (0,))

    p_keyval = tl.make_block_ptr(keyval + i_b * T2 * hidden_dim, (T2, hidden_dim), (hidden_dim, 1), (i_t2 * BT, 0), (BT, hidden_dim), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T2 * K, (K, T2), (1, K), (i_k * BK, i_t2 * BT), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T2 * V, (T2, V), (V, 1), (i_t2 * BT, i_v * BV), (BT, BV), (1, 0))    

    for i_t1 in range(T1):
        p_query = tl.make_block_ptr(query + i_b * T1 * hidden_dim, (T1 * hidden_dim,), (1,), (i_t1 * hidden_dim, ), (hidden_dim,), (0,))
        b_query = tl.load(p_query, boundary_check=(0,))

        # [BT, hidden_dim]
        b_keyval = tl.load(p_keyval, boundary_check=(0, 1))
        delta = b_query[None, :] - b_keyval
        b_w_mu = tl.load(p_w_mu, boundary_check=(0,))
        token_shift = b_keyval + delta * b_w_mu[None, :]

        # [BT, proj_low_rank_dim]
        b_w_mu_proj_a = tl.load(p_w_mu_proj_a, boundary_check=(0, 1))
        token_shift = tl.dot(token_shift, b_w_mu_proj_a.trans(1, 0))

        # triton tanh, [BT, proj_low_rank_dim]
        exp_tk = tl.exp(token_shift)
        exp_neg_tk = tl.exp(-token_shift)
        token_shift = (exp_tk - exp_neg_tk) / (exp_tk + exp_neg_tk)

        # [BT, hidden_dim]
        b_w_mu_proj_b = tl.load(p_w_mu_proj_b, boundary_check=(0, 1))
        b_w_mu_bias = tl.load(p_w_mu_bias, boundary_check=(0,))
        token_shift = tl.dot(token_shift, b_w_mu_proj_b.trans(1, 0)) + b_w_mu_bias[None, :]

        # ddlerp token shift
        # [BT, hidden_dim]
        token_shift = b_keyval + delta * token_shift

        # lora again
        # [BT, gate_low_rank_dim]
        b_w_proj_a = tl.load(p_w_proj_a, boundary_check=(0, 1))
        token_shift = tl.dot(token_shift, b_w_proj_a.trans(1, 0))

        # triton tanh, [BT, gate_low_rank_dim]
        exp_tk = tl.exp(token_shift)
        exp_neg_tk = tl.exp(-token_shift)
        token_shift = (exp_tk - exp_neg_tk) / (exp_tk + exp_neg_tk)

        # [BT, BK]
        b_w_proj_b = tl.load(p_w_proj_b, boundary_check=(0, 1))
        b_w_bias = tl.load(p_w_bias, boundary_check=(0,))
        token_shift = tl.dot(token_shift, b_w_proj_b.trans(1, 0)) + b_w_bias[None, :]
        g_org = -tl.exp(token_shift)
        
        # [BT, BT]
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

        # [BT, BK]
        b_g = tl.dot(m_s, g_org)
        b_g_sum = tl.sum(g_org, axis=0)
        p_g = tl.make_block_ptr(g + i_bh * T1 * NT * K + i_t1 * NT * K, (NT * K,), (1,), (i_t2 * K + i_k * BK,), (BK,), (0,))
        tl.store(p_g, b_g_sum.to(p_g.dtype.element_ty), boundary_check=(0,))

        b_k = tl.load(p_k, boundary_check=(0, 1)).to(TLTYPE)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(TLTYPE)

        b_h = tl.dot(tl.exp(b_g.trans(1, 0)) * b_k, b_v)
        p_h = tl.make_block_ptr(h + i_bh * T1 * NT * K * V + i_t1 * NT * K * V + i_t2 * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_fwd_kernal_inter(
    g, h, o, h0,
    T1: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    for i_t1 in range(T1):
        b_o = tl.zeros((BK, BV), dtype=TLTYPE)
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_bh * T1 * K * V + i_t1 * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_o += tl.load(p_h0, boundary_check=(0, 1))

        for i_t2 in range(NT):
            p_g = tl.make_block_ptr(g + i_bh * T1 * NT * K + i_t1 * NT * K, (NT * K,), (1,), (i_t2 * K + i_k * BK,), (BK,), (0,))
            p_h = tl.make_block_ptr(h + i_bh * T1 * NT * K * V + i_t1 * NT * K * V + i_t2 * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

            b_g = tl.load(p_g, boundary_check=(0,))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o = b_o * tl.exp(b_g[:, None]) + b_h
        
        p_o = tl.make_block_ptr(o + i_bh * T1 * K * V + i_t1 * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class ChunkRWKV_Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, query, keyval, w_mu, w_mu_proj_a, w_mu_proj_b, w_mu_bias, w_proj_a, w_proj_b, 
                w_bias, k, v, u, scale, initial_state, checkpoint_level, u_2d: bool = False, 
                training: bool = True, use_tf32: bool = False, BT: int = 32):
        B, H, T2, K = k.shape
        V = v.shape[-1]
        _, T1, hidden_dim = query.shape
        proj_low_rank_dim, _ = w_mu_proj_a.shape
        gate_low_rank_dim, _ = w_proj_a.shape
        BH = B * H
        BK = min(64, triton.next_power_of_2(K)) if device_capacity else min(32, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V)) if device_capacity else min(32, triton.next_power_of_2(V))
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        NT = triton.cdiv(T2, BT)

        torch_dtype = torch.float32 if k.dtype != torch.float16 else torch.float16
        tl_dtype = tl.float32 if k.dtype != torch.float16 else tl.float16

        g = torch.zeros((B, H, T1, NT, K), dtype=torch_dtype, device=query.device)
        h = torch.zeros((B, H, T1, NT * K, V), dtype=torch_dtype, device=query.device)
        o = torch.zeros((B, H, T1, K, V), dtype=torch_dtype, device=query.device)

        chunk_rwkv6_fwd_kernal_intra[(NV, NK, NT * BH)](
            query, keyval, 
            w_mu, w_mu_proj_a, w_mu_proj_b, w_mu_bias, 
            w_proj_a, w_proj_b, w_bias, 
            k, v, g, h,
            hidden_dim=hidden_dim,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            T1=T1, T2=T2, H=H, K=K, V=V,
            BT=BT, BK=BK, BV=BV, NT=NT,
            TLTYPE=tl_dtype
        )

        chunk_rwkv6_fwd_kernal_inter[(NV, NK, BH)](
            g, h, o, initial_state, 
            T1=T1, K=K, V=V, BK=BK, BV=BV, NT=NT,
            USE_INITIAL_STATE=initial_state is not None,
            TLTYPE=tl_dtype
        )

        if checkpoint_level > 1:
            del h
            h_t, initial_state = None, None
        else:
            h_t, initial_state = h, (None if initial_state is None else initial_state.clone())
        del g

        if training:
            ctx.save_for_backward(query, keyval, w_mu, w_mu_proj_a, w_mu_proj_b, w_mu_bias, w_proj_a, w_proj_b, w_bias, k, v, h_t, initial_state)
            ctx.BT = BT
        return o

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        pass