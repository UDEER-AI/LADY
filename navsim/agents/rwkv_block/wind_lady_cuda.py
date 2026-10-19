import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


CHUNK_LEN = 16
HEAD_SIZE = 64
flags = [
    '-res-usage',
    f'-D_C_={HEAD_SIZE}',
    "--use_fast_math",
    "-O3",
    "-Xptxas -O3",
    "--extra-device-vectorization",
    '-v',
    "-arch=sm_89",  # 使用 sm_89
    "-gencode=arch=compute_89,code=sm_89",  # 针对 RTX 4090
    "-gencode=arch=compute_80,code=sm_80",  # 兼容 A100
    "-gencode=arch=compute_86,code=sm_86",  # RTX 30 系列
]

# 加载 CUDA 扩展
script_dir = os.path.dirname(os.path.abspath(__file__)) 
load(
    name="wind",
    sources=[
        os.path.join(script_dir, 'cuda/attn.cu'),
        os.path.join(script_dir, 'cuda/attn.cpp')
    ],
    is_python_module=False,
    verbose=True,
    extra_cuda_cflags=flags
)


def fw_attn(w, q, k, v, a, b, s0):
    B, T, H, C = w.shape
    y = torch.empty_like(v)
    sT = torch.empty_like(s0)
    s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.bfloat16, device=w.device)
    torch.ops.wind.forward(w, q, k, v, a, b, s0, y, s, sT)
    return y, sT, s

def bw_attn(w, q, k, v, a, b, s, dy, dsT):
    dw, dq, dk, dv, da, db, ds0 = [torch.empty_like(x) for x in [w, q, k, v, a, b, dsT]]
    torch.ops.wind.backward(w, q, k, v, a, b, dy, s, dsT, dw, dq, dk, dv, da, db, ds0)
    return dw, dq, dk, dv, da, db, ds0

class WindRWKV7(torch.autograd.Function):
    @staticmethod
    def forward(w, q, k, v, a, b, s0):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w, q, k, v, a, b, s0])
        assert all(i.is_contiguous() for i in [w, q, k, v, a, b, s0])
        assert all(i.shape == w.shape for i in [w, q, k, v, a, b])
        assert list(s0.shape) == [B, H, C, C]
        y, sT, s = fw_attn(w, q, k, v, a, b, s0)
        return y, sT, s
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs[:-1], output[-1])
    @staticmethod
    def backward(ctx, dy, dsT, ds):
        w, q, k, v, a, b, s = ctx.saved_tensors
        B, T, H, C = w.shape
        if dsT is None: 
            dsT = torch.zeros(B, H, T // CHUNK_LEN, C, C, dtype=torch.bfloat16, device=w.device)
     
        assert ds is None
        assert all(i.dtype==torch.bfloat16 for i in [dy, dsT])
        assert all(i.is_contiguous() for i in [dy, dsT])
        dw, dq, dk, dv, da, db, ds0 = bw_attn(w, q, k, v, a, b, s, dy, dsT)
        return dw, dq, dk, dv, da, db, ds0


def RUN_CUDA_RWKV7(r, w, k, v, a, b, s0=None):
    B, T, HC = w.shape
    C = HEAD_SIZE
    H = HC // C

    # padding sequence length to multiple of CHUNK_LEN
    padding_size = (CHUNK_LEN - T % CHUNK_LEN) % CHUNK_LEN
    r, w, k, v, a, b = [F.pad(i, (0, 0, 0, padding_size)) for i in [r, w, k, v, a, b]]

    _, T_expand, _ = w.shape
    r, w, k, v, a, b = [i.view(B, T_expand, H, C).bfloat16() for i in [r, w, k, v, a, b]]
    if s0 is None:
        s0 = torch.zeros(B, H, C, C, dtype=torch.bfloat16, device=w.device)

    return WindRWKV7.apply(w, r, k, v, a, b, s0)[0].view(B, T_expand, HC)[:, :T, :]
