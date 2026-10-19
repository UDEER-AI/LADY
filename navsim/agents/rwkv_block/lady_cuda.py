import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


CHUNK_LEN = 16
HEAD_SIZE = 64
flags = [
    '-res-usage',
    f'-D_C_={HEAD_SIZE}',
    f"-D_CHUNK_LEN_={CHUNK_LEN}",
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
    name="wind_backstepping",
    sources=[
        os.path.join(script_dir, 'cuda/wkv7_cuda.cu'),
        os.path.join(script_dir, 'cuda/wkv7_op.cpp')
    ],
    is_python_module=False,
    verbose=True,
    extra_cuda_cflags=flags
)


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db

def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape

    # padding sequence length to multiple of CHUNK_LEN
    padding_size = (CHUNK_LEN - T % CHUNK_LEN) % CHUNK_LEN
    q, w, k, v, a, b = [F.pad(i, (0, 0, 0, padding_size)) for i in [q, w, k, v, a, b]]

    _, T_expand, _ = q.shape
    q, w, k, v, a, b = [i.view(B, T_expand, HC // HEAD_SIZE, HEAD_SIZE).bfloat16() for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T_expand, HC)[:, :T, :]
