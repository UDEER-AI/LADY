
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from navsim.agents.rwkv_block.lady_cuda import RUN_CUDA_RWKV7g  # pengbo
from navsim.agents.rwkv_block.wind_lady_cuda import RUN_CUDA_RWKV7  # johanwind


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, T, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape

    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output.flatten(2).transpose(1, 2)

def double_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/2
    B, T, C = input.shape
    output = torch.zeros_like(input)
    output[:, shift_pixel:T, int(C * gamma * 2):int(C * gamma * 3)] = input[:, 0:T - shift_pixel, int(C * gamma * 2):int(C * gamma * 3)]
    output[:, 0:T - shift_pixel, int(C * gamma * 3):int(C * gamma * 4)] = input[:, shift_pixel:T, int(C * gamma * 3):int(C * gamma * 4)]
    return output

def single_shift(input, shift_pixel=1, *args):
    assert len(input.shape) == 3  # [B, T, C]
    return nn.ZeroPad2d((0, 0, shift_pixel, -shift_pixel))(input)


class RWKV7Attention(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, layer_id, head_size_divisor: int = 8):
        super().__init__()
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_size_divisor = head_size_divisor

        assert self.n_embed % self.n_head == 0
        self.head_size = self.n_embed // self.n_head
        H = self.n_head
        N = self.head_size
        C = self.n_embed

        with torch.no_grad():
            if self.n_layer == 1:
                ratio_0_to_1 = 0.0
            else:
                ratio_0_to_1 = layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

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
            D_DECAY_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3 * (C ** 0.5)) / 32) * 32)) # suggestion
            if self.layer_id != 0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6 * (C ** 0.8)) / 32) * 32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5) * (self.head_size_divisor ** 2)) # !!! notice eps value !!!

            self._init_weights()

    def _init_weights(self):
        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.receptance.weight.data.uniform_(-0.5 / (self.n_embed ** 0.5), 0.5 / (self.n_embed ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (self.n_embed ** 0.5), 0.05 / (self.n_embed ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (self.n_embed ** 0.5), 0.5 / (self.n_embed ** 0.5))
        self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0 and v_first is None:
            v_first = v # store the v of the first layer
        else:
            _, v_first_seq_len, _ = v_first.shape
            if v_first_seq_len < T:
                kv_len = T - v_first_seq_len
                v[:, kv_len:] = v[:, kv_len:] + (v_first - v[:, kv_len:]) * torch.sigmoid(self.v0 + (xv[:, kv_len:] @ self.v1) @ self.v2) 
            else:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a).to(torch.float32)
        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a).to(torch.float32)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first

class RWKV7_CrossAttention(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, layer_id, head_size_divisor: int = 8):
        super().__init__()
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_size_divisor = head_size_divisor

        assert self.n_embed % self.n_head == 0
        self.head_size = self.n_embed // self.n_head
        H = self.n_head
        N = self.head_size
        C = self.n_embed

        with torch.no_grad():
            if self.n_layer == 1:
                ratio_0_to_1 = 0.0
            else:
                ratio_0_to_1 = layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

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
            D_DECAY_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3 * (C ** 0.5)) / 32) * 32)) # suggestion
            if self.layer_id != 0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6 * (C ** 0.8)) / 32) * 32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5) * (self.head_size_divisor ** 2)) # !!! notice eps value !!!

            self._init_weights()
    
    def _init_weights(self):
        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.receptance.weight.data.uniform_(-0.5 / (self.n_embed ** 0.5), 0.5 / (self.n_embed ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (self.n_embed ** 0.5), 0.05 / (self.n_embed ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (self.n_embed ** 0.5), 0.5 / (self.n_embed ** 0.5))
        self.output.weight.data.zero_()

    def forward(self, query, keyval, v_first, frames=None):
        batch_size, query_len, query_hidden_size = query.shape
        batch_size, keyval_len, keyval_hidden_size = keyval.shape
        assert query_hidden_size == keyval_hidden_size

        # expand the query and the keyval
        expanded_query = query.unsqueeze(2).repeat(1, 1, keyval_len, 1)
        expanded_query = expanded_query.reshape(-1, keyval_len, query_hidden_size)

        expanded_keyval = keyval.unsqueeze(1).repeat(1, query_len, 1, 1)
        expanded_keyval = expanded_keyval.reshape(-1, keyval_len, keyval_hidden_size)

        B, T, C = expanded_keyval.size()
        H = self.n_head

        xx_cross = expanded_query - expanded_keyval
        xx_keyval = self.time_shift(expanded_keyval) - expanded_keyval

        x = expanded_keyval
        xr = x + xx_cross * self.x_r
        xw = x + xx_cross * self.x_w
        xa = x + xx_cross * self.x_a
        xg = x + xx_cross * self.x_g
        xk = x + xx_keyval * self.x_k
        xv = x + xx_keyval * self.x_v
        
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a).to(torch.float32)
        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a).to(torch.float32)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = x * g
        if frames is not None:
            frame_indices = frames * 64  # each frame has 64 feature length
            frame_indices = frame_indices.repeat_interleave(query_len)
            x = x[torch.arange(B), frame_indices]
        else:
            x = x[:, -1, :]

        x = x.view(batch_size, query_len, query_hidden_size)
        x = self.output(x)
        return x, v_first


class RWKV7FeedForward(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, layer_id, hidden_rate: int = 4):
        super().__init__()
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.n_embed = n_embed
        self.n_head = n_head
        assert self.n_embed % self.n_head == 0
        self.head_size = self.n_embed // self.n_head
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embed)
            for i in range(self.n_embed):
                ddd[0, 0, i] = i / self.n_embed
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

        self.key = nn.Linear(self.n_embed, self.n_embed * hidden_rate, bias=False)
        self.value = nn.Linear(self.n_embed * hidden_rate, self.n_embed, bias=False)

        # Init weights for rwkv7
        self._init_wieghts()

    def _init_wieghts(self):
        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5 / (self.n_embed ** 0.5), 0.5 / (self.n_embed ** 0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


def rwkv_self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rwkv = RWKV7Attention(
        n_embed= 256,
        n_head= 4,
        n_layer= 6,
        layer_id= 0
    )
    rwkv.to(device)

    B = 4
    seq_len = 31

    query = torch.randn(B, seq_len, 256).to(device)

    v_first = None
    query, v_first = rwkv(query, v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("v_first Shape: ", v_first.shape)

def rwkv_cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rwkv = RWKV7_CrossAttention(
        n_embed= 256,
        n_head= 4,
        n_layer= 6,
        layer_id= 0
    )
    rwkv.to(device)

    B = 4
    que_seq_len = 31
    seq_len = 641

    query = torch.randn(B, que_seq_len, 256).to(device)
    keyval = torch.randn(B, seq_len, 256).to(device)

    v_first = None
    query, v_first = rwkv(query, keyval, v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("v_first Shape: ", v_first.shape)


if __name__ == '__main__':
    # rwkv_self_attn_test()
    rwkv_cross_attn_test()
