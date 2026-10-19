import torch
import torch.nn as nn

from navsim.agents.rwkv_block.lady_attention import RWKV7Attention, RWKV7_CrossAttention, RWKV7FeedForward


class RWKV7Block(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            num_heads: int = 4,
            num_layers: int = 3,
            norm_first: bool = True,
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = 0
        ):
        super().__init__()
        
        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)
        
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.att = RWKV7Attention(
            n_embed=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            layer_id=layer_idx
        )

        self.ffn = RWKV7FeedForward(
            n_embed=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            layer_id=layer_idx
        )
    
        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)
        
    def forward(
            self,
            x: torch.Tensor,
            v_first: torch.Tensor,
        ):
        if hasattr(self, 'pre_norm'):
            x = self.pre_norm(x)

        x_attn, v_first = self.att(self.attn_norm(x), v_first)
        x = self.time_drop(x + x_attn)

        x = self.channel_drop(x + self.ffn(self.ffn_norm(x)))
        return x, v_first


class BiRWKV7Block(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256,
            num_heads: int = 4,
            num_layers: int = 3,
            norm_first: bool = True,
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = 0
        ):
        super().__init__()
        
        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)
        
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.att = RWKV7Attention(
            n_embed=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            layer_id=layer_idx
        )

        self.ffn = RWKV7FeedForward(
            n_embed=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            layer_id=layer_idx
        )
    
        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)
        
    def forward(
            self,
            x: torch.Tensor,
            v_first: torch.Tensor,
        ):
        if hasattr(self, 'pre_norm'):
            x = self.pre_norm(x)

        x_attn_forward, v_first_forward = self.att(self.attn_norm(x), v_first)
        if v_first is not None:
            v_first = torch.flip(v_first, dims=[1])
        x_attn_backward, v_first_backward = self.att(self.attn_norm(torch.flip(x, dims=[1])), v_first)

        x = self.time_drop(x + x_attn_forward + torch.flip(x_attn_backward, dims=[1]))
        v_first = v_first_forward + torch.flip(v_first_backward, dims=[1])

        x = self.channel_drop(x + self.ffn(self.ffn_norm(x)))
        return x, v_first


class RWKV7_DoubleSelfAttBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int = 256,
            num_heads: int = 4,
            num_layers: int = 3,
            norm_first: bool = True,
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = 0
        ):
        super().__init__()

        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)

        self.qkv_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.query_attn = RWKV7Block(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            norm_first=norm_first,
            time_pdrop=time_pdrop,
            channel_pdrop=channel_pdrop,
            layer_idx=layer_idx
        )
        
        self.qkv_self_attn = RWKV7Attention(
            n_embed=hidden_size, 
            n_head=num_heads, 
            n_layer=num_layers, 
            layer_id=layer_idx
        )

        self.ffn = RWKV7FeedForward(
            n_embed=hidden_size, 
            n_head=num_heads, 
            n_layer=num_layers, 
            layer_id=layer_idx
        )

        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)

    def forward(self, query, keyval, query_v_first, qkv_v_first, frames=None):
        query, query_v_first = self.query_attn(query, query_v_first)
        
        if hasattr(self, 'pre_norm'):
            query = self.pre_norm(query)

        B, query_len, _ = query.size()
        if frames is not None:
            keyval_padding = torch.zeros_like(query, device=query.device)
            qkv = torch.cat([keyval, keyval_padding], dim=1)

            # put query in the right place
            frame_indices = frames * 64
            qkv[:, frame_indices:frame_indices + query_len, :] = query

            qkv_attn, query_v_first = self.qkv_self_attn(
                self.qkv_norm(qkv),
                query_v_first
            )
            query_attn = qkv_attn[torch.arange(B).unsqueeze(1), frame_indices.unsqueeze(1) + torch.arange(query_len).unsqueeze(0), :]
        else:
            qkv = torch.cat([keyval, query], dim=1)
            qkv_attn, query_v_first = self.qkv_self_attn(
                self.qkv_norm(qkv),
                query_v_first
            )
            query_attn = qkv_attn[:, -query_len:, :]
        
        query = self.time_drop(query + query_attn)
        query = self.channel_drop(query + self.ffn(self.ffn_norm(query)))
        
        return query, query_v_first, qkv_v_first


class RWKV7_CrossAttBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int = 256,
            num_heads: int = 4,
            num_layers: int = 3,
            norm_first: bool = True,
            time_pdrop: float = 0.0,
            channel_pdrop: float = 0.0,
            layer_idx: int = 0
        ):
        super().__init__()

        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size)

        self.query_norm = nn.LayerNorm(hidden_size)
        self.keyval_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.self_attn = RWKV7Block(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            norm_first=norm_first,
            time_pdrop=time_pdrop,
            channel_pdrop=channel_pdrop,
            layer_idx=layer_idx
        )
        
        self.cross_attn = RWKV7_CrossAttention(
            n_embed=hidden_size, 
            n_head=num_heads, 
            n_layer=num_layers, 
            layer_id=layer_idx
        )

        self.ffn = RWKV7FeedForward(
            n_embed=hidden_size, 
            n_head=num_heads, 
            n_layer=num_layers, 
            layer_id=layer_idx
        )

        self.time_drop = nn.Dropout(time_pdrop)
        self.channel_drop = nn.Dropout(channel_pdrop)

    def forward(
            self,
            query: torch.Tensor,
            keyval: torch.Tensor,
            query_v_first: torch.Tensor,
            cross_query_v_first: torch.Tensor,
            frames: int = None
        ):
        query, query_v_first = self.self_attn(query, query_v_first)

        if hasattr(self, 'pre_norm'):
            query = self.pre_norm(query)

        query_attn, cross_query_v_first = self.cross_attn(
            self.query_norm(query), 
            self.keyval_norm(keyval),
            cross_query_v_first,
            frames
        )
        query = self.time_drop(query + query_attn)
        
        query = self.channel_drop(query + self.ffn(self.ffn_norm(query)))
        return query, query_v_first, cross_query_v_first


def self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.ModuleList(
        [
            RWKV7Block(
                hidden_size= 256,
                num_heads= 4,
                num_layers= 2,
                norm_first = True,
                layer_idx= i
            )
            for i in range(2)
        ]
    )

    model.to(device)
    B = 16
    seq_len = 31

    keyval = torch.randn(B, seq_len, 256).to(device)
    v_first = torch.empty_like(keyval).to(device)

    for layer in model:
        keyval, v_first = layer(keyval, v_first)

    print("Success!")
    print("Output Shape: ", keyval.shape)
    print("v_first Shape: ", v_first.shape)

def cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.ModuleList(
        [
            RWKV7_CrossAttBlock(
                hidden_size= 256,
                num_heads= 4,
                num_layers= 6,
                norm_first = True,
                layer_idx= i,
            )
            for i in range(6)
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
        query, query_v_first, cross_query_v_first = layer(query, keyval, query_v_first, cross_query_v_first)

    print("Success!")
    print("Output Shape: ", query.shape)
    print("query_v_first Shape: ", query_v_first.shape)
    print("cross_query_v_first Shape: ", cross_query_v_first.shape)


if __name__ == '__main__':
    # self_attn_test()
    cross_attn_test()
