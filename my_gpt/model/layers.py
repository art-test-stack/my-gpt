from my_gpt.utils.settings import *
from my_gpt.model.module import Module

import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class Embedding(Module):
    '''Embedding layer'''
    def __init__(
            self, 
            config: Settings = Settings(),
        ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.emb_dim, 
            padding_idx=config.padding_idx, 
            max_norm=config.max_norm, 
            norm_type=config.norm_type, 
            scale_grad_by_freq=config.scale_grad_by_freq, 
            sparse=config.sparse or True, 
            device=config.device,
            dtype=config.dtype
        )
        # self.embedding = nn.Parameter(
        #     data=torch.randn(config.vocab_size, config.emb_dim),
        #     requires_grad=True
        # )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class Linear(nn.Linear, Module):
    '''Linear layer'''
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool = False, 
            device: torch.device = DEVICE, 
            dtype=None
        ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device)
        # TODO: Reparametrize weights and bias here


class PositionalEncoding(Module):
    def __init__(
            self,
            dim_model: int = DIM_MODEL,
            n_pos: int = MAX_CONTEXT
        ) -> None:
        super().__init__()
        self.register_buffer('table', self._get_sinusoid_encoding_table(dim_model, n_pos))
    
    def _get_sinusoid_encoding_table(self, d_model: int = DIM_MODEL, n_pos: int = MAX_CONTEXT):
        ''' Sinusoid position encoding table '''
        pos = torch.arange(n_pos, dtype=torch.float32)
        i = torch.arange(d_model)

        pos_enc = torch.ger(pos, 1e4 ** (- 2 * (i//2) / d_model))

        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) 
        return pos_enc
    
    def plot_table(self):
        pos_enc_np = self.table.cpu().numpy()
        plt.imshow(pos_enc_np, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xlabel("Embedding index")
        plt.ylabel("Sequence index")
        plt.title('Sinusoidal Positional Encoding Table')
        plt.show()

    def forward(self, x):
        return x + self.table[:,:x.size(2)]
    
class AttentionBlock(Module):
    '''Scaled Dot-Product Attention'''

    def __init__(self, dropout: float =0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            d_key: int = DIM_HEAD, 
            mask=None, 
            mask_value: int = MASK_VALUE
        ):
        attention = torch.matmul(q / math.sqrt(d_key), k.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, mask_value)

        attention = self.dropout(self.softmax(attention))
        output = torch.matmul(attention, v)
        # TODO: add a way to get attention mechanism weights representation
        return output, attention
    

class MultiHeadAttention(Module):
    '''Multi-Head Attention module'''
    def __init__(
            self, 
            dim_model: int = DIM_MODEL, 
            n_heads: int = NUM_HEADS, 
            d_head: int = DIM_HEAD
        ) -> None:
        super().__init__()
        assert(dim_model == d_head * n_heads, "Dimensions are not correct")
        self.n_heads = n_heads
        self.d_head = d_head

        self.w_q = Linear(dim_model, d_head * n_heads, bias=False) 
        self.w_k = Linear(dim_model, d_head * n_heads, bias=False) 
        self.w_v = Linear(dim_model, d_head * n_heads, bias=False)

        self.w_o = AttentionBlock()

    def forward(self, x: torch.Tensor, mask=None):
        n_heads = self.n_heads
        d_head = self.d_head

        batch_size = x.size(0)
        len_x = x.size(1)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = x.view(batch_size, len_x, n_heads, d_head).transpose(1,2)
        k = x.view(batch_size, len_x, n_heads, d_head).transpose(1,2)
        v = x.view(batch_size, len_x, n_heads, d_head).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        output, attention = self.w_o(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, d_head, -1)
        return output, attention


class FeedForward(Module):
    '''Position-Wise Feed Forward Network'''
    def __init__(
            self, 
            d_in: int = DIM_MODEL, 
            d_latent: int = DIM_FFN, 
            dropout: float = DROPOUT
        ) -> None:
        super().__init__()
        self.w_1 = Linear(d_in, d_latent, dropout)
        self.activation = nn.ReLU()
        self.w_2 = Linear(d_latent, d_in, dropout)
        self.dropout = nn.Dropout(DROPOUT)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_2(self.activation(self.w_1(x)))

        if self.training:
            h = self.dropout(h)
            
        h += x
        output = self.layer_norm(h)
        return output
    

class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(
            self, 
            dim_model: int = DIM_MODEL, 
            dim_ffn: int = DIM_FFN, 
            n_heads: int = NUM_HEADS, 
            d_head: int = DIM_HEAD, 
            dropout: float = DROPOUT
        ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_model=dim_model, n_heads=n_heads, d_head=d_head
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        
        self.ffn = FeedForward(d_in=dim_model, d_latent=dim_ffn)

    def forward(self, x, self_attention_mask=None,):
        
        h, self_attention = self.attention(x, mask=self_attention_mask)

        if self.training:
            h = self.dropout(h) 
        h += x
        h = self.layer_norm(h)
        output = h + self.ffn(h)

        return output, self_attention


