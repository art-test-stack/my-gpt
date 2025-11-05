from my_gpt.model.layers import *
from my_gpt.model.module import Module

from my_gpt.utils.settings import *

import torch.nn as nn


class Decoder(Module):
    def __init__(
            self,
            config: Settings = Settings(),
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim_model=config.dim_model,
                dim_ffn=config.dim_ffn, 
                n_heads=config.n_heads, 
                d_head=config.d_head, 
                dropout=config.dropout
            )
            for _ in range(config.n_layers)]
        )
    
    def forward(
            self, 
            x: torch.Tensor, 
            mask=None,
            return_attentions=False
        ):
        self_attn_list = []
        for layer in self.layers: 
            x, self_attention = layer(x=x, self_attention_mask=mask)
            if return_attentions:
                self_attn_list.append(self_attention) 

        if return_attentions:
            return x, self_attn_list
        
        return x,


class MichelTransformer(Module):
    def __init__(
            self,
            args = Settings()
        ) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.padding_idx = args.padding_idx
        self.max_content = args.max_content
        # TODO: Will change to customed embedding
        self.embedding = nn.Embedding(
            num_embeddings = args.vocab_size, 
            embedding_dim = args.dim, 
            padding_idx = args.padding_idx
        )
        self.pos_enc = PositionalEncoding(args.dim, args.max_content)

        self.decoder_stack = Decoder(
            config = args
        )
        
        self.model_head = Linear(config.dim, config.vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_pad_mask(x)

        x = self.embedding(x)
        x = self.pos_enc(x)

        output, *_ = self.decoder_stack(x=x, mask=mask)

        return self.model_head(output)
    
    def get_pad_mask(self, seq: torch.Tensor):

        pad_idx = self.padding_idx
        pad_mask = (seq != pad_idx).unsqueeze(-2)

        _, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return pad_mask & subsequent_mask