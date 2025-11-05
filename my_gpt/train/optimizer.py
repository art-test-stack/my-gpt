from my_gpt.utils.settings import *

import torch
from torch import Tensor, optim
from typing import Iterable, Dict, Any, Tuple

class AdamW(optim.AdamW):
    def __init__(
            self,
            params: Iterable[Tensor] | Iterable[Dict[str, Any]],
            lr: float | Tensor = MAX_LEARNING_RATE,
            eps: float = EPSILON,
            weight_decay: float = WEIGHT_DECAY,
            fused: bool | None = None,
            **kwargs
        ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=(BETA_1, BETA_2),
            weight_decay=weight_decay,
            eps=eps,
            fused=fused,
            **kwargs
        )

    def update_learning_rate(self, iter):
        if iter < WARMUP_ITERS:
            self.learning_rate = MAX_LEARNING_RATE * iter / WARMUP_ITERS
		
        elif iter < WARMUP_ITERS + DECAY_ITERS:
            ratio = (iter - WARMUP_ITERS) / (DECAY_ITERS - WARMUP_ITERS)
            assert 0 <= ratio <+ 1
            coeff = 0.5 * (1.0 + torch.cos(torch.pi * ratio))
            self.learning_rate = MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) 
		
        else:
            self.learning_rate = MIN_LEARNING_RATE
        
        for g in self.param_groups:
            g['lr'] = self.learning_rate