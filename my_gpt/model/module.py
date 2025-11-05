import torch
from torch import nn

import numpy as np

class Module(nn.Module):
    '''class Module'''
    def nb_parameters(self) -> int:
        '''Give the number of parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        '''Give the number of trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        '''Give the number of non-trainable parameters of the module'''
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        '''Summarize the module'''
        print(f'Number of parameters: {self.nb_parameters():,}')
        print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        '''Remove NaNs from the module gradients'''
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def clip_gradient(self, max_norm: float) -> None:
        '''Clip the module gradients'''
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)
