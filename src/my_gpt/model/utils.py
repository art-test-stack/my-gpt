import torch

class RowState:
    def __init__(self) -> None:
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

class KVState:
    def __init__(self, batch_size: int, n_layers: int, n_heads: int, d_head: int, seq_len: int) -> None:
        self.kv_shape = (n_layers, 2, batch_size, n_heads, seq_len, d_head)
        self.kv_cache = None
        self.current_pos = 0

    def reset(self):
        self.current_pos = 0

    def init(self, cache):
        assert self.kv_cache is None, "Cache is already initialized"
        assert cache.kv_cache is not None, "Provided cache is empty"

        pass

    def insert(self, layer, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(self.kv_shape, device=k.device, dtype=k.dtype)
        
        self.kv_cache[layer, 0, :, :, self.current_pos, :] = k
        self.kv_cache[layer, 1, :, :, self.current_pos, :] = v
        self.current_pos += 1