import pytest
from my_gpt.model.layers import (
    apply_rope,
    precompute_rope, 
    apply_rms_norm
)
import torch
import torch.nn.functional as F
import time


@pytest.mark.slow
def test_rms_norm():
    x = torch.rand(4, 16, 32) # B, S, E
    print(x.shape)
    eps = 1e-8
    t0 = time.time()
    rms_normed_x1 = apply_rms_norm(x, eps=eps, torch_impl=False)
    custom_time = time.time() - t0
    t0 = time.time()
    rms_normed_x2 = F.rms_norm(x, normalized_shape=(x.size(-1),), eps=eps)
    torch_time = time.time() - t0
    print(f"Custom RMSNorm time: {custom_time:.6f}s, Torch RMSNorm time: {torch_time:.6f}s")

    assert torch.allclose(rms_normed_x1, rms_normed_x2), "RMSNorm output does not match expected output"

class TestRoPE:
    _bs = 4
    seq_len = 16
    d_head = 8
    n_head = 4

    @pytest.mark.slow
    def test_precompute_rope(self):
        d_head = self.d_head
        seq_len = self.seq_len
        with pytest.raises(AssertionError) as excinfo:
            precompute_rope(seq_len, d_head + 1, device='cpu')  # d_head must be even
        assert excinfo.type is AssertionError

        rope_cache = precompute_rope(seq_len, d_head, device='cpu')
        assert rope_cache.shape == (seq_len, d_head // 2, 2), f"Unexpected rope_cache shape: {rope_cache.shape}"

    @pytest.mark.slow
    def test_rope(self):
        _bs = self._bs
        n_head = self.n_head
        d_head = self.d_head
        seq_len = self.seq_len

        rope_cache = precompute_rope(seq_len, d_head, device='cpu')
        assert rope_cache.shape == (seq_len, d_head // 2, 2), f"Unexpected rope_cache shape: {rope_cache.shape}"

        x = torch.randn(_bs, seq_len, d_head)
        with pytest.raises(AssertionError) as excinfo:
            apply_rope(x, rope_cache)  # x must have shape (B, n_heads, seq_len, d_head)
        assert excinfo.type is AssertionError

        x = torch.randn(_bs, n_head, seq_len, d_head) 
        x_rope = apply_rope(x, rope_cache)
        assert x_rope.shape == x.shape, f"Output shape mismatch: {x_rope.shape} != {x.shape}"

        # Check that applying RoPE twice returns to original (approximately)
        # x_recovered = apply_rope(x_rope, rope_cache)
        # assert torch.allclose(x, x_recovered, atol=1e-5), "RoPE application is not reversible"

# to test exception raising
# def test_mytest():
#     with pytest.raises(SystemExit):
#         f()