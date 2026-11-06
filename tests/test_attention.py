import pytest
import torch

from gpt_scratch.model.attention import CausalSelfAttention
from gpt_scratch.model.config import GPTConfig


def cfg(**kw):
    base = {"vocab_size": 50, "block_size": 16, "n_layer": 1, "n_head": 4, "n_embd": 32}
    return GPTConfig(**{**base, **kw})


def test_output_shape_matches_input():
    attn = CausalSelfAttention(cfg()).eval()
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == x.shape


def test_no_future_leakage():
    """Changing token t must not change outputs at positions < t."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg()).eval()
    x = torch.randn(1, 12, 32)
    base = attn(x)
    for t in (1, 5, 11):
        x2 = x.clone()
        x2[0, t] += torch.randn(32) * 5
        out = attn(x2)
        assert torch.allclose(base[0, :t], out[0, :t], atol=1e-6), f"leak before position {t}"
        assert not torch.allclose(base[0, t:], out[0, t:], atol=1e-6)


def test_position_zero_depends_only_on_itself():
    torch.manual_seed(1)
    attn = CausalSelfAttention(cfg()).eval()
    x = torch.randn(1, 6, 32)
    alone = attn(x[:, :1])
    assert torch.allclose(attn(x)[:, :1], alone, atol=1e-6)


def test_batch_items_are_independent():
    torch.manual_seed(2)
    attn = CausalSelfAttention(cfg()).eval()
    x = torch.randn(3, 8, 32)
    full = attn(x)
    single = attn(x[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-6)


def test_rejects_sequence_longer_than_block_size():
    attn = CausalSelfAttention(cfg(block_size=8)).eval()
    with pytest.raises(ValueError):
        attn(torch.randn(1, 9, 32))
