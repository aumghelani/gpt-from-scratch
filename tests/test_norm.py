import torch
from torch import nn

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.norm import RMSNorm, make_norm


def test_rmsnorm_matches_formula():
    x = torch.randn(4, 7, 16)
    norm = RMSNorm(16)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_output_has_unit_rms():
    out = RMSNorm(32)(torch.randn(3, 5, 32) * 10)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_is_scale_invariant():
    x = torch.randn(2, 3, 8)
    norm = RMSNorm(8)
    assert torch.allclose(norm(x), norm(x * 100), atol=1e-4)


def test_rmsnorm_has_learnable_gain_only():
    norm = RMSNorm(8)
    assert [n for n, _ in norm.named_parameters()] == ["weight"]


def test_make_norm_dispatches_on_config():
    base = {"vocab_size": 10, "block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 8}
    assert isinstance(make_norm(GPTConfig(**base, norm="layernorm")), nn.LayerNorm)
    assert isinstance(make_norm(GPTConfig(**base, norm="rmsnorm")), RMSNorm)
