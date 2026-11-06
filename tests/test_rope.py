import pytest
import torch

from gpt_scratch.model.rope import apply_rope, build_rope_cache


def test_cache_shapes():
    cos, sin = build_rope_cache(head_size=8, max_len=32)
    assert cos.shape == sin.shape == (32, 4)


def test_position_zero_is_identity():
    cos, sin = build_rope_cache(8, 16)
    x = torch.randn(1, 2, 1, 8)
    assert torch.allclose(apply_rope(x, cos, sin), x, atol=1e-6)


def test_rotation_preserves_norm():
    cos, sin = build_rope_cache(16, 24)
    x = torch.randn(2, 3, 24, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_dot_product_depends_only_on_relative_distance():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(16, 64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        qs = torch.zeros(1, 1, 64, 16)
        ks = torch.zeros(1, 1, 64, 16)
        qs[0, 0, m], ks[0, 0, n] = q, k
        return (apply_rope(qs, cos, sin)[0, 0, m] @ apply_rope(ks, cos, sin)[0, 0, n]).item()

    assert score(3, 1) == pytest.approx(score(13, 11), abs=1e-4)
    assert score(5, 5) == pytest.approx(score(40, 40), abs=1e-4)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-3)


def test_offset_matches_slicing_a_full_application():
    cos, sin = build_rope_cache(8, 32)
    x = torch.randn(1, 2, 10, 8)
    full = apply_rope(x, cos, sin)
    tail = apply_rope(x[:, :, 6:], cos, sin, offset=6)
    assert torch.allclose(full[:, :, 6:], tail, atol=1e-6)


def test_odd_head_size_rejected():
    with pytest.raises(ValueError):
        build_rope_cache(7, 8)


def test_offset_beyond_cache_rejected():
    cos, sin = build_rope_cache(8, 8)
    with pytest.raises(ValueError):
        apply_rope(torch.randn(1, 1, 4, 8), cos, sin, offset=6)
