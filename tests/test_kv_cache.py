import pytest
import torch

from gpt_scratch.model.attention import CausalSelfAttention
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT

COMBOS = [("learned", False), ("learned", True), ("rope", False), ("rope", True)]


def cfg(pos, flash, **kw):
    base = {"vocab_size": 40, "block_size": 24, "n_layer": 2, "n_head": 2, "n_embd": 16}
    return GPTConfig(**{**base, **kw, "pos_encoding": pos, "flash": flash})


@pytest.mark.parametrize("pos,flash", COMBOS)
def test_attention_incremental_matches_full(pos, flash):
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg(pos, flash)).eval()
    x = torch.randn(2, 10, 16)
    full = attn(x)
    out_prefix, past = attn(x[:, :6], return_kv=True)
    assert torch.allclose(out_prefix, full[:, :6], atol=1e-5)
    for t in range(6, 10):
        step, past = attn(x[:, t : t + 1], past=past, return_kv=True)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-5), f"mismatch at t={t}"
    assert past[0].shape[2] == 10


@pytest.mark.parametrize("pos,flash", COMBOS)
def test_gpt_cached_decoding_matches_full_forward(pos, flash):
    torch.manual_seed(1)
    m = GPT(cfg(pos, flash)).eval()
    idx = torch.randint(0, 40, (2, 12))
    full, _ = m(idx)
    logits, past = m.forward_cached(idx[:, :5])
    assert torch.allclose(logits, full[:, :5], atol=1e-4)
    for t in range(5, 12):
        step, past = m.forward_cached(idx[:, t : t + 1], past)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-4), f"mismatch at t={t}"


def test_multi_token_chunk_with_past_matches_full():
    torch.manual_seed(2)
    m = GPT(cfg("rope", True)).eval()
    idx = torch.randint(0, 40, (1, 14))
    full, _ = m(idx)
    _, past = m.forward_cached(idx[:, :6])
    chunk, _ = m.forward_cached(idx[:, 6:11], past)
    assert torch.allclose(chunk, full[:, 6:11], atol=1e-4)


def test_cache_cannot_exceed_block_size():
    m = GPT(cfg("rope", True, block_size=8)).eval()
    _, past = m.forward_cached(torch.randint(0, 40, (1, 8)))
    with pytest.raises(ValueError):
        m.forward_cached(torch.randint(0, 40, (1, 1)), past)


def test_past_is_not_mutated():
    m = GPT(cfg("rope", True)).eval()
    _, past = m.forward_cached(torch.randint(0, 40, (1, 4)))
    before = [t.clone() for layer in past for t in layer]
    m.forward_cached(torch.randint(0, 40, (1, 1)), past)
    after = [t for layer in past for t in layer]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
