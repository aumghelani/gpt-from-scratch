import math

import pytest
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT


def cfg(**kw):
    base = {"vocab_size": 50, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 16}
    return GPTConfig(**{**base, **kw})


def test_logits_shape_and_optional_loss():
    m = GPT(cfg())
    idx = torch.randint(0, 50, (3, 9))
    logits, loss = m(idx)
    assert logits.shape == (3, 9, 50) and loss is None
    _, loss = m(idx, idx)
    assert loss.ndim == 0


def test_initial_loss_near_uniform_entropy():
    torch.manual_seed(0)
    m = GPT(cfg(vocab_size=100, n_embd=32))
    x = torch.randint(0, 100, (32, 16))
    _, loss = m(x, torch.randint(0, 100, (32, 16)))
    assert abs(loss.item() - math.log(100)) < 0.3


def test_weight_tying_shares_storage():
    m = GPT(cfg(tie_weights=True))
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    untied = GPT(cfg(tie_weights=False))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()


def test_param_count_matches_hand_formula():
    c = cfg(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=16)
    C, V, T = c.n_embd, c.vocab_size, c.block_size
    per_layer = 2 * C + (C * 3 * C + 3 * C) + (C * C + C) + 2 * C + (C * 4 * C + 4 * C) + (4 * C * C + C)
    expected = V * C + T * C + c.n_layer * per_layer + 2 * C
    assert GPT(c).num_params() == expected


def test_model_is_causal_end_to_end():
    torch.manual_seed(0)
    m = GPT(cfg()).eval()
    idx = torch.randint(0, 50, (1, 12))
    base, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, 8] = (idx2[0, 8] + 1) % 50
    out, _ = m(idx2)
    assert torch.allclose(base[0, :8], out[0, :8], atol=1e-5)


def test_sequence_longer_than_block_size_rejected():
    with pytest.raises(ValueError):
        GPT(cfg(block_size=8))(torch.randint(0, 50, (1, 9)))


@pytest.mark.parametrize("norm,ffn", [("layernorm", "gelu"), ("rmsnorm", "swiglu"), ("rmsnorm", "gelu")])
def test_all_component_combinations_run(norm, ffn):
    m = GPT(cfg(norm=norm, ffn=ffn))
    logits, _ = m(torch.randint(0, 50, (2, 6)))
    assert torch.isfinite(logits).all()


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(cfg(vocab_size=20, n_embd=32))
    x = torch.randint(0, 20, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
