import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.train.optim import configure_optimizer
from gpt_scratch.train.step import train_step

CFG = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)


def fresh():
    torch.manual_seed(0)
    m = GPT(CFG)
    return m, configure_optimizer(m, lr=1e-2, weight_decay=0.0)


def batch(seed, n=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 16, (n, 8), generator=g)
    return x, torch.roll(x, -1, dims=1)


def test_accumulation_equals_one_big_batch():
    (x1, y1), (x2, y2) = batch(1), batch(2)
    m_a, opt_a = fresh()
    train_step(m_a, opt_a, [(x1, y1), (x2, y2)], grad_clip=0.0)
    m_b, opt_b = fresh()
    train_step(m_b, opt_b, [(torch.cat([x1, x2]), torch.cat([y1, y2]))], grad_clip=0.0)
    # Compare gradients, not post-Adam parameters: Adam amplifies float-rounding noise in
    # near-zero gradients into large updates, which would make a parameter comparison flaky.
    for pa, pb in zip(m_a.parameters(), m_b.parameters(), strict=True):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-7)


def test_returns_mean_loss_and_preclip_grad_norm():
    m, opt = fresh()
    stats = train_step(m, opt, [batch(1)], grad_clip=1e-6)
    assert stats.loss > 0 and stats.grad_norm > 1e-6  # norm reported before clipping


def test_clipping_bounds_the_applied_gradient():
    m, opt = fresh()
    train_step(m, opt, [batch(1)], grad_clip=0.05)
    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None))
    assert total <= 0.05 + 1e-5


def test_step_changes_parameters_and_lowers_loss_on_repeat():
    m, opt = fresh()
    before = [p.clone() for p in m.parameters()]
    first = train_step(m, opt, [batch(1)], grad_clip=1.0).loss
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters(), strict=True))
    for _ in range(20):
        last = train_step(m, opt, [batch(1)], grad_clip=1.0).loss
    assert last < first


def test_empty_micro_batches_rejected():
    import pytest

    m, opt = fresh()
    with pytest.raises(ValueError):
        train_step(m, opt, [], grad_clip=1.0)
