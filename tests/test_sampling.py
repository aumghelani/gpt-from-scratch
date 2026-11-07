import pytest
import torch

from gpt_scratch.sample.sampling import (
    apply_repetition_penalty,
    sample_next,
    top_k_filter,
    top_p_filter,
)

NEG_INF = float("-inf")


def test_top_k_keeps_only_k_largest():
    out = top_k_filter(torch.tensor([[1.0, 5.0, 3.0, 4.0]]), k=2)
    assert out[0].tolist() == [NEG_INF, 5.0, NEG_INF, 4.0]


@pytest.mark.parametrize("k", [None, 0, 4, 99])
def test_top_k_noop_cases(k):
    x = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    assert torch.equal(top_k_filter(x, k), x)


def test_top_k_does_not_mutate_input():
    x = torch.tensor([[1.0, 5.0, 3.0]])
    top_k_filter(x, 1)
    assert x.tolist() == [[1.0, 5.0, 3.0]]


def probs_to_logits(*p):
    return torch.log(torch.tensor([p]))


def test_top_p_keeps_smallest_nucleus():
    logits = probs_to_logits(0.5, 0.3, 0.15, 0.05)
    kept = torch.isfinite(top_p_filter(logits, 0.75))[0].tolist()
    assert kept == [True, True, False, False]


def test_top_p_always_keeps_top_token():
    logits = probs_to_logits(0.5, 0.3, 0.15, 0.05)
    assert torch.isfinite(top_p_filter(logits, 0.01))[0].tolist() == [True, False, False, False]


def test_top_p_one_is_noop_and_order_independent():
    logits = probs_to_logits(0.05, 0.5, 0.15, 0.3)
    assert torch.equal(top_p_filter(logits, 1.0), logits)
    kept = torch.isfinite(top_p_filter(logits, 0.75))[0].tolist()
    assert kept == [False, True, False, True]


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, [torch.tensor([0, 1])], penalty=2.0)
    assert out[0].tolist() == [1.0, -4.0, 1.0]


def test_repetition_penalty_of_one_is_noop():
    x = torch.randn(2, 5)
    assert torch.equal(apply_repetition_penalty(x, [torch.tensor([1]), torch.tensor([2])], 1.0), x)


def test_temperature_zero_is_greedy():
    g = torch.Generator().manual_seed(0)
    tok = sample_next(torch.tensor([[0.1, 3.0, 0.2]]), temperature=0.0, generator=g)
    assert tok.tolist() == [[1]]


def test_samples_match_softmax_distribution():
    probs = [0.6, 0.3, 0.1]
    logits = probs_to_logits(*probs).repeat(20000, 1)
    g = torch.Generator().manual_seed(0)
    draws = sample_next(logits, temperature=1.0, generator=g).squeeze(1)
    freq = torch.bincount(draws, minlength=3).float() / draws.numel()
    assert torch.allclose(freq, torch.tensor(probs), atol=0.02)


def test_low_temperature_sharpens_and_high_flattens():
    logits = probs_to_logits(0.6, 0.3, 0.1).repeat(10000, 1)
    g = torch.Generator().manual_seed(1)
    cold = sample_next(logits, temperature=0.2, generator=g).squeeze(1)
    hot = sample_next(logits, temperature=3.0, generator=g).squeeze(1)
    assert (cold == 0).float().mean() > 0.9
    assert (hot == 0).float().mean() < 0.5


def test_top_k_one_forces_argmax():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]]).repeat(50, 1)
    assert set(sample_next(logits, temperature=1.0, top_k=1, generator=g).flatten().tolist()) == {2}


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        sample_next(torch.zeros(1, 3), temperature=-1.0, generator=torch.Generator())


def test_same_seed_same_samples():
    logits = torch.randn(8, 20)
    a = sample_next(logits, 1.0, generator=torch.Generator().manual_seed(4))
    b = sample_next(logits, 1.0, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b)
