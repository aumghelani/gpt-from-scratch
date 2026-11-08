import pytest
import torch

from gpt_scratch.experiments.kv_bench import benchmark_cache
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=16, pos_encoding="rope")
    return GPT(cfg).eval()


def test_cached_and_uncached_greedy_decoding_produce_the_same_tokens(model):
    result = benchmark_cache(model, [1, 2, 3], new_tokens=20, device="cpu")
    assert result.identical and result.new_tokens == 20


def test_speedup_is_the_ratio_of_the_two_timings(model):
    r = benchmark_cache(model, [1, 2, 3], new_tokens=10, device="cpu")
    assert r.cached_seconds > 0 and r.uncached_seconds > 0
    assert r.speedup == pytest.approx(r.uncached_seconds / r.cached_seconds)


def test_rejects_a_request_that_would_overflow_the_context_window(model):
    with pytest.raises(ValueError, match="block_size"):
        benchmark_cache(model, [1] * 60, new_tokens=10, device="cpu")


def test_rejects_an_empty_prompt_or_non_positive_length(model):
    with pytest.raises(ValueError):
        benchmark_cache(model, [], new_tokens=5, device="cpu")
    with pytest.raises(ValueError):
        benchmark_cache(model, [1], new_tokens=0, device="cpu")
