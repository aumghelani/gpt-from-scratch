import pytest
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.sample.generate import generate, stream_generate


def make(pos="rope", block=32, peaky=True, **kw):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, block_size=block, n_layer=2, n_head=2, n_embd=32, pos_encoding=pos, **kw)
    m = GPT(cfg)
    if peaky:  # sharpen logits so greedy decoding is well-conditioned (no near-ties)
        with torch.no_grad():
            m.tok_emb.weight.mul_(30.0)
    return m


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_cached_and_uncached_greedy_are_identical(pos):
    m = make(pos)
    prompt = [1, 2, 3]
    a = generate(m, prompt, 20, temperature=0.0, use_cache=True)
    b = generate(m, prompt, 20, temperature=0.0, use_cache=False)
    assert a == b and len(a) == 20


def test_respects_max_new_tokens_including_zero():
    m = make()
    assert len(generate(m, [1], 7, temperature=0.0)) == 7
    assert generate(m, [1], 0, temperature=0.0) == []


def test_stop_token_is_not_emitted_and_ends_generation():
    m = make()
    full = generate(m, [1, 2], 12, temperature=0.0)
    assert generate(m, [1, 2], 12, temperature=0.0, stop_ids=(full[0],)) == []
    stop = full[-1]
    first = full.index(stop)
    assert generate(m, [1, 2], 12, temperature=0.0, stop_ids=(stop,)) == full[:first]


def test_cache_path_stops_when_context_is_full():
    m = make(block=8)
    assert len(generate(m, list(range(6)), 10, temperature=0.0, use_cache=True)) == 2


def test_uncached_path_slides_window_past_block_size():
    m = make(block=8)
    assert len(generate(m, [1, 2], 20, temperature=0.0, use_cache=False)) == 20


def test_long_prompt_is_cropped_to_leave_room():
    m = make(block=8)
    out = generate(m, list(range(20)), 5, temperature=0.0, use_cache=True)
    assert len(out) >= 1


def test_seeded_sampling_is_deterministic_and_seed_sensitive():
    m = make(peaky=False)
    a = generate(m, [1], 15, temperature=1.0, seed=3)
    assert a == generate(m, [1], 15, temperature=1.0, seed=3)
    assert a != generate(m, [1], 15, temperature=1.0, seed=4)


def test_stream_is_lazy_and_matches_generate():
    m = make()
    gen = stream_generate(m, [1, 2], 10, temperature=0.0)
    first = next(gen)
    assert [first, *gen] == generate(m, [1, 2], 10, temperature=0.0)


def test_leaves_model_mode_and_grads_untouched():
    m = make()
    m.train()
    generate(m, [1], 3, temperature=0.0)
    assert m.training and all(p.grad is None for p in m.parameters())


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        generate(make(), [], 3)


def test_tokens_are_valid_ids():
    out = generate(make(peaky=False), [1], 30, temperature=1.0, top_k=5, seed=0)
    assert all(0 <= t < 30 for t in out)
