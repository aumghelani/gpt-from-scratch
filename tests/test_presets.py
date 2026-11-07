import pytest

from gpt_scratch.presets import PRESETS, with_steps


def test_with_steps_returns_new_object_and_leaves_original():
    base = PRESETS["tiny"]
    short = with_steps(base, 40)
    assert short is not base and short.max_steps == 40 and base.max_steps == 1500


def test_warmup_always_less_than_steps_and_at_least_one():
    for steps in (2, 5, 40, 100, 1500, 10_000):
        p = with_steps(PRESETS["tiny"], steps)
        assert 1 <= p.warmup_steps < p.max_steps


def test_warmup_not_inflated_for_long_runs():
    assert with_steps(PRESETS["tiny"], 10_000).warmup_steps == PRESETS["tiny"].warmup_steps


def test_eval_interval_never_exceeds_steps():
    assert with_steps(PRESETS["tiny"], 40).eval_interval <= 40


def test_too_few_steps_rejected():
    with pytest.raises(ValueError):
        with_steps(PRESETS["tiny"], 1)


def test_every_preset_is_internally_valid():
    for p in PRESETS.values():
        assert p.n_embd % p.n_head == 0 and 1 <= p.warmup_steps < p.max_steps


def test_model_config_for_applies_preset_and_overrides():
    from gpt_scratch.presets import model_config_for

    cfg = model_config_for(PRESETS["test"], vocab_size=65, pos="rope", norm="rmsnorm", ffn="swiglu")
    assert cfg.vocab_size == 65 and cfg.n_embd == 16 and cfg.block_size == 16
    assert (cfg.pos_encoding, cfg.norm, cfg.ffn) == ("rope", "rmsnorm", "swiglu")
