import math

import pytest

from gpt_scratch.train.schedule import lr_at

KW = {"max_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 10, "max_steps": 110}


def test_warmup_is_linear_and_reaches_max_on_last_warmup_step():
    assert lr_at(0, **KW) == pytest.approx(1e-4)  # max_lr * 1/10
    assert lr_at(4, **KW) == pytest.approx(5e-4)
    assert lr_at(9, **KW) == pytest.approx(1e-3)


def test_peak_at_end_of_warmup_then_decays():
    assert lr_at(10, **KW) == pytest.approx(1e-3)
    assert lr_at(60, **KW) == pytest.approx((1e-3 + 1e-4) / 2)  # cosine midpoint


def test_reaches_min_at_max_steps_and_stays():
    assert lr_at(110, **KW) == pytest.approx(1e-4)
    assert lr_at(10_000, **KW) == pytest.approx(1e-4)


def test_monotonic_non_increasing_after_warmup():
    lrs = [lr_at(s, **KW) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:], strict=False))


def test_always_within_bounds():
    for s in range(0, 200):
        assert 0 < lr_at(s, **KW) <= 1e-3 and not math.isnan(lr_at(s, **KW))


@pytest.mark.parametrize(
    "bad",
    [
        {"warmup_steps": 0},
        {"warmup_steps": 200},
        {"min_lr": 1e-2},
        {"max_lr": 0.0},
    ],
)
def test_invalid_arguments_rejected(bad):
    with pytest.raises(ValueError):
        lr_at(1, **{**KW, **bad})
