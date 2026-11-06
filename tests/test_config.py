import dataclasses

import pytest

from gpt_scratch.model.config import GPTConfig


def make(**kw):
    base = dict(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=32)
    return GPTConfig(**{**base, **kw})


def test_head_size():
    assert make().head_size == 8


def test_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().n_layer = 3  # type: ignore[misc]


def test_with_changes_returns_new_object():
    a = make()
    b = a.with_changes(n_layer=4)
    assert a.n_layer == 2 and b.n_layer == 4 and a is not b


def test_dict_roundtrip():
    c = make(pos_encoding="rope", norm="rmsnorm", ffn="swiglu")
    assert GPTConfig.from_dict(c.to_dict()) == c


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_embd=30, n_head=4),  # not divisible
        dict(vocab_size=0),
        dict(block_size=0),
        dict(n_layer=0),
        dict(dropout=1.0),
        dict(dropout=-0.1),
        dict(pos_encoding="sinusoid"),
        dict(norm="batchnorm"),
        dict(ffn="relu"),
        dict(pos_encoding="rope", n_embd=36, n_head=4),  # head_size 9 is odd
    ],
)
def test_invalid_configs_rejected(kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_from_dict_ignores_nothing_unknown():
    with pytest.raises(TypeError):
        GPTConfig.from_dict({**make().to_dict(), "bogus": 1})
