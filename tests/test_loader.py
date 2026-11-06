import numpy as np
import pytest
import torch
from gpt_scratch.data.loader import get_batch


@pytest.fixture
def data():
    return np.arange(1000, dtype=np.uint16)


def gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_shapes_and_dtype(data):
    x, y = get_batch(data, batch_size=4, block_size=8, generator=gen())
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == y.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(data):
    x, y = get_batch(data, batch_size=16, block_size=8, generator=gen())
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)  # data is arange, so next token = token + 1


def test_deterministic_for_same_seed(data):
    a = get_batch(data, 4, 8, gen(7))
    b = get_batch(data, 4, 8, gen(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_different_seeds_differ(data):
    a, _ = get_batch(data, 8, 8, gen(1))
    b, _ = get_batch(data, 8, 8, gen(2))
    assert not torch.equal(a, b)


def test_never_reads_past_end(data):
    for seed in range(50):
        _, y = get_batch(data, 32, 100, gen(seed))
        assert int(y.max()) <= 999


def test_block_size_too_large_raises(data):
    with pytest.raises(ValueError):
        get_batch(data, 2, 1000, gen())
