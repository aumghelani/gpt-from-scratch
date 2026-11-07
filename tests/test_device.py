import pytest
import torch

from gpt_scratch.train.device import pick_device, seed_everything


def test_seed_makes_torch_reproducible():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    assert torch.equal(a, torch.rand(5))


def test_seed_makes_numpy_and_random_reproducible():
    import random

    import numpy as np

    seed_everything(7)
    a, b = np.random.rand(3), random.random()
    seed_everything(7)
    assert np.array_equal(a, np.random.rand(3)) and b == random.random()


def test_cpu_always_available():
    assert pick_device("cpu") == "cpu"


def test_auto_returns_a_known_device():
    assert pick_device("auto") in {"cpu", "mps", "cuda"}


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        pick_device("tpu")


def test_unavailable_device_rejected():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            pick_device("cuda")
