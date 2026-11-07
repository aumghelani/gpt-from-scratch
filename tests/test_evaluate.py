import math

import numpy as np
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.train.evaluate import estimate_loss

CFG = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.3)
DATA = np.tile(np.arange(16, dtype=np.uint16), 40)


def test_deterministic_for_same_seed():
    torch.manual_seed(0)
    m = GPT(CFG)
    a = estimate_loss(m, DATA, batch_size=4, block_size=8, n_batches=5, seed=1)
    b = estimate_loss(m, DATA, batch_size=4, block_size=8, n_batches=5, seed=1)
    assert a == b


def test_restores_training_mode_and_disables_dropout_and_grad():
    m = GPT(CFG)
    m.train()
    estimate_loss(m, DATA, 4, 8, 3, seed=0)
    assert m.training
    assert all(p.grad is None for p in m.parameters())


def test_untrained_loss_near_log_vocab():
    torch.manual_seed(0)
    loss = estimate_loss(GPT(CFG), DATA, 8, 8, 10, seed=0)
    assert abs(loss - math.log(16)) < 0.3


def test_eval_mode_makes_result_dropout_free():
    m = GPT(CFG)
    assert estimate_loss(m, DATA, 4, 8, 4, seed=3) == estimate_loss(m, DATA, 4, 8, 4, seed=3)
