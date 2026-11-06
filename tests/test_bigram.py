import math

import torch

from gpt_scratch.model.bigram import BigramLM


def test_forward_shapes_and_optional_loss():
    m = BigramLM(vocab_size=11)
    idx = torch.randint(0, 11, (3, 5))
    logits, loss = m(idx)
    assert logits.shape == (3, 5, 11) and loss is None
    logits, loss = m(idx, idx)
    assert loss is not None and loss.ndim == 0


def test_initial_loss_is_near_uniform_entropy():
    torch.manual_seed(0)
    m = BigramLM(vocab_size=65)
    idx = torch.randint(0, 65, (64, 32))
    _, loss = m(idx, torch.randint(0, 65, (64, 32)))
    assert abs(loss.item() - math.log(65)) < 0.15


def test_can_memorise_a_deterministic_next_token_map():
    torch.manual_seed(0)
    m = BigramLM(vocab_size=8)
    x = torch.arange(8).repeat(4, 1)
    y = (x + 1) % 8
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    for _ in range(100):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
