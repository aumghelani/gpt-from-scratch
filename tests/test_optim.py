import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.train.optim import configure_optimizer


def model():
    return GPT(GPTConfig(vocab_size=30, block_size=8, n_layer=2, n_head=2, n_embd=16))


def test_two_groups_with_expected_weight_decay():
    opt = configure_optimizer(model(), lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]


def test_decay_group_has_only_matrices_nodecay_has_vectors():
    opt = configure_optimizer(model(), lr=1e-3, weight_decay=0.1)
    decay, nodecay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in nodecay["params"])


def test_every_parameter_in_exactly_one_group():
    m = model()
    opt = configure_optimizer(m, lr=1e-3, weight_decay=0.1)
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(m.parameters()))


def test_learning_rate_and_betas_applied():
    opt = configure_optimizer(model(), lr=3e-4, weight_decay=0.0, betas=(0.9, 0.99))
    assert all(g["lr"] == 3e-4 and g["betas"] == (0.9, 0.99) for g in opt.param_groups)
