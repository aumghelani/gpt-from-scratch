import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.train.checkpoint import load_checkpoint, save_checkpoint
from gpt_scratch.train.optim import configure_optimizer

CFG = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)


def trained_model(steps=3):
    torch.manual_seed(0)
    m = GPT(CFG)
    opt = configure_optimizer(m, lr=1e-2, weight_decay=0.1)
    x = torch.randint(0, 20, (4, 8))
    for _ in range(steps):
        _, loss = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m, opt


def test_roundtrip_restores_weights_optimizer_and_step(tmp_path):
    m, opt = trained_model()
    gen = torch.Generator().manual_seed(5)
    path = save_checkpoint(tmp_path / "ck.pt", m, opt, step=3, config=CFG, generator=gen)

    m2 = GPT(CFG)
    opt2 = configure_optimizer(m2, lr=1e-2, weight_decay=0.1)
    gen2 = torch.Generator().manual_seed(0)
    state = load_checkpoint(path, m2, opt2, generator=gen2)

    assert state.step == 3 and state.config == CFG
    for (n1, p1), (_, p2) in zip(m.named_parameters(), m2.named_parameters(), strict=True):
        assert torch.equal(p1, p2), n1
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and all(
        torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1
    )
    assert torch.equal(gen.get_state(), gen2.get_state())


def test_restores_global_rng(tmp_path):
    m, opt = trained_model()
    torch.manual_seed(99)
    path = save_checkpoint(tmp_path / "ck.pt", m, opt, step=1, config=CFG)
    expected = torch.rand(3)
    torch.manual_seed(1)
    load_checkpoint(path, GPT(CFG))
    assert torch.equal(torch.rand(3), expected)


def test_load_without_optimizer_for_inference(tmp_path):
    m, opt = trained_model()
    path = save_checkpoint(tmp_path / "ck.pt", m, opt, step=2, config=CFG)
    m2 = GPT(CFG)
    load_checkpoint(path, m2)
    assert torch.equal(next(m.parameters()), next(m2.parameters()))


def test_save_creates_parent_dirs_and_is_atomic(tmp_path):
    m, opt = trained_model(1)
    path = save_checkpoint(tmp_path / "a" / "b" / "ck.pt", m, opt, step=1, config=CFG)
    assert path.exists() and not list((tmp_path / "a" / "b").glob("*.tmp"))
