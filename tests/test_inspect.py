import pytest
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.model.inspect import attention_weights


def make(pos="rope", flash=True):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=3, n_head=4, n_embd=32,
                         pos_encoding=pos, flash=flash)).eval()


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_shapes_rows_sum_to_one_and_are_causal(pos):
    m = make(pos)
    idx = torch.randint(0, 30, (1, 10))
    maps = attention_weights(m, idx)
    assert len(maps) == 3 and all(w.shape == (4, 10, 10) for w in maps)
    for w in maps:
        assert torch.allclose(w.sum(-1), torch.ones(4, 10), atol=1e-5)
        assert torch.count_nonzero(torch.triu(w, diagonal=1)) == 0  # nothing attends to the future


def test_first_token_attends_only_to_itself():
    maps = attention_weights(make(), torch.randint(0, 30, (1, 8)))
    assert all(torch.allclose(w[:, 0, 0], torch.ones(4)) for w in maps)


@pytest.mark.parametrize("flash", [False, True])
def test_weights_reproduce_the_real_attention_output(flash):
    m = make(flash=flash)
    idx = torch.randint(0, 30, (1, 9))
    captured = {}
    attn0 = m.blocks[0].attn
    attn0.register_forward_hook(lambda mod, inp, out: captured.update(x=inp[0], out=out))
    m(idx)
    w = attention_weights(m, idx)[0]
    _, _, v = attn0.qkv(captured["x"]).split(32, dim=2)
    v = v.view(1, 9, 4, 8).transpose(1, 2)
    recomputed = attn0.proj((w.unsqueeze(0) @ v).transpose(1, 2).reshape(1, 9, 32))
    assert torch.allclose(recomputed, captured["out"], atol=1e-5)


def test_model_state_and_mode_are_untouched():
    m = make()
    m.train()
    before = [p.clone() for p in m.parameters()]
    attention_weights(m, torch.randint(0, 30, (1, 6)))
    assert m.training and all(torch.equal(a, b) for a, b in zip(before, m.parameters(), strict=True))
    assert all(len(b.attn._forward_pre_hooks) == 0 for b in m.blocks)  # hooks removed


def test_requires_single_sequence():
    with pytest.raises(ValueError):
        attention_weights(make(), torch.randint(0, 30, (2, 6)))
