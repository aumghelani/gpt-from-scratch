import torch

from gpt_scratch.model.attention import CausalSelfAttention
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT


def cfg(**kw):
    base = {"vocab_size": 50, "block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 16}
    return GPTConfig(**{**base, **kw})


def test_rope_model_has_no_learned_position_table():
    learned, rope = GPT(cfg(pos_encoding="learned")), GPT(cfg(pos_encoding="rope"))
    assert rope.pos_emb is None
    assert learned.num_params() - rope.num_params() == 32 * 16


def test_rope_model_is_causal():
    torch.manual_seed(0)
    m = GPT(cfg(pos_encoding="rope")).eval()
    idx = torch.randint(0, 50, (1, 14))
    base, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, 9] = (idx2[0, 9] + 1) % 50
    assert torch.allclose(base[0, :9], m(idx2)[0][0, :9], atol=1e-5)


def test_rope_changes_attention_output_vs_no_rope():
    torch.manual_seed(1)
    x = torch.randn(1, 8, 16)
    a = CausalSelfAttention(cfg(pos_encoding="learned")).eval()
    b = CausalSelfAttention(cfg(pos_encoding="rope")).eval()
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(x), b(x), atol=1e-4)


def test_rope_flash_and_manual_agree():
    torch.manual_seed(2)
    m = CausalSelfAttention(cfg(pos_encoding="rope", flash=False)).eval()
    f = CausalSelfAttention(cfg(pos_encoding="rope", flash=True)).eval()
    f.load_state_dict(m.state_dict())
    x = torch.randn(2, 11, 16)
    assert torch.allclose(m(x), f(x), atol=1e-5)
