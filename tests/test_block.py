import torch

from gpt_scratch.model.block import Block
from gpt_scratch.model.config import GPTConfig


def cfg(**kw):
    base = {"vocab_size": 20, "block_size": 16, "n_layer": 1, "n_head": 2, "n_embd": 16}
    return GPTConfig(**{**base, **kw})


def test_shape_preserved():
    x = torch.randn(2, 8, 16)
    assert Block(cfg()).eval()(x).shape == x.shape


def test_is_identity_when_sublayer_outputs_are_zero():
    blk = Block(cfg()).eval()
    torch.nn.init.zeros_(blk.attn.proj.weight)
    torch.nn.init.zeros_(blk.attn.proj.bias)
    torch.nn.init.zeros_(blk.ffn.down.weight)
    torch.nn.init.zeros_(blk.ffn.down.bias)
    x = torch.randn(2, 8, 16)
    assert torch.allclose(blk(x), x, atol=1e-6)  # residual path carries x through


def test_block_is_causal():
    torch.manual_seed(0)
    blk = Block(cfg(norm="rmsnorm", ffn="swiglu")).eval()
    x = torch.randn(1, 10, 16)
    base = blk(x)
    x2 = x.clone()
    x2[0, 6] += 3.0
    assert torch.allclose(base[0, :6], blk(x2)[0, :6], atol=1e-6)
