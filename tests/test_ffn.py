import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.ffn import FeedForward


def cfg(**kw):
    base = {"vocab_size": 10, "block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 32}
    return GPTConfig(**{**base, **kw})


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_gelu_ffn_shape():
    x = torch.randn(2, 5, 32)
    assert FeedForward(cfg(ffn="gelu"))(x).shape == x.shape


def test_swiglu_ffn_shape():
    x = torch.randn(2, 5, 32)
    assert FeedForward(cfg(ffn="swiglu"))(x).shape == x.shape


def test_positionwise_independence():
    torch.manual_seed(0)
    for kind in ("gelu", "swiglu"):
        ffn = FeedForward(cfg(ffn=kind)).eval()
        x = torch.randn(1, 6, 32)
        full = ffn(x)
        for t in range(6):
            assert torch.allclose(full[:, t], ffn(x[:, t : t + 1])[:, 0], atol=1e-6)


def test_swiglu_hidden_is_multiple_of_eight_and_param_matched():
    gelu, swiglu = FeedForward(cfg(ffn="gelu", n_embd=64, n_head=2)), FeedForward(
        cfg(ffn="swiglu", n_embd=64, n_head=2)
    )
    assert swiglu.hidden % 8 == 0
    assert abs(n_params(swiglu) - n_params(gelu)) / n_params(gelu) < 0.1


def test_dropout_only_active_in_training():
    torch.manual_seed(0)
    ffn = FeedForward(cfg(dropout=0.5))
    x = torch.randn(2, 4, 32)
    ffn.eval()
    assert torch.equal(ffn(x), ffn(x))
    ffn.train()
    assert not torch.equal(ffn(x), ffn(x))
