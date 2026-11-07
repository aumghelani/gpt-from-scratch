import json

import numpy as np
import pytest
import torch

from gpt_scratch.chat.sft import IGNORE_INDEX, build_example, load_conversations, make_batch_source
from gpt_scratch.chat.template import CHAT_SPECIALS, Message
from gpt_scratch.tokenizer.bpe import BPETokenizer

CORPUS = "hello there how are you i am fine thank you what is your name my name is bot " * 30
CONV = [Message("user", "hello there"), Message("assistant", "i am fine")]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=CHAT_SPECIALS)


def test_example_is_shifted_and_masked(tok):
    ex = build_example(tok, CONV, block_size=64)
    assert len(ex.x) == len(ex.y)
    a = ex.x.index(tok.special_id("<|assistant|>"))
    # position a predicts the first assistant token; earlier positions predict prompt tokens -> ignored
    assert all(label == IGNORE_INDEX for label in ex.y[:a])
    assert all(label != IGNORE_INDEX for label in ex.y[a:])
    assert ex.y[-1] == tok.special_id("<|end|>")


def test_example_truncated_to_block_size(tok):
    long = [Message("user", "hello there"), Message("assistant", "i am fine " * 40)]
    ex = build_example(tok, long, block_size=16)
    assert ex is not None and len(ex.x) == 16 and len(ex.y) == 16


def test_example_dropped_when_truncation_removes_all_targets(tok):
    long = [Message("user", "hello there " * 40), Message("assistant", "i am fine")]
    assert build_example(tok, long, block_size=8) is None


def test_batch_source_pads_with_pad_and_ignore(tok):
    exs = [build_example(tok, CONV, 64)]
    source = make_batch_source(exs, pad_id=tok.special_id("<|endoftext|>"))
    g = torch.Generator().manual_seed(0)
    x, y = source(batch_size=3, block_size=64, generator=g, device="cpu")
    assert x.shape == y.shape == (3, 64)
    n = len(exs[0].x)
    assert (x[:, n:] == tok.special_id("<|endoftext|>")).all()
    assert (y[:, n:] == IGNORE_INDEX).all()
    assert x.dtype == y.dtype == torch.long


def test_batch_source_is_deterministic_for_seed(tok):
    exs = [build_example(tok, CONV, 64), build_example(tok, [CONV[0], Message("assistant", "bot")], 64)]
    source = make_batch_source(exs, pad_id=0)
    a = source(4, 64, torch.Generator().manual_seed(1), "cpu")
    b = source(4, 64, torch.Generator().manual_seed(1), "cpu")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_empty_example_list_rejected():
    with pytest.raises(ValueError):
        make_batch_source([], pad_id=0)


def test_load_conversations_jsonl(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    convs = load_conversations(p)
    assert len(convs) == 2 and convs[0][1] == Message("assistant", "yo")


def test_load_conversations_reports_bad_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"messages": [{"role": "ghost", "content": "x"}]}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_conversations(p)


def test_loss_ignores_masked_positions():
    from gpt_scratch.model.config import GPTConfig
    from gpt_scratch.model.gpt import GPT

    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16))
    x = torch.randint(0, 20, (2, 8))
    y_all = torch.randint(0, 20, (2, 8))
    y_masked = y_all.clone()
    y_masked[:, :4] = IGNORE_INDEX
    _, full = m(x, y_all)
    _, part = m(x, y_masked)
    logits, _ = m(x)
    expected = torch.nn.functional.cross_entropy(logits[:, 4:].reshape(-1, 20), y_all[:, 4:].reshape(-1))
    assert torch.allclose(part, expected, atol=1e-6) and not torch.allclose(full, part)
    assert np.isfinite(part.item())
