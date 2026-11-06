import pytest

from gpt_scratch.tokenizer.bpe import BPETokenizer

SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>")
CORPUS = "hello world, hello there. " * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=280, special_tokens=SPECIALS)


def test_special_ids_follow_merges(tok):
    base = 256 + len(tok.merges)
    assert tok.special_id("<|endoftext|>") == base
    assert tok.special_id("<|end|>") == base + 3
    assert tok.vocab_size == base + 4


def test_special_encodes_to_single_id(tok):
    ids = tok.encode("<|user|>hello<|end|>")
    assert ids[0] == tok.special_id("<|user|>")
    assert ids[-1] == tok.special_id("<|end|>")


def test_special_roundtrip(tok):
    s = "<|user|>hello world<|end|><|assistant|>hello there<|end|>"
    assert tok.decode(tok.encode(s)) == s


def test_special_text_is_plain_when_not_allowed(tok):
    ids = tok.encode("<|end|>", allow_special=False)
    assert tok.special_id("<|end|>") not in ids
    assert tok.decode(ids) == "<|end|>"


def test_unknown_special_lookup_raises(tok):
    with pytest.raises(KeyError):
        tok.special_id("<|nope|>")


def test_duplicate_special_tokens_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc abc", 256, special_tokens=("<|a|>", "<|a|>"))
