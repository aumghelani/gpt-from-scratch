import pytest

from gpt_scratch.tokenizer.bpe import BPETokenizer

CORPUS = "the quick brown fox jumps over the lazy dog. " * 30


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300)


def test_worked_example_applies_merges_in_rank_order():
    t = BPETokenizer.train("aaabdaaabac", vocab_size=259)
    assert t.encode("aaab") == [258]


def test_roundtrip_ascii(tok):
    s = "the lazy dog jumps"
    assert tok.decode(tok.encode(s)) == s


@pytest.mark.parametrize("s", ["héllo wörld", "日本語のテキスト", "emoji 😀 test", "tabs\t\tand\n\nnewlines", ""])
def test_roundtrip_unicode_and_whitespace(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_encoding_compresses_seen_text(tok):
    s = "the quick brown fox"
    assert len(tok.encode(s)) < len(s.encode("utf-8"))


def test_unseen_bytes_fall_back_to_byte_ids():
    t = BPETokenizer.train("aaaa", vocab_size=260)
    ids = t.encode("zzz")
    assert ids == [122, 122, 122]
