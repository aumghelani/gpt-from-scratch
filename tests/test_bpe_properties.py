from hypothesis import given, settings
from hypothesis import strategies as st

from gpt_scratch.tokenizer.bpe import BPETokenizer

TOK = BPETokenizer.train(
    "Hello, world! The quick brown fox. 日本語 café naïve 123 " * 40,
    vocab_size=320,
    special_tokens=("<|end|>",),
)


@settings(max_examples=300, deadline=None)
@given(st.text())
def test_roundtrip_any_text(s):
    assert TOK.decode(TOK.encode(s, allow_special=False)) == s


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_ids_always_in_vocab(s):
    ids = TOK.encode(s, allow_special=False)
    assert all(0 <= i < TOK.vocab_size for i in ids)


@settings(max_examples=200, deadline=None)
@given(st.text(), st.text())
def test_encoding_never_longer_than_utf8_bytes(a, b):
    s = a + b
    assert len(TOK.encode(s, allow_special=False)) <= len(s.encode("utf-8"))
