from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer
from gpt_scratch.tokenizer.stream import make_stream_decoder


def test_multibyte_chars_split_across_tokens_decode_cleanly():
    tok = BPETokenizer.train("plain ascii text only " * 20, vocab_size=260)
    ids = tok.encode("héllo 😀 日本")
    feed, flush = make_stream_decoder(tok)
    text = "".join(feed(i) for i in ids) + flush()
    assert text == "héllo 😀 日本" and "�" not in text


def test_partial_sequence_is_buffered_not_emitted():
    tok = BPETokenizer.train("abc " * 20, vocab_size=256)
    feed, flush = make_stream_decoder(tok)
    assert feed(0xF0) == ""  # first byte of a 4-byte emoji: nothing to show yet
    assert flush() == "�"  # stream ended mid-character -> replacement, not a crash


def test_special_tokens_render_as_their_text():
    tok = BPETokenizer.train("abc " * 20, vocab_size=260, special_tokens=("<|end|>",))
    feed, flush = make_stream_decoder(tok)
    assert feed(tok.special_id("<|end|>")) == "<|end|>" and flush() == ""


def test_char_tokenizer_passthrough():
    tok = CharTokenizer.train("hello")
    feed, flush = make_stream_decoder(tok)
    assert "".join(feed(i) for i in tok.encode("hello")) == "hello" and flush() == ""


def test_decoders_are_independent():
    tok = BPETokenizer.train("abc " * 20, vocab_size=256)
    f1, _ = make_stream_decoder(tok)
    f2, _ = make_stream_decoder(tok)
    f1(0xF0)
    assert f2(ord("a")) == "a"
