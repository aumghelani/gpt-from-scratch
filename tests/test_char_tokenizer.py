import pytest

from gpt_scratch.tokenizer.char import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.train("hello world")
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer.train("banana")
    assert tok.vocab_size == 3
    assert tok.decode([0, 1, 2]) == "abn"


def test_unknown_char_raises():
    tok = CharTokenizer.train("abc")
    with pytest.raises(KeyError):
        tok.encode("abz")


def test_save_load_roundtrip(tmp_path):
    tok = CharTokenizer.train("héllo wörld")
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("héllo") == tok.encode("héllo")


def test_empty_training_text_rejected():
    with pytest.raises(ValueError):
        CharTokenizer.train("")
