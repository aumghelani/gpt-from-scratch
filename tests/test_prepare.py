import json

import pytest

from gpt_scratch.data.prepare import prepare_dataset
from gpt_scratch.data.shards import load_shard
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer

TEXT = "To be, or not to be, that is the question.\n" * 60


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


def test_char_dataset(tmp_path, src):
    out = tmp_path / "ds"
    meta = prepare_dataset(src, out, tokenizer="char")
    tok = load_tokenizer(out / "tokenizer.json")
    assert isinstance(tok, CharTokenizer)
    assert meta["vocab_size"] == tok.vocab_size
    assert meta["n_train"] + meta["n_val"] == len(TEXT)
    train = load_shard(out / "train.bin")
    assert tok.decode(train[:12].tolist()) == TEXT[:12]


def test_bpe_dataset_is_shorter_than_chars(tmp_path, src):
    out = tmp_path / "ds"
    meta = prepare_dataset(src, out, tokenizer="bpe", vocab_size=300)
    assert isinstance(load_tokenizer(out / "tokenizer.json"), BPETokenizer)
    assert meta["n_train"] + meta["n_val"] < len(TEXT)
    assert meta["vocab_size"] <= 300


def test_cleaning_is_applied(tmp_path):
    p = tmp_path / "dirty.txt"
    p.write_text("ab\r\ncd\x00ef\n" * 40, encoding="utf-8")
    out = tmp_path / "ds"
    prepare_dataset(p, out, tokenizer="char")
    tok = load_tokenizer(out / "tokenizer.json")
    assert "\r" not in "".join(tok.chars) and "\x00" not in "".join(tok.chars)


def test_unknown_tokenizer_rejected(tmp_path, src):
    with pytest.raises(ValueError):
        prepare_dataset(src, tmp_path / "ds", tokenizer="word")


def test_load_tokenizer_rejects_unknown_type(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"type": "mystery"}))
    with pytest.raises(ValueError):
        load_tokenizer(p)
