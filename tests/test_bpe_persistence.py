import json

import pytest

from gpt_scratch.tokenizer.bpe import BPETokenizer


def test_save_load_preserves_behaviour(tmp_path):
    t = BPETokenizer.train("hello world " * 30, 290, special_tokens=("<|end|>",))
    path = tmp_path / "tok.json"
    t.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded == t
    assert loaded.encode("hello<|end|>") == t.encode("hello<|end|>")


def test_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"type": "char", "chars": ["a"]}))
    with pytest.raises(ValueError):
        BPETokenizer.load(path)
