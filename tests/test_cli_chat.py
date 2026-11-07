import json

import pytest
import torch

from gpt_scratch.cli import main

LINES = "\n\n".join(f"SPEAKER{i % 3}:\nto be or not\nto be that is\nthe question now" for i in range(60))


@pytest.fixture
def base_run(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(LINES, encoding="utf-8")
    data, run = tmp_path / "data", tmp_path / "base"
    assert main(["prepare", "--input", str(src), "--out", str(data), "--tokenizer", "bpe", "--vocab-size", "300"]) == 0
    assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"]) == 0
    return src, run


def test_finetune_from_text_then_chat(tmp_path, base_run, capsys):
    src, run = base_run
    out = tmp_path / "sft"
    assert main(["finetune", "--run", str(run), "--from-text", str(src), "--out", str(out),
                 "--steps", "6", "--device", "cpu"]) == 0
    assert (out / "latest.pt").exists() and (out / "tokenizer.json").exists()
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 6
    capsys.readouterr()
    assert main(["chat", "--run", str(out), "--message", "SPEAKER1:", "--max-new", "10",
                 "--temperature", "0", "--device", "cpu"]) == 0
    assert "<|end|>" not in capsys.readouterr().out


def test_finetune_from_jsonl(tmp_path, base_run):
    _, run = base_run
    jl = tmp_path / "c.jsonl"
    jl.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": "to be"}, {"role": "assistant", "content": "or not"}]}) for _ in range(8)))
    assert main(["finetune", "--run", str(run), "--conversations", str(jl), "--out", str(tmp_path / "o"),
                 "--steps", "4", "--device", "cpu"]) == 0


def test_finetune_requires_a_data_source(tmp_path, base_run):
    _, run = base_run
    assert main(["finetune", "--run", str(run), "--out", str(tmp_path / "o")]) != 0


def test_finetune_refuses_char_tokenizer_runs(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(LINES, encoding="utf-8")
    data, run = tmp_path / "d", tmp_path / "r"
    main(["prepare", "--input", str(src), "--out", str(data), "--tokenizer", "char"])
    main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"])
    assert main(["finetune", "--run", str(run), "--from-text", str(src), "--out", str(tmp_path / "o"),
                 "--steps", "4", "--device", "cpu"]) != 0
