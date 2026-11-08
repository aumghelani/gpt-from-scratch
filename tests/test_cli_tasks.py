import json

import pytest

from gpt_scratch.cli import main

PLAY = "\n\n".join(
    f"SPEAKER{i % 3}:\nthe quick brown fox jumps over number {i}\nand then it runs away quickly"
    for i in range(60)
)


@pytest.fixture
def text_file(tmp_path):
    p = tmp_path / "play.txt"
    p.write_text(PLAY, encoding="utf-8")
    return p


@pytest.fixture
def base_run(tmp_path, text_file):
    data, run = tmp_path / "data", tmp_path / "base"
    assert main(["prepare", "--input", str(text_file), "--out", str(data), "--tokenizer", "bpe", "--vocab-size", "300"]) == 0
    assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"]) == 0
    return run


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_make_tasks_writes_a_train_set_and_a_held_out_who_set(tmp_path, text_file, capsys):
    out = tmp_path / "tasks"
    assert main(["make-tasks", "--text", str(text_file), "--out-dir", str(out), "--min-speeches", "3"]) == 0
    train, held = read_jsonl(out / "train.jsonl"), read_jsonl(out / "heldout.jsonl")
    assert len(held) == 6 and all(c["messages"][0]["content"].startswith("Who says this?") for c in held)
    kinds = {c["messages"][0]["content"].split("\n")[0] for c in train}
    assert kinds == {"Who says this?", "Speak as SPEAKER0.", "Speak as SPEAKER1.", "Speak as SPEAKER2."}
    # nothing held out leaks into training
    held_prompts = {c["messages"][0]["content"] for c in held}
    assert held_prompts.isdisjoint({c["messages"][0]["content"] for c in train})
    assert "6 held-out" in capsys.readouterr().out


def test_make_tasks_is_deterministic(tmp_path, text_file):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        assert main(["make-tasks", "--text", str(text_file), "--out-dir", str(out), "--min-speeches", "3"]) == 0
    assert (a / "train.jsonl").read_text() == (b / "train.jsonl").read_text()


def test_make_tasks_fails_clearly_without_speeches(tmp_path):
    empty = tmp_path / "e.txt"
    empty.write_text("no speaker labels in here at all", encoding="utf-8")
    assert main(["make-tasks", "--text", str(empty), "--out-dir", str(tmp_path / "o")]) != 0


def test_finetune_then_eval_chat_reports_accuracy_and_baseline(tmp_path, text_file, base_run, capsys):
    tasks, sft = tmp_path / "tasks", tmp_path / "sft"
    assert main(["make-tasks", "--text", str(text_file), "--out-dir", str(tasks), "--min-speeches", "3"]) == 0
    assert main(["finetune", "--run", str(base_run), "--conversations", str(tasks / "train.jsonl"),
                 "--out", str(sft), "--steps", "6", "--device", "cpu", "--checkpoint", "best"]) == 0
    capsys.readouterr()
    assert main(["eval-chat", "--run", str(sft), "--tasks", str(tasks / "heldout.jsonl"), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "accuracy" in out and "majority baseline" in out and "6 questions" in out


def test_finetune_rejects_unknown_checkpoint_choice(tmp_path, text_file, base_run):
    with pytest.raises(SystemExit):
        main(["finetune", "--run", str(base_run), "--from-text", str(text_file), "--out", str(tmp_path / "o"),
              "--checkpoint", "middle"])


def test_checkpoint_path_prefers_best_and_falls_back_to_latest(tmp_path):
    from gpt_scratch.cli_chat import checkpoint_path

    (tmp_path / "latest.pt").write_bytes(b"x")
    assert checkpoint_path(tmp_path, "best") == tmp_path / "latest.pt"  # no best.pt yet
    (tmp_path / "best.pt").write_bytes(b"x")
    assert checkpoint_path(tmp_path, "best") == tmp_path / "best.pt"
    assert checkpoint_path(tmp_path, "latest") == tmp_path / "latest.pt"


def test_checkpoint_path_rejects_unknown_choice(tmp_path):
    from gpt_scratch.cli_chat import checkpoint_path

    with pytest.raises(ValueError):
        checkpoint_path(tmp_path, "middle")
