import pytest

from gpt_scratch.cli import main

TEXT = "to be or not to be that is the question\n" * 80


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


def run_pipeline(tmp_path, corpus, extra_train=()):
    data, run = tmp_path / "data", tmp_path / "run"
    assert main(["prepare", "--input", str(corpus), "--out", str(data), "--tokenizer", "char"]) == 0
    assert main(
        ["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu", *extra_train]
    ) == 0
    return data, run


def test_prepare_train_sample_roundtrip(tmp_path, corpus, capsys):
    _, run = run_pipeline(tmp_path, corpus)
    assert (run / "latest.pt").exists() and (run / "tokenizer.json").exists()
    capsys.readouterr()
    code = main(["sample", "--run", str(run), "--prompt", "to be", "--max-new", "12",
                 "--temperature", "0", "--device", "cpu"])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("to be") and len(out.strip()) >= len("to be")


def test_train_flags_change_architecture(tmp_path, corpus):
    _, run = run_pipeline(tmp_path, corpus, ["--pos", "rope", "--norm", "rmsnorm", "--ffn", "swiglu"])
    import torch

    cfg = torch.load(run / "latest.pt", weights_only=False)["config"]
    assert (cfg["pos_encoding"], cfg["norm"], cfg["ffn"]) == ("rope", "rmsnorm", "swiglu")


def test_resume_continues_from_checkpoint(tmp_path, corpus):
    data, run = run_pipeline(tmp_path, corpus, ["--steps", "6", "--stop-after", "3"])
    code = main(["train", "--data", str(data), "--out", str(run), "--preset", "test",
                 "--device", "cpu", "--steps", "6", "--resume"])
    assert code == 0
    import torch

    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 6


def test_unknown_preset_exits_nonzero(tmp_path, corpus):
    data = tmp_path / "data"
    main(["prepare", "--input", str(corpus), "--out", str(data)])
    with pytest.raises(SystemExit):
        main(["train", "--data", str(data), "--out", str(tmp_path / "r"), "--preset", "nope"])


def test_missing_data_dir_is_a_clear_error(tmp_path):
    assert main(["train", "--data", str(tmp_path / "nothing"), "--out", str(tmp_path / "r"),
                 "--preset", "test"]) != 0


def test_training_snapshots_samples_at_start_and_each_eval(tmp_path, corpus):
    _, run = run_pipeline(tmp_path, corpus)
    files = sorted(p.name for p in (run / "samples").glob("step-*.txt"))
    assert files == ["step-000000.txt", "step-000003.txt", "step-000006.txt"]
    assert all(len((run / "samples" / f).read_text()) > 0 for f in files)


def test_eval_interval_flag_overrides_the_preset(tmp_path, corpus):
    _, run = run_pipeline(tmp_path, corpus, ["--eval-interval", "2"])
    files = sorted(p.name for p in (run / "samples").glob("step-*.txt"))
    assert files == ["step-000000.txt", "step-000002.txt", "step-000004.txt", "step-000006.txt"]


def test_eval_interval_must_be_positive(tmp_path, corpus):
    data = tmp_path / "data"
    main(["prepare", "--input", str(corpus), "--out", str(data), "--tokenizer", "char"])
    with pytest.raises(SystemExit):
        main(["train", "--data", str(data), "--out", str(tmp_path / "r"), "--preset", "test",
              "--eval-interval", "0"])


def test_a_crash_mid_training_can_be_resumed_from_the_last_eval(tmp_path, corpus, monkeypatch):
    import torch

    from gpt_scratch import cli

    data, run = tmp_path / "data", tmp_path / "run"
    assert main(["prepare", "--input", str(corpus), "--out", str(data), "--tokenizer", "char"]) == 0

    def crashing_hook_factory(tok, out_dir):
        def hook(step, model):
            if step == 4:
                raise RuntimeError("simulated crash")
        return hook

    monkeypatch.setattr(cli, "make_sample_hook", crashing_hook_factory)
    args = ["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu",
            "--eval-interval", "2"]
    with pytest.raises(RuntimeError):
        main(args)
    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 2  # saved at the step-2 evaluation

    monkeypatch.undo()
    assert main([*args, "--resume"]) == 0
    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 6
