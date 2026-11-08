import pytest

from gpt_scratch.cli import main
from gpt_scratch.tokenizer.compare import measure, split_for_comparison

TEXT = "the quick brown fox jumps over the lazy dog and runs away quickly\n" * 200


def test_measure_reports_bytes_per_token():
    result = measure("fake", 100, lambda s: list(range(len(s.encode()) // 4)), "a" * 400)
    assert (result.tokens, result.bytes_per_token) == (100, 4.0)


def test_measure_counts_bytes_not_characters():
    result = measure("fake", 100, lambda s: [0] * 5, "é" * 5)  # 5 characters, 10 bytes
    assert result.bytes_per_token == 2.0


def test_measure_rejects_empty_text_and_empty_encodings():
    with pytest.raises(ValueError):
        measure("x", 1, lambda s: [], "")
    with pytest.raises(ValueError):
        measure("x", 1, lambda s: [], "abc")


def test_split_holds_out_the_tail():
    train, test = split_for_comparison("0123456789", 0.2)
    assert (train, test) == ("01234567", "89")


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_split_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError):
        split_for_comparison("abc", fraction)


def test_compare_tokenizer_command_prints_one_row_per_vocab_size(tmp_path, capsys):
    p = tmp_path / "t.txt"
    p.write_text(TEXT, encoding="utf-8")
    assert main(["compare-tokenizer", "--text", str(p), "--vocab-sizes", "300", "400"]) == 0
    out = capsys.readouterr().out
    assert "own BPE (vocab 300)" in out and "own BPE (vocab 400)" in out and "bytes/token" in out


def test_compare_tokenizer_includes_tiktoken_when_asked(tmp_path, capsys):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        tiktoken.get_encoding("gpt2")  # downloads the vocabulary on first use
    except (OSError, ValueError) as exc:
        pytest.skip(f"tiktoken vocabulary unavailable (offline?): {exc}")
    p = tmp_path / "t.txt"
    p.write_text(TEXT, encoding="utf-8")
    assert main(["compare-tokenizer", "--text", str(p), "--vocab-sizes", "300", "--tiktoken", "gpt2"]) == 0
    assert "tiktoken gpt2" in capsys.readouterr().out


def test_bench_cache_command_reports_speedup(tmp_path, capsys):
    p = tmp_path / "t.txt"
    p.write_text(TEXT, encoding="utf-8")
    data, run = tmp_path / "d", tmp_path / "r"
    assert main(["prepare", "--input", str(p), "--out", str(data), "--tokenizer", "char"]) == 0
    assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"]) == 0
    capsys.readouterr()
    assert main(["bench-cache", "--run", str(run), "--prompt", "the quick", "--new-tokens", "20", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "identical tokens: yes" in out and "speedup" in out
