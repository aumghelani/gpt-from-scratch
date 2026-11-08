import pytest

from gpt_scratch.cli import main
from gpt_scratch.tour.render import best_val_loss

TEXT = "to be or not to be that is the question\n" * 120
SECTIONS = ["1. Text becomes numbers", "2. What the model is made of", "3. Knowing nothing", "4. What comes next",
            "5. Temperature", "6. Attention", "7. The KV cache", "8. Generating"]


@pytest.fixture(params=["char", "bpe"])
def run(tmp_path, request):
    src = tmp_path / "in.txt"
    src.write_text(TEXT, encoding="utf-8")
    data, run = tmp_path / "d", tmp_path / "r"
    extra = ["--vocab-size", "300"] if request.param == "bpe" else []
    assert main(["prepare", "--input", str(src), "--out", str(data), "--tokenizer", request.param, *extra]) == 0
    assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu",
                 "--pos", "rope", "--norm", "rmsnorm", "--ffn", "swiglu"]) == 0
    return run


def test_tour_prints_every_section_with_real_numbers(run, capsys):
    capsys.readouterr()
    assert main(["tour", "--run", str(run), "--prompt", "to be or", "--device", "cpu", "--layer", "0"]) == 0
    out = capsys.readouterr().out
    for title in SECTIONS:
        assert title in out, title
    assert "random guess" in out and "best validation loss" in out
    assert "attends most to" in out and "tokens fed" in out


def test_tour_rejects_a_layer_that_does_not_exist(run, capsys):
    assert main(["tour", "--run", str(run), "--prompt", "to be", "--device", "cpu", "--layer", "9"]) != 0


def test_tour_rejects_an_empty_prompt(run):
    assert main(["tour", "--run", str(run), "--prompt", "", "--device", "cpu"]) != 0


def test_best_val_loss_reads_the_lowest_logged_value(tmp_path):
    (tmp_path / "log.csv").write_text("step,lr,loss,grad_norm,val_loss,tokens_per_sec\n"
                                      "1,0,2,1,,1\n2,0,2,1,3.5,1\n3,0,2,1,3.1,1\n4,0,2,1,3.3,1\n")
    assert best_val_loss(tmp_path) == (3, 3.1)


def test_best_val_loss_is_none_without_a_log(tmp_path):
    assert best_val_loss(tmp_path) is None
