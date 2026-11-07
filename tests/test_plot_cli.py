import json

import pytest

from gpt_scratch.cli import main
from gpt_scratch.report.logs import read_series

LOG = "step,lr,loss,grad_norm,val_loss,tokens_per_sec\n1,0.1,3.0,1.0,,100\n2,0.1,2.5,1.0,2.9,100\n3,0.1,2.0,1.0,,100\n4,0.1,1.5,1.0,1.7,100\n"


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(LOG)
    return p


def test_read_series_skips_blank_cells(log_file):
    assert read_series(log_file, "val_loss") == [(2.0, 2.9), (4.0, 1.7)]
    assert read_series(log_file, "loss")[0] == (1.0, 3.0)


def test_read_series_unknown_column_is_clear_error(log_file):
    with pytest.raises(ValueError, match="no_such"):
        read_series(log_file, "no_such")


def test_plot_command_writes_svg(tmp_path, log_file):
    out = tmp_path / "c.svg"
    assert main(["plot", "--log", f"run-a={log_file}", "--log", f"run-b={log_file}", "--column", "val_loss",
                 "--out", str(out), "--title", "Val loss"]) == 0
    svg = out.read_text()
    assert svg.count("<polyline") == 2 and "run-a" in svg and "Val loss" in svg


def test_plot_command_rejects_malformed_log_arg(tmp_path, log_file):
    assert main(["plot", "--log", str(log_file), "--out", str(tmp_path / "x.svg")]) != 0


def test_plot_command_missing_file_fails_cleanly(tmp_path):
    assert main(["plot", "--log", f"a={tmp_path / 'nope.csv'}", "--out", str(tmp_path / "x.svg")]) != 0


def test_plot_ablation_draws_deltas_vs_baseline(tmp_path):
    rows = [
        {"name": "bigram", "params": 1, "val_loss": 2.6, "steps": 1, "seconds": 1},
        {"name": "baseline", "params": 1, "val_loss": 1.64, "steps": 1, "seconds": 1},
        {"name": "+rope", "params": 1, "val_loss": 1.56, "steps": 1, "seconds": 1},
    ]
    res = tmp_path / "r.json"
    res.write_text(json.dumps(rows))
    out = tmp_path / "a.svg"
    assert main(["plot-ablation", "--results", str(res), "--out", str(out)]) == 0
    svg = out.read_text()
    assert "+rope" in svg and "-0.080" in svg and "bigram" not in svg
