import re

import pytest
import torch

from gpt_scratch.cli import main
from gpt_scratch.report.heatmap import heatmap_svg, visible_token


def test_visible_token_makes_whitespace_legible():
    assert visible_token(" ") == "·" and visible_token("\n") == "⏎" and visible_token("a") == "a"
    assert visible_token("\t") == "⇥"


def test_one_cell_per_query_key_pair_plus_background():
    t = 4
    w = torch.tril(torch.ones(t, t)) / torch.arange(1, t + 1).unsqueeze(1)
    svg = heatmap_svg(w.tolist(), list("abcd"), "T")
    assert len(re.findall(r'<rect class="cell"', svg)) == t * t
    assert svg.startswith("<svg")


def test_labels_are_escaped():
    svg = heatmap_svg([[1.0]], ["<b>"], "x & y")
    assert "<b>" not in svg and "&lt;b&gt;" in svg and "x &amp; y" in svg


@pytest.mark.parametrize("bad", [[], [[1.0, 0.0]], [[1.0, 0.0], [0.5, 0.5], [0.3, 0.3]]])
def test_non_square_or_empty_rejected(bad):
    with pytest.raises(ValueError):
        heatmap_svg(bad, ["a"] * max(len(bad), 1), "t")


def test_label_count_must_match():
    with pytest.raises(ValueError):
        heatmap_svg([[1.0, 0.0], [0.5, 0.5]], ["a"], "t")


def test_cells_encode_weight_as_opacity():
    svg = heatmap_svg([[1.0, 0.0], [0.25, 0.75]], ["a", "b"], "t")
    opacities = [float(m) for m in re.findall(r'class="cell"[^>]*fill-opacity="([\d.]+)"', svg)]
    assert opacities[0] == pytest.approx(1.0) and opacities[1] == 0.0 and opacities[2] < opacities[3]


def test_attention_command_end_to_end(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("to be or not to be that is the question\n" * 80, encoding="utf-8")
    data, run, out = tmp_path / "d", tmp_path / "r", tmp_path / "attn.svg"
    assert main(["prepare", "--input", str(src), "--out", str(data)]) == 0
    assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"]) == 0
    assert main(["attention", "--run", str(run), "--text", "to be or", "--layer", "0", "--head", "0",
                 "--out", str(out)]) == 0
    assert len(re.findall(r'<rect class="cell"', out.read_text())) == 8 * 8


def test_attention_command_rejects_bad_layer(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("to be or not to be\n" * 80, encoding="utf-8")
    data, run = tmp_path / "d", tmp_path / "r"
    main(["prepare", "--input", str(src), "--out", str(data)])
    main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"])
    assert main(["attention", "--run", str(run), "--text", "to be", "--layer", "9", "--out", str(tmp_path / "o.svg")]) != 0
