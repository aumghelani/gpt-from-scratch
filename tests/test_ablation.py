import itertools
import json

import pytest

from gpt_scratch.cli import main
from gpt_scratch.experiments.ablation import (
    BIGRAM,
    Variant,
    default_variants,
    render_markdown,
    run_ablation,
)

TEXT = "to be or not to be that is the question\n" * 80


@pytest.fixture
def data_dir(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(TEXT, encoding="utf-8")
    out = tmp_path / "data"
    assert main(["prepare", "--input", str(src), "--out", str(out)]) == 0
    return out


def test_default_variants_progress_one_change_at_a_time():
    vs = default_variants()
    assert vs[0].name == "baseline" and (vs[0].pos, vs[0].norm, vs[0].ffn) == ("learned", "layernorm", "gelu")
    for a, b in itertools.pairwise(vs):
        diffs = sum(x != y for x, y in zip((a.pos, a.norm, a.ffn), (b.pos, b.norm, b.ffn), strict=True))
        assert diffs == 1
    assert len({v.name for v in vs}) == len(vs)


def test_run_ablation_writes_results_and_table(tmp_path, data_dir):
    variants = [Variant("baseline", "learned", "layernorm", "gelu"), Variant("+rope", "rope", "layernorm", "gelu")]
    results = run_ablation(data_dir, tmp_path / "abl", preset="test", steps=6, variants=variants, device="cpu")
    names = [r["name"] for r in results]
    assert names == [BIGRAM, "baseline", "+rope"]
    assert all(r["val_loss"] > 0 and r["params"] > 0 for r in results)
    saved = json.loads((tmp_path / "abl" / "results.json").read_text())
    assert [r["name"] for r in saved] == names
    md = (tmp_path / "abl" / "results.md").read_text()
    assert "| baseline |" in md and "| +rope |" in md and BIGRAM in md


def test_render_markdown_computes_delta_vs_baseline():
    rows = [
        {"name": BIGRAM, "params": 100, "val_loss": 2.5, "steps": 10, "seconds": 1.0},
        {"name": "baseline", "params": 900, "val_loss": 2.0, "steps": 10, "seconds": 2.0},
        {"name": "+rope", "params": 800, "val_loss": 1.8, "steps": 10, "seconds": 2.0},
    ]
    md = render_markdown(rows)
    assert "-0.200" in md  # +rope improves by 0.2 over baseline
    assert md.startswith("| Variant")


def test_cli_ablate_command(tmp_path, data_dir):
    code = main(["ablate", "--data", str(data_dir), "--out", str(tmp_path / "o"), "--preset", "test",
                 "--steps", "6", "--device", "cpu", "--variants", "baseline,+rope"])
    assert code == 0 and (tmp_path / "o" / "results.md").exists()


def test_cli_ablate_rejects_unknown_variant(tmp_path, data_dir):
    assert main(["ablate", "--data", str(data_dir), "--out", str(tmp_path / "o"), "--preset", "test",
                 "--steps", "6", "--variants", "nope"]) != 0
