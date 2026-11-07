import csv

import numpy as np
import pytest
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.train.loop import TrainConfig, train

MODEL_CFG = GPTConfig(vocab_size=12, block_size=8, n_layer=1, n_head=2, n_embd=16)
DATA = np.tile(np.arange(12, dtype=np.uint16), 80)  # perfectly predictable cycle


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def tcfg(tmp_path, **kw):
    base = {
        "max_steps": 12, "batch_size": 8, "block_size": 8, "max_lr": 5e-3, "min_lr": 5e-4,
        "warmup_steps": 2, "eval_interval": 6, "eval_batches": 2, "device": "cpu",
        "out_dir": str(tmp_path / "run"),
    }
    return TrainConfig(**{**base, **kw})


def fresh_model():
    torch.manual_seed(0)
    return GPT(MODEL_CFG)


def test_invalid_config_rejected(tmp_path):
    with pytest.raises(ValueError):
        tcfg(tmp_path, max_steps=0)
    with pytest.raises(ValueError):
        tcfg(tmp_path, grad_accum_steps=0)


def test_overfits_a_deterministic_sequence(tmp_path):
    result = train(fresh_model(), DATA, DATA, tcfg(tmp_path, max_steps=120, eval_interval=60), MODEL_CFG)
    assert result.train_losses[0] > 2.0
    assert result.train_losses[-1] < 0.3
    assert result.steps_done == 120


def test_writes_csv_log_and_checkpoints(tmp_path):
    cfg = tcfg(tmp_path)
    result = train(fresh_model(), DATA, DATA, cfg, MODEL_CFG)
    out = tmp_path / "run"
    assert (out / "latest.pt").exists() and (out / "best.pt").exists()
    rows = read_rows(out / "log.csv")
    assert len(rows) == 12 and rows[0]["step"] == "1"
    assert [h[0] for h in result.val_history] == [6, 12]
    assert rows[5]["val_loss"] != "" and rows[0]["val_loss"] == ""


def test_logged_lr_follows_schedule(tmp_path):
    train(fresh_model(), DATA, DATA, tcfg(tmp_path), MODEL_CFG)
    rows = read_rows(tmp_path / "run" / "log.csv")
    lrs = [float(r["lr"]) for r in rows]
    assert lrs[0] < lrs[1] <= 5e-3 + 1e-12 and lrs[-1] < lrs[2]


def test_resume_matches_uninterrupted_run(tmp_path):
    straight = fresh_model()
    r_full = train(straight, DATA, DATA, tcfg(tmp_path / "a"), MODEL_CFG)

    part = fresh_model()
    cfg_b = tcfg(tmp_path / "b")
    train(part, DATA, DATA, cfg_b, MODEL_CFG, stop_after=5)
    resumed = GPT(MODEL_CFG)  # different init on purpose; checkpoint must overwrite it
    r_res = train(resumed, DATA, DATA, cfg_b, MODEL_CFG, resume_from=f"{cfg_b.out_dir}/latest.pt")

    assert r_res.steps_done == 12
    for pa, pb in zip(straight.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6)
    assert r_res.train_losses == pytest.approx(r_full.train_losses[5:], abs=1e-6)


def test_stop_after_saves_resumable_checkpoint(tmp_path):
    cfg = tcfg(tmp_path)
    r = train(fresh_model(), DATA, DATA, cfg, MODEL_CFG, stop_after=4)
    assert r.steps_done == 4 and (tmp_path / "run" / "latest.pt").exists()


def _cycle_source(batch_size, block_size, generator, device):
    """Toy batch source: always the 0..11 cycle, with the first two targets masked out."""
    starts = torch.randint(0, 12, (batch_size,), generator=generator)
    base = (starts[:, None] + torch.arange(block_size + 1)[None, :]) % 12
    x, y = base[:, :-1].clone(), base[:, 1:].clone()
    y[:, :2] = -100
    return x.to(device), y.to(device)


def test_trains_from_a_callable_batch_source(tmp_path):
    result = train(
        fresh_model(), _cycle_source, _cycle_source,
        tcfg(tmp_path, max_steps=120, eval_interval=60), MODEL_CFG,
    )
    assert result.train_losses[-1] < 0.5 < result.train_losses[0]
    assert len(result.val_history) == 2


def test_estimate_loss_accepts_callable():
    from gpt_scratch.train.evaluate import estimate_loss

    loss = estimate_loss(fresh_model(), _cycle_source, 4, 8, 3, seed=0)
    assert loss > 0
