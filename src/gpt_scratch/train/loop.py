"""The training loop: schedule, accumulation, eval, logging, checkpoint/resume."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gpt_scratch.data.loader import BatchSource, sample_batch
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.train.checkpoint import load_checkpoint, save_checkpoint
from gpt_scratch.train.device import pick_device, seed_everything
from gpt_scratch.train.evaluate import estimate_loss
from gpt_scratch.train.logger import CSVLogger
from gpt_scratch.train.optim import configure_optimizer
from gpt_scratch.train.schedule import lr_at
from gpt_scratch.train.step import train_step

LOG_FIELDS = ("step", "lr", "loss", "grad_norm", "val_loss", "tokens_per_sec")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int
    batch_size: int
    block_size: int
    grad_accum_steps: int = 1
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    ckpt_interval: int = 0  # 0 = only at the end / on stop_after
    seed: int = 1337
    device: str = "auto"
    out_dir: str = "runs/default"

    def __post_init__(self) -> None:
        for name in ("max_steps", "batch_size", "block_size", "grad_accum_steps", "eval_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.ckpt_interval < 0 or self.eval_batches < 1:
            raise ValueError("ckpt_interval must be >= 0 and eval_batches >= 1")


@dataclass(frozen=True)
class TrainResult:
    steps_done: int
    train_losses: tuple[float, ...]  # losses for the steps run in *this* call
    val_history: tuple[tuple[int, float], ...]  # (step, val_loss) from this call
    checkpoint: Path


def train(
    model: nn.Module,
    train_data: np.ndarray | BatchSource,
    val_data: np.ndarray | BatchSource,
    cfg: TrainConfig,
    model_config: GPTConfig,
    resume_from: str | Path | None = None,
    stop_after: int | None = None,
) -> TrainResult:
    """Train `model` in place. `stop_after` ends the run early (still checkpointing)."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    model.to(device)
    optimizer = configure_optimizer(model, lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    start = 0
    if resume_from is not None:
        start = load_checkpoint(resume_from, model, optimizer, generator, map_location=device).step
    end = cfg.max_steps if stop_after is None else min(cfg.max_steps, start + stop_after)

    out = Path(cfg.out_dir)
    latest, best = out / "latest.pt", out / "best.pt"
    losses: list[float] = []
    val_history: list[tuple[int, float]] = []
    best_val = float("inf")
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps

    def checkpoint(path: Path, step: int) -> None:
        save_checkpoint(path, model, optimizer, step, model_config, generator)

    with CSVLogger(out / "log.csv", LOG_FIELDS) as logger:
        for step in range(start, end):
            lr = lr_at(
                step, max_lr=cfg.max_lr, min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            micro = [
                sample_batch(train_data, cfg.batch_size, cfg.block_size, generator, device)
                for _ in range(cfg.grad_accum_steps)
            ]
            began = time.perf_counter()
            stats = train_step(model, optimizer, micro, cfg.grad_clip)
            elapsed = max(time.perf_counter() - began, 1e-9)
            losses.append(stats.loss)

            done = step + 1
            row = {
                "step": done, "lr": lr, "loss": stats.loss, "grad_norm": stats.grad_norm,
                "tokens_per_sec": tokens_per_step / elapsed,
            }
            if done % cfg.eval_interval == 0 or done == cfg.max_steps:
                val = estimate_loss(
                    model, val_data, cfg.batch_size, cfg.block_size, cfg.eval_batches,
                    seed=cfg.seed, device=device,
                )
                row["val_loss"] = val
                val_history.append((done, val))
                if val < best_val:
                    best_val = val
                    checkpoint(best, done)
            logger.log(**row)
            if cfg.ckpt_interval and done % cfg.ckpt_interval == 0:
                checkpoint(latest, done)
    checkpoint(latest, end)
    return TrainResult(end, tuple(losses), tuple(val_history), latest)
