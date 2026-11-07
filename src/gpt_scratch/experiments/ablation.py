"""Ablation study: train each architecture variant under identical conditions and compare."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gpt_scratch.data.shards import load_shard
from gpt_scratch.model.bigram import BigramLM
from gpt_scratch.model.gpt import GPT
from gpt_scratch.presets import PRESETS, Preset, model_config_for, with_steps
from gpt_scratch.train.loop import TrainConfig, train

BIGRAM = "bigram"


@dataclass(frozen=True)
class Variant:
    name: str
    pos: str
    norm: str
    ffn: str


def default_variants() -> list[Variant]:
    """Each variant changes exactly one thing relative to the previous one."""
    return [
        Variant("baseline", "learned", "layernorm", "gelu"),
        Variant("+rope", "rope", "layernorm", "gelu"),
        Variant("+rmsnorm", "rope", "rmsnorm", "gelu"),
        Variant("+swiglu", "rope", "rmsnorm", "swiglu"),
    ]


def _train_config(preset: Preset, seed: int, device: str, out_dir: Path) -> TrainConfig:
    return TrainConfig(
        max_steps=preset.max_steps, batch_size=preset.batch_size, block_size=preset.block_size,
        max_lr=preset.max_lr, min_lr=preset.min_lr, warmup_steps=preset.warmup_steps,
        eval_interval=preset.eval_interval, eval_batches=preset.eval_batches,
        seed=seed, device=device, out_dir=str(out_dir),
    )


def _run_one(
    name: str, model: nn.Module, data_dir: Path, run_dir: Path, preset: Preset, vocab: int,
    seed: int, device: str, model_cfg_kwargs: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = _train_config(preset, seed, device, run_dir)
    result = train(
        model, load_shard(data_dir / "train.bin"), load_shard(data_dir / "val.bin"), cfg,
        model_config_for(preset, vocab, **model_cfg_kwargs),
    )
    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": result.val_history[-1][1],
        "steps": result.steps_done,
        "seconds": round(time.perf_counter() - started, 1),
    }


def run_ablation(
    data_dir: str | Path,
    out_dir: str | Path,
    preset: str = "tiny",
    steps: int | None = None,
    variants: list[Variant] | None = None,
    device: str = "auto",
    seed: int = 1337,
) -> list[dict[str, Any]]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab_size"]
    chosen = PRESETS[preset] if steps is None else with_steps(PRESETS[preset], steps)
    results: list[dict[str, Any]] = []

    torch.manual_seed(seed)
    results.append(
        _run_one(BIGRAM, BigramLM(vocab), data_dir, out_dir / BIGRAM, chosen, vocab, seed, device, {})
    )
    for variant in variants or default_variants():
        torch.manual_seed(seed)  # identical init stream for every variant
        kwargs = {"pos": variant.pos, "norm": variant.norm, "ffn": variant.ffn}
        model = GPT(model_config_for(chosen, vocab, **kwargs))
        results.append(
            _run_one(variant.name, model, data_dir, out_dir / variant.name.strip("+"), chosen,
                     vocab, seed, device, kwargs)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
    return results


def render_markdown(rows: list[dict[str, Any]]) -> str:
    base = next((r["val_loss"] for r in rows if r["name"] == "baseline"), None)
    lines = [
        "| Variant | Params | Val loss | Δ vs baseline | Steps | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        delta = "—" if base is None or r["name"] in (BIGRAM, "baseline") else f"{r['val_loss'] - base:+.3f}"
        lines.append(
            f"| {r['name']} | {r['params']:,} | {r['val_loss']:.3f} | {delta} | {r['steps']} | {r['seconds']} |"
        )
    return "\n".join(lines) + "\n"
