"""Named model/training size presets. Frozen: derive variants with `dataclasses.replace`."""

from __future__ import annotations

from dataclasses import dataclass, replace

from gpt_scratch.model.config import GPTConfig


@dataclass(frozen=True)
class Preset:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    batch_size: int
    max_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    eval_interval: int
    eval_batches: int
    dropout: float = 0.0


PRESETS: dict[str, Preset] = {
    # seconds, for tests
    "test": Preset(1, 2, 16, 16, 4, 6, 3e-3, 3e-4, 2, 3, 2),
    # ~0.8M params, minutes on a laptop; used for the ablation study
    "tiny": Preset(4, 4, 128, 128, 32, 1500, 3e-3, 3e-4, 100, 250, 20, 0.0),
    # ~10M params (video-sized: 6 layers, 6 heads, 384 dims)
    "small": Preset(6, 6, 384, 256, 32, 5000, 1e-3, 1e-4, 200, 500, 20, 0.1),
}


def with_steps(preset: Preset, steps: int) -> Preset:
    """Return a copy running `steps` steps, with warmup and eval interval kept consistent."""
    if steps < 2:
        raise ValueError("steps must be >= 2")
    warmup = max(1, min(preset.warmup_steps, steps // 10 or 1))
    return replace(
        preset, max_steps=steps, warmup_steps=min(warmup, steps - 1),
        eval_interval=min(preset.eval_interval, steps),
    )


def model_config_for(
    preset: Preset, vocab_size: int, pos: str = "learned", norm: str = "layernorm", ffn: str = "gelu"
) -> GPTConfig:
    """Build the model config for a preset with the chosen architecture switches."""
    return GPTConfig(
        vocab_size=vocab_size, block_size=preset.block_size, n_layer=preset.n_layer,
        n_head=preset.n_head, n_embd=preset.n_embd, dropout=preset.dropout,
        pos_encoding=pos, norm=norm, ffn=ffn,
    )
