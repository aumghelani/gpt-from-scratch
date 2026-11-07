"""Hook that snapshots generated text during training (shows the model improving over time)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch import nn

from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.tokenizer import Tokenizer
from gpt_scratch.tokenizer.stream import make_stream_decoder


def make_sample_hook(
    tok: Tokenizer,
    out_dir: str | Path,
    n_tokens: int = 300,
    prompt: str = "",
    seed: int = 0,
    temperature: float = 0.8,
    device: str = "auto",
) -> Callable[[int, nn.Module], None]:
    """Return an `on_eval` callback writing `<out_dir>/samples/step-NNNNNN.txt`."""
    samples_dir = Path(out_dir) / "samples"
    prompt_ids = tok.encode(prompt) if prompt else [0]

    def hook(step: int, model: nn.Module) -> None:
        feed, flush = make_stream_decoder(tok)
        pieces = [prompt]
        for token_id in stream_generate(
            model, prompt_ids, n_tokens, temperature=temperature, seed=seed,
            device=None if device == "auto" else device,
        ):
            pieces.append(feed(token_id))
        pieces.append(flush())
        samples_dir.mkdir(parents=True, exist_ok=True)
        (samples_dir / f"step-{step:06d}.txt").write_text("".join(pieces), encoding="utf-8")

    return hook
