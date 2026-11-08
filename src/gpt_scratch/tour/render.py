"""Turn the numbers from `facts` into a plain-English walkthrough."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from gpt_scratch.model.gpt import GPT
from gpt_scratch.sample.generate import generate
from gpt_scratch.tokenizer import Tokenizer
from gpt_scratch.tour.facts import (
    attention_links,
    cache_growth,
    last_logits,
    next_token_candidates,
    param_breakdown,
    random_guess_loss,
    temperature_entropies,
    token_pieces,
)

BAR = 30
TEMPERATURES = (0.3, 0.7, 1.0, 1.5)


def best_val_loss(run: Path) -> tuple[int, float] | None:
    """(step, loss) of the lowest validation loss in the run's log.csv, or None."""
    path = run / "log.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [(int(r["step"]), float(r["val_loss"])) for r in csv.DictReader(fh) if r.get("val_loss")]
    return min(rows, key=lambda r: r[1]) if rows else None


def _show(text: str) -> str:
    return repr(text)[1:-1] if text.strip() else repr(text)[1:-1].replace(" ", "·") or "·"


def _section(title: str, why: str) -> list[str]:
    return ["", f"{title}", "-" * len(title), why, ""]


def render_tour(
    model: GPT, tok: Tokenizer, prompt: str, *, layer: int = 0, best: tuple[int, float] | None = None,
    sample_tokens: int = 40, seed: int = 7,
) -> str:
    ids = tok.encode(prompt)
    cfg = model.config
    out: list[str] = [f"Tour of a {model.num_params():,}-parameter GPT, following the prompt {prompt!r}"]

    out += _section(
        "1. Text becomes numbers",
        "The network cannot read letters. A tokenizer cuts the text into pieces and replaces each\n"
        f"piece with an integer id. This tokenizer has {cfg.vocab_size} possible tokens.",
    )
    pieces = token_pieces(tok, prompt)
    out.append(f"{len(prompt)} characters -> {len(pieces)} tokens   (· marks a space)")
    out += [f"  id {i:>5}  ->  {_show(text)!r}" for i, text in pieces]

    out += _section(
        "2. What the model is made of",
        f"{cfg.n_layer} blocks, each with attention ({cfg.n_head} heads) and a feed-forward network, "
        f"{cfg.n_embd} numbers per token.\nEvery parameter below is one learned number.",
    )
    parts = param_breakdown(model)
    total = sum(parts.values())
    out += [f"  {name:<20}{n:>12,}  {n / total:>6.1%}" for name, n in parts.items() if n]
    out.append(f"  {'total':<20}{total:>12,}")
    if cfg.tie_weights:
        out.append("  (the output head reuses the token-embedding table, so it adds no parameters)")

    guess = random_guess_loss(cfg.vocab_size)
    out += _section(
        "3. Knowing nothing",
        "Training minimises the average surprise (cross-entropy, in nats) at the true next token.\n"
        "A model that knows nothing gives every token equal probability, so its loss is ln(vocabulary).",
    )
    out.append(f"  random guess loss      ln({cfg.vocab_size}) = {guess:.3f}")
    if best:
        out.append(f"  best validation loss   {best[1]:.3f} at step {best[0]}  (perplexity {math.exp(best[1]):.1f}"
                   f" ~ choosing among {math.exp(best[1]):.0f} equally likely tokens, instead of {cfg.vocab_size})")

    out += _section(
        "4. What comes next",
        "The model outputs one score per token; softmax turns scores into probabilities. This is its\n"
        "entire answer: a probability for every possible next token.",
    )
    for text, p in next_token_candidates(model, tok, ids, k=8):
        out.append(f"  {_show(text)!r:<14}{p:>7.1%}  {'#' * round(p * BAR)}")

    logits = last_logits(model, ids)
    out += _section(
        "5. Temperature",
        "Dividing scores by a temperature before softmax sharpens (<1) or flattens (>1) the distribution.\n"
        "Entropy (bits) measures how undecided the model is; 0 = certain.",
    )
    for t, h in temperature_entropies(logits, TEMPERATURES).items():
        out.append(f"  temperature {t:<4}  entropy {h:5.2f} bits  (uniform over all tokens would be {math.log2(cfg.vocab_size):.2f})")

    out += _section(
        f"6. Attention (layer {layer}, averaged over heads)",
        "Each token looks back at earlier tokens (never forward) and mixes in what it finds. Below is,\n"
        "for every token, the token it attends to most and how much of its attention goes there.\n"
        "(The first token can only see itself, so it is always 100%.)",
    )
    for link in attention_links(model, tok, ids, layer):
        q, k = pieces[link.query_position][1], pieces[link.key_position][1]
        out.append(f"  {_show(q)!r:<10} attends most to {_show(k)!r:<10} {link.weight:>6.1%}")

    out += _section(
        "7. The KV cache",
        "To produce each new token the model needs every earlier token's keys and values. Instead of\n"
        "recomputing them, it keeps them. After the first step only ONE new token is fed in.",
    )
    out.append(f"  {'step':<6}{'tokens fed':>11}{'cached positions':>18}   layer-0 key tensor (batch, heads, positions, head_size)")
    for step, row in enumerate(cache_growth(model, ids, steps=3)):
        out.append(f"  {step:<6}{row.tokens_fed:>11}{row.cached_positions:>18}   {row.key_shape}")

    out += _section(
        "8. Generating",
        "Generation is a loop: predict, pick one token, append it, repeat. Greedy always picks the\n"
        "most likely token; sampling draws from the distribution, so it can differ every time.",
    )
    budget = min(sample_tokens, cfg.block_size - len(ids))
    greedy = generate(model, ids, budget, temperature=0.0) if budget > 0 else []
    sampled = generate(model, ids, budget, temperature=0.8, top_k=40, seed=seed) if budget > 0 else []
    out.append(f"  prompt   : {prompt!r}")
    out.append(f"  greedy   : continues with {tok.decode(greedy)!r}")
    out.append(f"  sampled  : continues with {tok.decode(sampled)!r}   (temperature 0.8, top-k 40, seed {seed})")
    return "\n".join(out) + "\n"
