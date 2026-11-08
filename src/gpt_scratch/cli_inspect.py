"""`bench-cache` and `compare-tokenizer`: commands that measure rather than train."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gpt_scratch.experiments.kv_bench import benchmark_cache
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.compare import Compression, measure, split_for_comparison
from gpt_scratch.train.checkpoint import load_checkpoint
from gpt_scratch.train.device import pick_device

log = logging.getLogger("gpt_scratch")


def add_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    bc = sub.add_parser("bench-cache", help="time generation with and without the KV cache")
    bc.add_argument("--run", required=True)
    bc.add_argument("--prompt", default="\n")
    bc.add_argument("--new-tokens", type=int, default=200)
    bc.add_argument("--device", default="auto")

    tr = sub.add_parser("tour", help="walk one prompt through tokens, model, attention, sampling and the KV cache")
    tr.add_argument("--run", required=True)
    tr.add_argument("--prompt", required=True)
    tr.add_argument("--layer", type=int, default=0, help="which layer's attention to show")
    tr.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    tr.add_argument("--device", default="auto")

    ct = sub.add_parser("compare-tokenizer", help="bytes/token of own BPE (several vocab sizes) vs tiktoken")
    ct.add_argument("--text", required=True)
    ct.add_argument("--vocab-sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ct.add_argument("--tiktoken", metavar="ENCODING", help="also measure a tiktoken encoding, e.g. gpt2")
    ct.add_argument("--test-fraction", type=float, default=0.1)


def cmd_bench_cache(args: argparse.Namespace) -> int:
    run = Path(args.run)
    device = pick_device(args.device)
    tok = load_tokenizer(run / "tokenizer.json")
    ckpt = run / "latest.pt"
    import torch

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(payload["config"]))
    load_checkpoint(ckpt, model)
    prompt = tok.encode(args.prompt)
    result = benchmark_cache(model.eval(), prompt, args.new_tokens, device)
    sys.stdout.write(
        f"{result.new_tokens} new tokens on {device}\n"
        f"with KV cache     {result.cached_seconds:8.3f} s\n"
        f"without KV cache  {result.uncached_seconds:8.3f} s\n"
        f"speedup           {result.speedup:8.1f}x\n"
        f"identical tokens: {'yes' if result.identical else 'NO'}\n"
    )
    return 0 if result.identical else 1


def _row(c: Compression) -> str:
    return f"{c.name:<26}{c.vocab_size:>8}{c.tokens:>10}{c.bytes_per_token:>13.2f}\n"


def cmd_compare_tokenizer(args: argparse.Namespace) -> int:
    text = Path(args.text).read_text(encoding="utf-8")
    train, test = split_for_comparison(text, args.test_fraction)
    rows: list[Compression] = []
    for size in args.vocab_sizes:
        log.info("training own BPE with vocab %d on %d characters", size, len(train))
        tok = BPETokenizer.train(train, size)
        rows.append(measure(f"own BPE (vocab {size})", tok.vocab_size, tok.encode, test))
    if args.tiktoken:
        import tiktoken

        enc = tiktoken.get_encoding(args.tiktoken)
        rows.append(
            measure(f"tiktoken {args.tiktoken}", enc.n_vocab, lambda s: enc.encode(s, disallowed_special=()), test)
        )
    sys.stdout.write(f"measured on the held-out last {args.test_fraction:.0%} ({len(test.encode())} bytes)\n")
    sys.stdout.write(f"{'tokenizer':<26}{'vocab':>8}{'tokens':>10}{'bytes/token':>13}\n")
    for row in rows:
        sys.stdout.write(_row(row))
    return 0


def cmd_tour(args: argparse.Namespace) -> int:
    from gpt_scratch.cli_chat import _load_model
    from gpt_scratch.tour.render import best_val_loss, render_tour

    run = Path(args.run)
    if not args.prompt:
        log.error("--prompt must not be empty")
        return 2
    model = _load_model(run, args.checkpoint, pick_device(args.device))
    if not 0 <= args.layer < len(model.blocks):
        log.error("--layer must be between 0 and %d for this model", len(model.blocks) - 1)
        return 2
    tok = load_tokenizer(run / "tokenizer.json")
    sys.stdout.write(render_tour(model, tok, args.prompt, layer=args.layer, best=best_val_loss(run)))
    return 0
