"""Command-line interface: prepare data, train, sample."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from gpt_scratch.data.download import fetch_tiny_shakespeare
from gpt_scratch.data.prepare import TOKENIZERS, prepare_dataset
from gpt_scratch.data.shards import load_shard
from gpt_scratch.model.config import FFNS, NORMS, POS_ENCODINGS, GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.presets import PRESETS, with_steps
from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.train.checkpoint import load_checkpoint
from gpt_scratch.train.device import pick_device
from gpt_scratch.train.loop import TrainConfig, train

log = logging.getLogger("gpt_scratch")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt-scratch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="clean + tokenize a text file into train/val shards")
    prep.add_argument("--input", help="path to a UTF-8 text file")
    prep.add_argument("--tiny-shakespeare", action="store_true", help="download the Tiny Shakespeare corpus")
    prep.add_argument("--out", required=True)
    prep.add_argument("--tokenizer", choices=TOKENIZERS, default="char")
    prep.add_argument("--vocab-size", type=int, default=512, help="BPE vocabulary size")

    tr = sub.add_parser("train", help="train a GPT")
    tr.add_argument("--data", required=True, help="directory made by `prepare`")
    tr.add_argument("--out", required=True, help="run directory (checkpoints, log, tokenizer)")
    tr.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    tr.add_argument("--steps", type=int, help="override preset max_steps")
    tr.add_argument("--pos", choices=POS_ENCODINGS, default="learned")
    tr.add_argument("--norm", choices=NORMS, default="layernorm")
    tr.add_argument("--ffn", choices=FFNS, default="gelu")
    tr.add_argument("--device", default="auto")
    tr.add_argument("--seed", type=int, default=1337)
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--stop-after", type=int, help="stop after N steps (for interruption tests)")

    sm = sub.add_parser("sample", help="generate text from a trained run")
    sm.add_argument("--run", required=True)
    sm.add_argument("--checkpoint", default="latest.pt")
    sm.add_argument("--prompt", default="")
    sm.add_argument("--max-new", type=int, default=200)
    sm.add_argument("--temperature", type=float, default=0.8)
    sm.add_argument("--top-k", type=int)
    sm.add_argument("--top-p", type=float, default=1.0)
    sm.add_argument("--seed", type=int)
    sm.add_argument("--device", default="auto")
    return parser


def _cmd_prepare(args: argparse.Namespace) -> int:
    if args.tiny_shakespeare:
        source = fetch_tiny_shakespeare(Path(args.out) / "raw" / "tinyshakespeare.txt")
    elif args.input:
        source = Path(args.input)
    else:
        log.error("provide --input or --tiny-shakespeare")
        return 2
    meta = prepare_dataset(source, args.out, args.tokenizer, args.vocab_size)
    sys.stdout.write(json.dumps(meta) + "\n")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    data_dir, run_dir = Path(args.data), Path(args.out)
    try:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.error("no prepared dataset at %s (run `prepare` first)", data_dir)
        return 2
    preset = PRESETS[args.preset]
    if args.steps:
        preset = with_steps(preset, args.steps)
    model_cfg = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=preset.block_size, n_layer=preset.n_layer,
        n_head=preset.n_head, n_embd=preset.n_embd, dropout=preset.dropout,
        pos_encoding=args.pos, norm=args.norm, ffn=args.ffn,
    )
    cfg = TrainConfig(
        max_steps=preset.max_steps, batch_size=preset.batch_size, block_size=preset.block_size,
        max_lr=preset.max_lr, min_lr=preset.min_lr, warmup_steps=preset.warmup_steps,
        eval_interval=preset.eval_interval, eval_batches=preset.eval_batches,
        seed=args.seed, device=args.device, out_dir=str(run_dir),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "tokenizer.json", run_dir / "tokenizer.json")
    torch.manual_seed(args.seed)
    model = GPT(model_cfg)
    log.info("model: %d parameters", model.num_params())
    result = train(
        model, load_shard(data_dir / "train.bin"), load_shard(data_dir / "val.bin"), cfg, model_cfg,
        resume_from=run_dir / "latest.pt" if args.resume else None, stop_after=args.stop_after,
    )
    final_val = result.val_history[-1][1] if result.val_history else float("nan")
    log.info("done at step %d, last val loss %.4f", result.steps_done, final_val)
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    run = Path(args.run)
    device = pick_device(args.device)
    payload = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(payload["config"]))
    load_checkpoint(run / args.checkpoint, model)
    model.to(device)
    tok = load_tokenizer(run / "tokenizer.json")
    prompt_ids = tok.encode(args.prompt) if args.prompt else [0]
    sys.stdout.write(args.prompt)
    for token_id in stream_generate(
        model, prompt_ids, args.max_new, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, device=device,
    ):
        sys.stdout.write(tok.decode([token_id]))
        sys.stdout.flush()
    sys.stdout.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _build_parser().parse_args(argv)
    handlers = {"prepare": _cmd_prepare, "train": _cmd_train, "sample": _cmd_sample}
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc.filename)
        return 2


if __name__ == "__main__":
    sys.exit(main())
