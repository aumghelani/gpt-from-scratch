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

from gpt_scratch import cli_chat
from gpt_scratch.data.download import fetch_tiny_shakespeare
from gpt_scratch.data.prepare import TOKENIZERS, prepare_dataset
from gpt_scratch.data.shards import load_shard
from gpt_scratch.experiments.ablation import default_variants, render_markdown, run_ablation
from gpt_scratch.model.config import FFNS, NORMS, POS_ENCODINGS, GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.model.inspect import attention_weights
from gpt_scratch.presets import PRESETS, model_config_for, with_steps
from gpt_scratch.report.heatmap import heatmap_svg
from gpt_scratch.report.logs import read_series
from gpt_scratch.report.svg import delta_bar_svg, line_chart_svg
from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.train.checkpoint import load_checkpoint
from gpt_scratch.train.device import pick_device
from gpt_scratch.train.loop import TrainConfig, train
from gpt_scratch.train.samples import make_sample_hook

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

    ab = sub.add_parser("ablate", help="train architecture variants and compare validation loss")
    ab.add_argument("--data", required=True)
    ab.add_argument("--out", required=True)
    ab.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ab.add_argument("--steps", type=int)
    ab.add_argument("--device", default="auto")
    ab.add_argument("--seed", type=int, default=1337)
    ab.add_argument("--variants", help="comma-separated subset of: baseline,+rope,+rmsnorm,+swiglu")

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
    pl = sub.add_parser("plot", help="draw training curves from CSV logs as an SVG")
    pl.add_argument("--log", action="append", required=True, metavar="NAME=PATH")
    pl.add_argument("--column", default="loss", help="loss, val_loss, lr, grad_norm, ...")
    pl.add_argument("--out", required=True)
    pl.add_argument("--title", default="Training curve")
    pl.add_argument("--theme", choices=("light", "dark"), default="light")

    pa = sub.add_parser("plot-ablation", help="draw ablation deltas vs baseline as an SVG")
    pa.add_argument("--results", required=True, help="results.json from `ablate`")
    pa.add_argument("--out", required=True)
    pa.add_argument("--theme", choices=("light", "dark"), default="light")

    at = sub.add_parser("attention", help="draw what one attention head looks at (SVG heatmap)")
    at.add_argument("--run", required=True)
    at.add_argument("--text", required=True)
    at.add_argument("--layer", type=int, default=0)
    at.add_argument("--head", type=int, help="head index (default: average over heads)")
    at.add_argument("--out", required=True)
    at.add_argument("--theme", choices=("light", "dark"), default="light")

    sv = sub.add_parser("serve", help="serve a run over HTTP with a web UI")
    sv.add_argument("--run", required=True)
    sv.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--device", default="auto")

    cli_chat.add_parsers(sub)
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
    model_cfg = model_config_for(preset, meta["vocab_size"], args.pos, args.norm, args.ffn)
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
    sampler = make_sample_hook(load_tokenizer(run_dir / "tokenizer.json"), run_dir)
    log.info("model: %d parameters", model.num_params())
    result = train(
        model, load_shard(data_dir / "train.bin"), load_shard(data_dir / "val.bin"), cfg, model_cfg,
        resume_from=run_dir / "latest.pt" if args.resume else None, stop_after=args.stop_after,
        on_eval=sampler,
    )
    final_val = result.val_history[-1][1] if result.val_history else float("nan")
    log.info("done at step %d, last val loss %.4f", result.steps_done, final_val)
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    available = {v.name: v for v in default_variants()}
    chosen = list(available.values())
    if args.variants:
        names = [n.strip() for n in args.variants.split(",")]
        unknown = [n for n in names if n not in available]
        if unknown:
            log.error("unknown variants: %s (choose from %s)", unknown, sorted(available))
            return 2
        chosen = [available[n] for n in names]
    results = run_ablation(args.data, args.out, args.preset, args.steps, chosen, args.device, args.seed)
    sys.stdout.write(render_markdown(results))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    series = {}
    for spec in args.log:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            log.error("--log must look like NAME=PATH, got %r", spec)
            return 2
        series[name] = read_series(path, args.column)
    svg = line_chart_svg(series, args.title, "step", args.column, theme=args.theme)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    return 0


def _cmd_plot_ablation(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    base = next((r["val_loss"] for r in rows if r["name"] == "baseline"), None)
    if base is None:
        log.error("results contain no 'baseline' row")
        return 2
    items = [(r["name"], r["val_loss"] - base) for r in rows if r["name"] not in ("bigram", "baseline")]
    svg = delta_bar_svg(items, "Validation loss change vs baseline", "Δ val loss", theme=args.theme)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    return 0


def _cmd_attention(args: argparse.Namespace) -> int:
    run = Path(args.run)
    payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(payload["config"]))
    load_checkpoint(run / "latest.pt", model)
    tok = load_tokenizer(run / "tokenizer.json")
    ids = tok.encode(args.text)[: model.config.block_size]
    maps = attention_weights(model, torch.tensor([ids]))
    if not 0 <= args.layer < len(maps):
        log.error("layer must be in [0, %d), got %d", len(maps), args.layer)
        return 2
    layer = maps[args.layer]
    if args.head is not None and not 0 <= args.head < layer.size(0):
        log.error("head must be in [0, %d), got %d", layer.size(0), args.head)
        return 2
    weights = layer.mean(0) if args.head is None else layer[args.head]
    which = "mean of heads" if args.head is None else f"head {args.head}"
    svg = heatmap_svg(weights.tolist(), [tok.decode([i]) for i in ids], f"Layer {args.layer}, {which}", args.theme)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn  # imported lazily: only needed for this command

    from gpt_scratch.serve.app import create_app

    try:
        app = create_app(args.run, args.device)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
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
    handlers = {
        "prepare": _cmd_prepare, "train": _cmd_train, "ablate": _cmd_ablate, "sample": _cmd_sample,
        "serve": _cmd_serve, "attention": _cmd_attention, "plot": _cmd_plot, "plot-ablation": _cmd_plot_ablation, "finetune": cli_chat.cmd_finetune, "chat": cli_chat.cmd_chat,
    }
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc.filename)
        return 2


if __name__ == "__main__":
    sys.exit(main())
