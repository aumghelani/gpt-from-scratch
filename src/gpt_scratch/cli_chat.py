"""`finetune` and `chat` commands (split from cli.py to keep files small)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import torch

from gpt_scratch.chat.reply import generate_reply
from gpt_scratch.chat.sft import build_example, load_conversations, make_batch_source
from gpt_scratch.chat.synthetic import passages_to_conversations
from gpt_scratch.chat.tasks import (
    evaluate_who,
    frequent_speakers,
    parse_speeches,
    speak_conversations,
    split_speeches,
    who_conversations,
)
from gpt_scratch.chat.template import END, Message
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.train.checkpoint import CHECKPOINTS, checkpoint_path, load_checkpoint
from gpt_scratch.train.device import pick_device
from gpt_scratch.train.loop import TrainConfig, train

log = logging.getLogger("gpt_scratch")


def add_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ft = sub.add_parser("finetune", help="supervised fine-tune a BPE run on conversations")
    ft.add_argument("--run", required=True, help="pretrained run directory")
    ft.add_argument("--out", required=True)
    src = ft.add_mutually_exclusive_group()
    src.add_argument("--conversations", help="JSONL of {'messages': [...]}")
    src.add_argument("--from-text", help="derive 'continue the passage' chats from a text file")
    ft.add_argument("--checkpoint", choices=CHECKPOINTS, default="best", help="which base checkpoint to start from")
    ft.add_argument("--steps", type=int, default=300)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--lr", type=float, default=5e-4)
    ft.add_argument("--device", default="auto")
    ft.add_argument("--seed", type=int, default=1337)

    ch = sub.add_parser("chat", help="talk to a fine-tuned run")
    ch.add_argument("--run", required=True)
    ch.add_argument("--checkpoint", choices=CHECKPOINTS, default="best")
    ch.add_argument("--message", required=True)
    ch.add_argument("--system")
    ch.add_argument("--max-new", type=int, default=200)
    ch.add_argument("--temperature", type=float, default=0.7)
    ch.add_argument("--top-k", type=int, default=40)
    ch.add_argument("--seed", type=int)
    ch.add_argument("--device", default="auto")

    mt = sub.add_parser("make-tasks", help="derive who-says-this / speak-as chat tasks from a play-formatted text")
    mt.add_argument("--text", required=True, help="text with 'NAME:' speaker labels, e.g. Tiny Shakespeare")
    mt.add_argument("--out-dir", required=True)
    mt.add_argument("--val-fraction", type=float, default=0.1, help="tail share held out for scoring")
    mt.add_argument("--min-speeches", type=int, default=5, help="only speakers with this many training speeches")
    mt.add_argument("--kinds", nargs="+", choices=("who", "speak"), default=["who", "speak"],
                    help="which task types go in the training set")
    mt.add_argument("--who-lines", type=int, default=2, help="speech lines shown in each who-says-this question")
    mt.add_argument("--seed", type=int, default=1337)

    ev = sub.add_parser("eval-chat", help="score a fine-tuned run on held-out who-says-this questions")
    ev.add_argument("--run", required=True)
    ev.add_argument("--tasks", required=True, help="heldout.jsonl from make-tasks")
    ev.add_argument("--checkpoint", choices=CHECKPOINTS, default="best")
    ev.add_argument("--limit", type=int, help="score only the first N questions")
    ev.add_argument("--device", default="auto")


def _load_model(run: Path, which: str, device: str | torch.device | None = None) -> GPT:
    path = checkpoint_path(run, which)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(payload["config"]))
    load_checkpoint(path, model)
    return model.to(device) if device is not None else model


def _load_bpe(run: Path) -> BPETokenizer | None:
    tok = load_tokenizer(run / "tokenizer.json")
    if isinstance(tok, BPETokenizer) and END in tok.special_tokens:
        return tok
    return None


def cmd_finetune(args: argparse.Namespace) -> int:
    run, out = Path(args.run), Path(args.out)
    tok = _load_bpe(run)
    if tok is None:
        log.error("fine-tuning needs a BPE run with chat tokens (prepare with --tokenizer bpe)")
        return 2
    if args.conversations:
        conversations = load_conversations(args.conversations)
    elif args.from_text:
        text = Path(args.from_text).read_text(encoding="utf-8")
        conversations = [list(c) for c in passages_to_conversations(text)]
    else:
        log.error("provide --conversations or --from-text")
        return 2

    base_path = checkpoint_path(run, args.checkpoint)
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    model_cfg = GPTConfig.from_dict(payload["config"])
    examples = [e for c in conversations if (e := build_example(tok, c, model_cfg.block_size))]
    if len(examples) < 2:
        log.error("only %d usable examples after truncation; need at least 2", len(examples))
        return 2
    n_val = max(1, len(examples) // 10)
    pad = tok.special_id("<|endoftext|>")
    train_src = make_batch_source(examples[:-n_val], pad)
    val_src = make_batch_source(examples[-n_val:], pad)

    model = GPT(model_cfg)
    load_checkpoint(base_path, model)
    cfg = TrainConfig(
        max_steps=args.steps, batch_size=args.batch_size, block_size=model_cfg.block_size,
        max_lr=args.lr, min_lr=args.lr / 10, warmup_steps=max(1, min(args.steps - 1, args.steps // 10)),
        eval_interval=max(1, min(50, args.steps)), eval_batches=5, ckpt_interval=max(1, min(50, args.steps)), seed=args.seed,
        device=args.device, out_dir=str(out),
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "tokenizer.json", out / "tokenizer.json")
    log.info("fine-tuning from %s on %d examples (%d held out)", base_path.name, len(examples) - n_val, n_val)
    result = train(model, train_src, val_src, cfg, model_cfg)
    log.info("done at step %d, last val loss %.4f", result.steps_done, result.val_history[-1][1])
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    run = Path(args.run)
    tok = _load_bpe(run)
    if tok is None:
        log.error("chat needs a BPE run with chat tokens")
        return 2
    device = pick_device(args.device)
    model = _load_model(run, args.checkpoint, device)
    messages = ([Message("system", args.system)] if args.system else []) + [Message("user", args.message)]
    reply = generate_reply(
        model, tok, messages, max_new=args.max_new, temperature=args.temperature,
        top_k=args.top_k, seed=args.seed, device=device,
    )
    sys.stdout.write(reply + "\n")
    return 0


def _write_jsonl(path: Path, conversations: list) -> None:  # type: ignore[type-arg]
    lines = (
        json.dumps({"messages": [{"role": m.role, "content": m.content} for m in convo]}, ensure_ascii=False)
        for convo in conversations
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_make_tasks(args: argparse.Namespace) -> int:
    speeches = parse_speeches(Path(args.text).read_text(encoding="utf-8"))
    if not speeches:
        log.error("no 'NAME:' speaker-labelled speeches found in %s", args.text)
        return 2
    train_speeches, held_speeches = split_speeches(speeches, args.val_fraction)
    speakers = frequent_speakers(train_speeches, args.min_speeches)
    if not speakers:
        log.error("no speaker has %d training speeches; lower --min-speeches", args.min_speeches)
        return 2
    train = []
    if "who" in args.kinds:
        train += who_conversations(train_speeches, speakers, max_lines=args.who_lines)
    if "speak" in args.kinds:
        train += speak_conversations(train_speeches, speakers)
    random.Random(args.seed).shuffle(train)
    held = who_conversations(held_speeches, speakers, max_lines=args.who_lines)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "heldout.jsonl", held)
    sys.stdout.write(
        f"{len(train)} train conversations, {len(held)} held-out who-says-this questions, "
        f"{len(speakers)} speakers (from {len(speeches)} speeches)\n"
    )
    return 0


def cmd_eval_chat(args: argparse.Namespace) -> int:
    run = Path(args.run)
    tok = _load_bpe(run)
    if tok is None:
        log.error("eval-chat needs a BPE run with chat tokens")
        return 2
    cases = [(c[0], c[1]) for c in load_conversations(args.tasks) if len(c) == 2]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        log.error("no usable questions in %s", args.tasks)
        return 2
    device = pick_device(args.device)
    model = _load_model(run, args.checkpoint, device)
    report = evaluate_who(
        cases,
        lambda prompt: generate_reply(model, tok, [Message("user", prompt)], max_new=16, temperature=0.0, device=device),
    )
    sys.stdout.write(
        f"{report.total} questions: {report.correct} correct\n"
        f"accuracy           {report.accuracy:.1%}\n"
        f"majority baseline  {report.majority_baseline:.1%}  (always answer the most common speaker)\n"
    )
    for expected, got in report.mistakes[:5]:
        sys.stdout.write(f"  expected {expected!r:<22} got {got!r}\n")
    return 0
