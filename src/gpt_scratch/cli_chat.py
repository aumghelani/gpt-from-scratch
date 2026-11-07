"""`finetune` and `chat` commands (split from cli.py to keep files small)."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch

from gpt_scratch.chat.sft import build_example, load_conversations, make_batch_source
from gpt_scratch.chat.synthetic import passages_to_conversations
from gpt_scratch.chat.template import END, Message, encode_chat
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.stream import make_stream_decoder
from gpt_scratch.train.checkpoint import load_checkpoint
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
    ft.add_argument("--steps", type=int, default=300)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--lr", type=float, default=5e-4)
    ft.add_argument("--device", default="auto")
    ft.add_argument("--seed", type=int, default=1337)

    ch = sub.add_parser("chat", help="talk to a fine-tuned run")
    ch.add_argument("--run", required=True)
    ch.add_argument("--message", required=True)
    ch.add_argument("--system")
    ch.add_argument("--max-new", type=int, default=200)
    ch.add_argument("--temperature", type=float, default=0.7)
    ch.add_argument("--top-k", type=int, default=40)
    ch.add_argument("--seed", type=int)
    ch.add_argument("--device", default="auto")


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

    payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
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
    load_checkpoint(run / "latest.pt", model)
    cfg = TrainConfig(
        max_steps=args.steps, batch_size=args.batch_size, block_size=model_cfg.block_size,
        max_lr=args.lr, min_lr=args.lr / 10, warmup_steps=max(1, min(args.steps - 1, args.steps // 10)),
        eval_interval=max(1, min(50, args.steps)), eval_batches=5, seed=args.seed,
        device=args.device, out_dir=str(out),
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "tokenizer.json", out / "tokenizer.json")
    log.info("fine-tuning on %d examples (%d held out)", len(examples) - n_val, n_val)
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
    payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(payload["config"]))
    load_checkpoint(run / "latest.pt", model)
    model.to(device)

    messages = ([Message("system", args.system)] if args.system else []) + [Message("user", args.message)]
    prompt_ids, _ = encode_chat(tok, messages, add_generation_prompt=True)
    feed, flush = make_stream_decoder(tok)
    for token_id in stream_generate(
        model, prompt_ids, args.max_new, temperature=args.temperature, top_k=args.top_k,
        stop_ids=(tok.special_id(END),), seed=args.seed, device=device,
    ):
        sys.stdout.write(feed(token_id))
        sys.stdout.flush()
    sys.stdout.write(flush() + "\n")
    return 0
