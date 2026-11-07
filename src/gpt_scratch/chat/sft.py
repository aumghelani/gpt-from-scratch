"""Supervised fine-tuning data: conversations -> (x, y) with loss only on assistant tokens."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from gpt_scratch.chat.template import Message, encode_chat
from gpt_scratch.data.loader import BatchSource
from gpt_scratch.tokenizer.bpe import BPETokenizer

IGNORE_INDEX = -100  # torch cross_entropy ignores this target value


@dataclass(frozen=True)
class SFTExample:
    x: tuple[int, ...]
    y: tuple[int, ...]


def build_example(tok: BPETokenizer, messages: Sequence[Message], block_size: int) -> SFTExample | None:
    """Next-token pairs for one conversation; None if no assistant target survives truncation."""
    ids, mask = encode_chat(tok, messages)
    x = ids[:-1][:block_size]
    y = [t if learn else IGNORE_INDEX for t, learn in zip(ids[1:], mask[1:], strict=True)][:block_size]
    if all(label == IGNORE_INDEX for label in y):
        return None
    return SFTExample(tuple(x), tuple(y))


def make_batch_source(examples: Sequence[SFTExample], pad_id: int) -> BatchSource:
    """Random batches of right-padded examples (pad inputs with `pad_id`, targets with IGNORE)."""
    if not examples:
        raise ValueError("need at least one SFT example")
    pool = tuple(examples)

    def source(
        batch_size: int, block_size: int, generator: torch.Generator, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        picks = torch.randint(0, len(pool), (batch_size,), generator=generator).tolist()
        x = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
        y = torch.full((batch_size, block_size), IGNORE_INDEX, dtype=torch.long)
        for row, i in enumerate(picks):
            n = min(len(pool[i].x), block_size)
            x[row, :n] = torch.tensor(pool[i].x[:n])
            y[row, :n] = torch.tensor(pool[i].y[:n])
        return x.to(device), y.to(device)

    return source


def load_conversations(path: str | Path) -> list[list[Message]]:
    """Read JSONL where each line is {"messages": [{"role": ..., "content": ...}, ...]}."""
    conversations: list[list[Message]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)["messages"]
            conversations.append([Message(m["role"], m["content"]) for m in raw])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"bad conversation at line {number}: {exc}") from exc
    return conversations
