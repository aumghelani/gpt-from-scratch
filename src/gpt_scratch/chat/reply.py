"""Generate one assistant reply to a conversation (shared by the chat CLI and the task scorer)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gpt_scratch.chat.template import END, Message, encode_chat
from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.stream import make_stream_decoder


def generate_reply(
    model: nn.Module,
    tok: BPETokenizer,
    messages: Sequence[Message],
    *,
    max_new: int = 200,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Encode `messages` with the generation prompt, sample until `<|end|>`, return the decoded text."""
    prompt_ids, _ = encode_chat(tok, messages, add_generation_prompt=True)
    feed, flush = make_stream_decoder(tok)
    pieces = [
        feed(token_id)
        for token_id in stream_generate(
            model, prompt_ids, max_new, temperature=temperature, top_k=top_k,
            stop_ids=(tok.special_id(END),), seed=seed, device=device,
        )
    ]
    pieces.append(flush())
    return "".join(pieces)
