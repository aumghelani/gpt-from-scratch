"""Model + tokenizer wrapper that turns requests into streams of decoded text."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import torch

from gpt_scratch.chat.template import END, Message, encode_chat
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.sample.generate import stream_generate
from gpt_scratch.serve.schemas import ChatMessage, SamplingParams
from gpt_scratch.tokenizer import load_tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.stream import make_stream_decoder
from gpt_scratch.train.checkpoint import load_checkpoint
from gpt_scratch.train.device import pick_device


class PromptError(ValueError):
    """The prompt cannot be tokenized by this model (reported to the client as HTTP 400)."""


class Engine:
    def __init__(self, run_dir: str | Path, device: str = "auto") -> None:
        run = Path(run_dir)
        ckpt = run / "latest.pt"
        if not ckpt.exists() or not (run / "tokenizer.json").exists():
            raise FileNotFoundError(f"{run} is not a trained run directory")
        self.device = pick_device(device)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig.from_dict(payload["config"]))
        load_checkpoint(ckpt, self.model)
        self.model.to(self.device).eval()
        self.tok = load_tokenizer(run / "tokenizer.json")
        self.is_chat = isinstance(self.tok, BPETokenizer) and END in self.tok.special_tokens
        self._lock = threading.Lock()  # one generation at a time on a single model

    @property
    def num_params(self) -> int:
        return self.model.num_params()

    @property
    def block_size(self) -> int:
        return self.model.config.block_size

    def _stream(self, prompt_ids: list[int], params: SamplingParams, stop: tuple[int, ...]) -> Iterator[str]:
        feed, flush = make_stream_decoder(self.tok)
        with self._lock:
            for token_id in stream_generate(
                self.model, prompt_ids, params.max_new_tokens, temperature=params.temperature,
                top_k=params.top_k, top_p=params.top_p, stop_ids=stop, seed=params.seed,
                device=self.device,
            ):
                piece = feed(token_id)
                if piece:
                    yield piece
        tail = flush()
        if tail:
            yield tail

    def generate(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        try:
            ids = (
                self.tok.encode(prompt, allow_special=False)
                if isinstance(self.tok, BPETokenizer)
                else self.tok.encode(prompt)
            )
        except KeyError as exc:
            raise PromptError("prompt contains characters outside the model's vocabulary") from exc
        return self._stream(ids, params, ())

    def chat(self, messages: list[ChatMessage], params: SamplingParams) -> Iterator[str]:
        if not self.is_chat or not isinstance(self.tok, BPETokenizer):
            raise PromptError("this model was not fine-tuned for chat")
        ids, _ = encode_chat(
            self.tok, [Message(m.role, m.content) for m in messages], add_generation_prompt=True
        )
        return self._stream(ids, params, (self.tok.special_id(END),))
