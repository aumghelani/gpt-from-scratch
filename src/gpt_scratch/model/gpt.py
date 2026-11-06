"""The GPT language model: embeddings -> N pre-norm blocks -> final norm -> LM head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from gpt_scratch.model.attention import KV
from gpt_scratch.model.block import Block
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.norm import make_norm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = (
            nn.Embedding(config.block_size, config.n_embd) if config.pos_encoding == "learned" else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = make_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("attn.proj.weight", "ffn.down.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total trainable parameters (tied weights are counted once)."""
        return sum(p.numel() for p in self.parameters())

    def _embed(self, idx: torch.Tensor, offset: int) -> torch.Tensor:
        seq = idx.size(1)
        if offset + seq > self.config.block_size:
            raise ValueError(f"sequence length {offset + seq} exceeds block_size {self.config.block_size}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(offset, offset + seq, device=idx.device))
        return self.drop(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self._embed(idx, offset=0)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_cached(
        self, idx: torch.Tensor, past: tuple[KV, ...] | None = None
    ) -> tuple[torch.Tensor, tuple[KV, ...]]:
        """Incremental decoding: feed new tokens plus per-layer (k, v); returns logits and new cache."""
        offset = 0 if past is None else past[0][0].size(2)
        x = self._embed(idx, offset)
        present: list[KV] = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past=None if past is None else past[i], return_kv=True)
            present.append(kv)
        return self.lm_head(self.final_norm(x)), tuple(present)
