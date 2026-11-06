"""Bigram baseline: next-token logits come straight from an embedding row."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.table(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss
