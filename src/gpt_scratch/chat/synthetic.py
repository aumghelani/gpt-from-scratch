"""Derive 'continue the passage' conversations from plain text (for demos and smoke tests)."""

from __future__ import annotations

import re

from gpt_scratch.chat.template import Message

Conversation = tuple[Message, Message]


def passages_to_conversations(text: str, min_lines: int = 3, limit: int | None = None) -> list[Conversation]:
    """Each blank-line-separated passage becomes: user = its first line, assistant = the rest."""
    out: list[Conversation] = []
    for passage in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in passage.splitlines() if ln.strip()]
        if len(lines) < min_lines:
            continue
        prompt = f"Continue this passage:\n{lines[0]}"
        out.append((Message("user", prompt), Message("assistant", "\n".join(lines[1:]))))
        if limit is not None and len(out) >= limit:
            break
    return out
