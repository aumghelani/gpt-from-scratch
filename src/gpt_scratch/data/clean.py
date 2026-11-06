"""Text cleaning applied before tokenization."""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Return a normalised copy: NFC, LF endings, no control chars, tidy whitespace."""
    out = unicodedata.normalize("NFC", text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _CONTROL.sub("", out)
    out = _TRAILING_WS.sub("", out)
    return _BLANK_RUNS.sub("\n\n", out)
