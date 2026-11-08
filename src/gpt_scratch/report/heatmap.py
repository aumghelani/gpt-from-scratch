"""Attention heatmap as SVG: rows = query position, columns = key position."""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape

from gpt_scratch.report.svg import FONT, THEMES

_VISIBLE = {" ": "·", "\n": "⏎", "\t": "⇥", "\r": "↵"}
MAX_TOKENS = 64


def visible_token(token: str) -> str:
    """Make whitespace tokens legible in axis labels."""
    return _VISIBLE.get(token, token)


def heatmap_svg(
    weights: Sequence[Sequence[float]], labels: Sequence[str], title: str, theme: str = "light",
    cell: int = 22,
) -> str:
    n = len(weights)
    if n == 0 or any(len(row) != n for row in weights):
        raise ValueError("weights must be a non-empty square matrix")
    if len(labels) != n:
        raise ValueError(f"need {n} labels, got {len(labels)}")
    if n > MAX_TOKENS:
        raise ValueError(f"at most {MAX_TOKENS} tokens can be drawn")
    c = THEMES[theme]
    margin_left, margin_top = 56, 92
    size = n * cell
    width, height = margin_left + size + 24, margin_top + size + 24
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
        f'<text x="16" y="26" {FONT} font-size="16" font-weight="600" fill="{c["fg"]}">{escape(title)}</text>',
        f'<text x="16" y="46" {FONT} font-size="11" fill="{c["muted"]}">row = token asking · column = token it looks at</text>',
    ]
    for i, label in enumerate(labels):
        text = escape(visible_token(label))
        x = margin_left + i * cell + cell / 2
        out.append(f'<text x="{x}" y="{margin_top - 6}" text-anchor="middle" {FONT} font-size="11" fill="{c["fg"]}">{text}</text>')
        y = margin_top + i * cell + cell / 2 + 4
        out.append(f'<text x="{margin_left - 6}" y="{y}" text-anchor="end" {FONT} font-size="11" fill="{c["fg"]}">{text}</text>')
    accent = "#2f7d6a" if theme == "light" else "#7fb8a4"
    for r in range(n):
        for k in range(n):
            w = max(0.0, min(1.0, float(weights[r][k])))
            out.append(
                f'<rect class="cell" x="{margin_left + k * cell}" y="{margin_top + r * cell}" '
                f'width="{cell - 1}" height="{cell - 1}" fill="{accent}" fill-opacity="{w:.3f}" '
                f'stroke="{c["grid"]}" stroke-width="0.5"/>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"
