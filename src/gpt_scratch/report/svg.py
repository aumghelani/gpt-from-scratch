"""Minimal SVG chart writers (no plotting dependency). All text is XML-escaped."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape

PALETTE = ("#2f7d6a", "#c0572b", "#3b6ea5", "#8a6bb8", "#b08d2c")
THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1d1d1b", "muted": "#6b6b66", "grid": "#e4e3dd"},
    "dark": {"bg": "#161715", "fg": "#ecebe6", "muted": "#9a9a93", "grid": "#33342f"},
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick positions covering [lo, hi]."""
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    raw = (hi - lo) / max(n - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value < hi + step * 0.999:
        ticks.append(round(value, 10))
        value += step
        if len(ticks) > 20:
            break
    return ticks


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def line_chart_svg(
    series: Mapping[str, Sequence[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 440,
    theme: str = "light",
) -> str:
    named = {k: list(v) for k, v in series.items() if len(v) > 0}
    if not named:
        raise ValueError("no data points to plot")
    points = [p for pts in named.values() for p in pts]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError("data contains NaN or infinity")
    c = THEMES[theme]
    left, right, top, bottom = 64, 24, 52, 56
    pw, ph = width - left - right, height - top - bottom
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    yt = nice_ticks(min(ys), max(ys), 5)
    xt = nice_ticks(min(xs), max(xs), 6)
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]

    def px(x: float) -> float:
        return left + (x - x0) / ((x1 - x0) or 1.0) * pw

    def py(y: float) -> float:
        return top + ph - (y - y0) / ((y1 - y0) or 1.0) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
        f'<text x="{left}" y="30" {FONT} font-size="18" font-weight="600" fill="{c["fg"]}">{escape(title)}</text>',
    ]
    for t in yt:
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="{c["grid"]}"/>')
        out.append(f'<text x="{left - 8}" y="{py(t) + 4:.1f}" text-anchor="end" {FONT} font-size="12" fill="{c["muted"]}">{_fmt(t)}</text>')
    for t in xt:
        out.append(f'<text x="{px(t):.1f}" y="{top + ph + 20}" text-anchor="middle" {FONT} font-size="12" fill="{c["muted"]}">{_fmt(t)}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{height - 12}" text-anchor="middle" {FONT} font-size="13" fill="{c["fg"]}">{escape(x_label)}</text>')
    out.append(f'<text transform="translate(16 {top + ph / 2}) rotate(-90)" text-anchor="middle" {FONT} font-size="13" fill="{c["fg"]}">{escape(y_label)}</text>')
    for i, (name, pts) in enumerate(named.items()):
        color = PALETTE[i % len(PALETTE)]
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        if len(pts) == 1:
            out.append(f'<circle cx="{px(pts[0][0]):.1f}" cy="{py(pts[0][1]):.1f}" r="4" fill="{color}"/>')
        else:
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        ly = top + 8 + i * 20
        out.append(f'<line x1="{left + pw - 150}" x2="{left + pw - 128}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{left + pw - 122}" y="{ly + 4}" {FONT} font-size="13" fill="{c["fg"]}">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def delta_bar_svg(
    items: Sequence[tuple[str, float]], title: str, x_label: str, width: int = 760, theme: str = "light"
) -> str:
    """Horizontal bars around zero. Negative (lower loss) is drawn in the 'better' colour."""
    if not items:
        raise ValueError("no bars to draw")
    c = THEMES[theme]
    row, left, right, top = 44, 150, 70, 60
    height = top + row * len(items) + 50
    span = max(abs(v) for _, v in items) or 1.0
    half = (width - left - right) / 2
    zero_x = left + half
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
        f'<text x="24" y="32" {FONT} font-size="18" font-weight="600" fill="{c["fg"]}">{escape(title)}</text>',
        f'<line x1="{zero_x}" x2="{zero_x}" y1="{top - 8}" y2="{top + row * len(items)}" stroke="{c["muted"]}"/>',
    ]
    for i, (label, value) in enumerate(items):
        y = top + i * row
        length = abs(value) / span * (half - 70)  # leave room for the value label beside the bar
        x = zero_x - length if value < 0 else zero_x
        color = PALETTE[0] if value < 0 else PALETTE[1]
        out.append(f'<text x="{left - 12}" y="{y + 24}" text-anchor="end" {FONT} font-size="14" fill="{c["fg"]}">{escape(label)}</text>')
        out.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{max(length, 1):.1f}" height="28" rx="3" fill="{color}"/>')
        tx = x - 8 if value < 0 else x + length + 8
        anchor = "end" if value < 0 else "start"
        out.append(f'<text x="{tx:.1f}" y="{y + 25}" text-anchor="{anchor}" {FONT} font-size="13" fill="{c["fg"]}">{value:+.3f}</text>')
    out.append(f'<text x="{zero_x}" y="{height - 14}" text-anchor="middle" {FONT} font-size="12" fill="{c["muted"]}">{escape(x_label)} (left of zero = better)</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
