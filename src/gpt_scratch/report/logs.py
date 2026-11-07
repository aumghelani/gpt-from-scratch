"""Read metric series out of training CSV logs."""

from __future__ import annotations

import csv
from pathlib import Path


def read_series(path: str | Path, column: str, x_column: str = "step") -> list[tuple[float, float]]:
    """Return (x, y) pairs for `column`, skipping rows where it is blank."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column {column!r} not found in {path} (have {reader.fieldnames})")
        return [(float(row[x_column]), float(row[column])) for row in reader if row[column] != ""]
