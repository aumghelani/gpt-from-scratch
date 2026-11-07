"""Flush-on-write CSV metrics logger."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class CSVLogger:
    def __init__(self, path: str | Path, fields: Sequence[str]) -> None:
        self.fields = tuple(fields)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **metrics: Any) -> None:
        unknown = set(metrics) - set(self.fields)
        if unknown:
            raise KeyError(f"unknown log fields: {sorted(unknown)}")
        self._writer.writerow(metrics)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
