"""Checksum-verified dataset download. Raw data is never committed to git."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


class ChecksumError(RuntimeError):
    """Downloaded bytes did not match the expected SHA-256."""


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: str | Path, expected_sha256: str) -> Path:
    """Fetch `url` to `dest` atomically; skip if `dest` already matches the checksum."""
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out)
        actual = sha256_of(part)
        if actual != expected_sha256:
            raise ChecksumError(f"sha256 mismatch for {url}: expected {expected_sha256}, got {actual}")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def fetch_tiny_shakespeare(dest: str | Path = "data/raw/tinyshakespeare.txt") -> Path:
    return download_file(TINY_SHAKESPEARE_URL, dest, TINY_SHAKESPEARE_SHA256)
