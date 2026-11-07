#!/usr/bin/env bash
# Strict local gate: lint then tests; exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/ruff check src tests --output-format concise
.venv/bin/pytest "$@"
