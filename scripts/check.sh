#!/usr/bin/env bash
# Strict local gate: lint then tests; exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
# Guard: a .gitignore rule once silently excluded a whole source package. Never again.
ignored=$(git ls-files --others --ignored --exclude-standard -- src tests scripts \
  | grep -E '\.(py|sh)$' | grep -v -E '__pycache__|\.egg-info' || true)
if [ -n "$ignored" ]; then
  echo "ERROR: source files are git-ignored (check .gitignore):" >&2
  echo "$ignored" >&2
  exit 1
fi
.venv/bin/ruff check src tests --output-format concise
.venv/bin/pytest "$@"
