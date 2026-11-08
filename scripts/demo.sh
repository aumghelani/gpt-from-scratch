#!/usr/bin/env bash
# Replay the demo from trained checkpoints. Usage: scripts/demo.sh [tour|samples|bench|serve|all]
# Checkpoints are not in git (they are large). Create them with scripts/reproduce.sh (the flagship step is the long one; the README lists its measured time).
set -euo pipefail
cd "$(dirname "$0")/.."
GPT=".venv/bin/gpt-scratch"
FLAGSHIP="runs/flagship-best"
CHAT="runs/chat-who"

need() {
  if [ ! -f "$1/best.pt" ] && [ ! -f "$1/latest.pt" ]; then
    echo "Missing checkpoint: $1  (run scripts/reproduce.sh first)" >&2
    exit 2
  fi
}

step_samples() {
  need "$FLAGSHIP"
  echo "== before training vs after (prompt ROMEO:) =="
  head -c 400 demo/assets/samples/step-000000-untrained.txt; echo; echo "-- after --"
  "$GPT" sample --run "$FLAGSHIP" --prompt "ROMEO:" --max-new 300 --temperature 0.8 --seed 7
}
step_tour() { need "$FLAGSHIP"; "$GPT" tour --run "$FLAGSHIP" --prompt "ROMEO: But soft" --layer 3; }
step_bench() { need "$FLAGSHIP"; "$GPT" bench-cache --run "$FLAGSHIP" --prompt "ROMEO:" --new-tokens 240; }
step_serve() {
  need "$FLAGSHIP"; need "$CHAT"
  echo "Open http://127.0.0.1:8000 - use the model dropdown, Enter sends, Stop and Redo work."
  "$GPT" serve --run "$FLAGSHIP" "$CHAT"
}

case "${1:-all}" in
  samples) step_samples ;;
  tour) step_tour ;;
  bench) step_bench ;;
  serve) step_serve ;;
  all) step_samples; step_tour; step_bench; step_serve ;;
  *) echo "usage: $0 [tour|samples|bench|serve|all]" >&2; exit 64 ;;
esac
