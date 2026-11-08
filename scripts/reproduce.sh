#!/usr/bin/env bash
# Rebuild every checkpoint used by the demo from scratch. Run end to end from a fresh clone on 2026-11-06: all four steps pass.
# The char model reproduced exactly (best val 1.463 at step 1,500); the BPE and chat numbers moved slightly between runs (see comments).
# Overfitting means training is deliberately short; the best checkpoint (lowest validation loss) is what gets used.
set -euo pipefail
cd "$(dirname "$0")/.."
GPT=".venv/bin/gpt-scratch"
DEVICE="${DEVICE:-auto}"
ARCH=(--pos rope --norm rmsnorm --ffn swiglu)

echo "== 1/4 char-level flagship (best validation loss ~1.46 around step 1,500) =="
$GPT prepare --tiny-shakespeare --out data/shakespeare-char --tokenizer char
$GPT train --data data/shakespeare-char --out runs/flagship --preset small --steps 2500 \
  --eval-interval 250 "${ARCH[@]}" --device "$DEVICE"
mkdir -p runs/flagship-best
cp runs/flagship/best.pt runs/flagship-best/latest.pt && cp runs/flagship/tokenizer.json runs/flagship-best/

echo "== 2/4 BPE base model (best validation loss 3.34-3.42 around step 400-500) =="
$GPT prepare --tiny-shakespeare --out data/shakespeare-bpe --tokenizer bpe --vocab-size 1024
$GPT train --data data/shakespeare-bpe --out runs/bpe-base --preset small --steps 600 \
  --eval-interval 100 "${ARCH[@]}" --device "$DEVICE"

echo "== 3/4 chat tasks and fine-tuning =="
$GPT make-tasks --text data/shakespeare-bpe/raw/tinyshakespeare.txt --out-dir data/tasks-who \
  --min-speeches 5 --kinds who --who-lines 4
$GPT finetune --run runs/bpe-base --checkpoint best --conversations data/tasks-who/train.jsonl \
  --out runs/chat-who --steps 800 --batch-size 16 --lr 3e-4 --device "$DEVICE"

echo "== 4/4 score it (held-out accuracy was 3.3% and 4.0% in two runs; baseline 24.9%) =="
$GPT eval-chat --run runs/chat-who --tasks data/tasks-who/heldout.jsonl --device "$DEVICE"
