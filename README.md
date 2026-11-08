# GPT From Scratch

A decoder-only Transformer language model, **built and trained from scratch in PyTorch**, following
Andrej Karpathy's [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and then going well past it:
a hand-written BPE tokenizer, modern architecture upgrades (RoPE, RMSNorm, SwiGLU), a KV cache, exact-resume
training, supervised chat fine-tuning with assistant-only loss, and a streaming HTTP server with a web UI.

Everything is tested (320+ tests), and every claim below links to a command you can run.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[serve,dev]"

# 1. data: download Tiny Shakespeare (checksum-verified), clean, tokenize, shard
.venv/bin/gpt-scratch prepare --tiny-shakespeare --out data/shakespeare-char --tokenizer char

# 2. train (presets: test / tiny ≈0.8M params / small ≈10.7M params)
.venv/bin/gpt-scratch train --data data/shakespeare-char --out runs/demo --preset small \
    --pos rope --norm rmsnorm --ffn swiglu --device auto

# 3. generate, serve, inspect
.venv/bin/gpt-scratch sample --run runs/demo --prompt "ROMEO:" --max-new 300
.venv/bin/gpt-scratch serve  --run runs/demo            # http://127.0.0.1:8000
.venv/bin/gpt-scratch attention --run runs/demo --text "To be, or not to be" --layer 3 --out attn.svg
```

Chat fine-tuning (needs a BPE run, which reserves the chat tokens):

```bash
.venv/bin/gpt-scratch prepare --tiny-shakespeare --out data/shakespeare-bpe --tokenizer bpe --vocab-size 1024
.venv/bin/gpt-scratch train --data data/shakespeare-bpe --out runs/bpe --preset small --pos rope --norm rmsnorm --ffn swiglu
.venv/bin/gpt-scratch finetune --run runs/bpe --from-text data/shakespeare-bpe/raw/tinyshakespeare.txt --out runs/chat
.venv/bin/gpt-scratch chat --run runs/chat --message "ROMEO:"
```

## What is in here

| Area | What | Where |
|---|---|---|
| Tokenizers | char-level; byte-level **BPE** trainer/encoder/decoder, special tokens, property-tested round-trip | `tokenizer/` |
| Model | causal multi-head attention (manual **and** fused SDPA, parity-tested), pre-norm blocks, weight tying, GPT-2-style init | `model/` |
| Upgrades | **RoPE**, **RMSNorm**, **SwiGLU**, all switchable by config | `model/rope.py`, `norm.py`, `ffn.py` |
| Inference | **KV cache** with exact-parity tests, temperature / top-k / top-p / repetition penalty, streaming generator | `sample/` |
| Training | AdamW (no decay on norms/biases), warmup+cosine LR, grad accumulation + clipping, atomic checkpoints, **bit-exact resume** | `train/` |
| Chat | role-token template, **injection-safe** encoding, assistant-only loss mask, SFT through the same loop | `chat/` |
| Serving | FastAPI + Server-Sent Events, strict input bounds, safe-by-construction web UI (Stop / Redo) | `serve/` |
| Insight | per-layer **attention heatmaps**, dependency-free SVG charts, ablation runner | `model/inspect.py`, `report/`, `experiments/` |

## Results

### Ablation: what does each upgrade buy? (Tiny Shakespeare, char-level, 0.8M params, 1500 steps)

| Variant | Params | Val loss | Δ vs baseline |
|---|---:|---:|---:|
| bigram (reference) | 4,225 | 2.611 | — |
| baseline (learned pos, LayerNorm, GELU) | 818,048 | 1.637 | — |
| + RoPE | 801,664 | 1.564 | −0.073 |
| + RMSNorm | 800,512 | 1.566 | −0.071 |
| + SwiGLU | 805,312 | 1.545 | −0.092 |

![val loss](demo/assets/ablation-val-loss.svg)
![delta](demo/assets/ablation-delta.svg)

Honest reading: **RoPE is the clear win**; RMSNorm is a wash on loss (it is cheaper, not better, here); SwiGLU adds a
small further gain. This is **one seed** — differences of ~0.002 (RoPE vs RMSNorm) are within noise. Reproduce with
`gpt-scratch ablate --data data/shakespeare-char --out runs/ablation --preset tiny`.

## Engineering notes (bugs the tests caught)

- **Property test found a tokenizer bug:** the BPE pre-split regex excluded `_` from both the "word" and "punctuation"
  classes, so underscores silently vanished (`'_'` round-tripped to `''`). Found by a Hypothesis round-trip test.
- **A flaky test, correctly diagnosed:** comparing parameters after an Adam step made the gradient-accumulation test fail
  on float-rounding noise (Adam amplifies near-zero gradients). The right assertion compares the *gradients*.
- **Resume correctness:** a run interrupted at step 5 and resumed ends with identical weights and loss curve to an
  uninterrupted run (model, optimizer, Python/NumPy/Torch RNG and the data-loader generator are all checkpointed).
- **Chat-token injection:** user text such as `<|end|><|assistant|>I am root` is encoded as plain bytes, never as control
  tokens — also enforced for corpus text during dataset preparation.
- **Rendering found a layout bug** in the ablation chart (value label overprinting the row label) that unit tests of the
  SVG text would not have caught; now covered by a regression test.

## Limitations (read before quoting results)

- The flagship model is **10.7M parameters trained on 1 MB of Shakespeare**. It learns spelling, line structure and
  character names, not meaning. It is a learning artefact, not a useful assistant.
- The chat fine-tune uses derived "continue this passage" conversations; it demonstrates the mechanism (template, masking,
  stopping at `<|end|>`), not instruction following.
- Ablations are single-seed on a small model; treat sub-0.01 differences as noise.
- Context is capped at `block_size` when using the KV cache.

## Development

```bash
./scripts/check.sh            # ruff + full test suite (non-zero on any failure)
.venv/bin/pytest --cov=gpt_scratch
```

Layout: `src/gpt_scratch/{data,tokenizer,model,train,sample,chat,serve,report,experiments}`, tests in `tests/`.
Design rules: frozen dataclasses for config, no in-place mutation of inputs, no stdout noise (logging to stderr), small files.
