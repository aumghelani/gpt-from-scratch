# 01 · GPT From Scratch: A Decoder-Only Transformer with BPE Tokenization, RoPE, KV-Cache and Chat Fine-Tuning

> Short name: **GPT From Scratch**. Original brief: *Apna GPT*.
> **Scope:** A Decoder-Only Transformer Language Model Trained From Scratch (Byte-Pair Tokenizer, RoPE, KV-Cache, Supervised Chat Fine-Tuning, Streaming Inference Server)

## Source

| | |
|---|---|
| Video | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| Author | Andrej Karpathy ("Neural Networks: Zero to Hero") |
| Length | ~1h56m |
| Companion code | [ng-video-lecture](https://github.com/karpathy/ng-video-lecture), [nanoGPT](https://github.com/karpathy/nanoGPT) |

> **Finding:** this is *not* a chatbot-app tutorial. It trains a small decoder-only Transformer in PyTorch
> on Tiny Shakespeare. That matches "Apna GPT" (my own GPT) well, so GPT From Scratch is a **real model you train
> yourself**, then wrapped in a chat interface. Transcript was not retrievable; the outline below comes
> from the creator's chapter list (verified), not spoken content.

## Research summary

### What the video builds (chapter map)
1. Data: tiny Shakespeare, char-level tokenizer, train/val split, batch loader (`block_size` chunks).
2. Bigram baseline: embedding table, cross-entropy loss, `generate()`.
3. The "math trick" for self-attention: for-loops → matmul with lower-triangular mask → softmax.
4. Self-attention proper: query / key / value, `1/sqrt(head_size)` scaling, positional embeddings.
5. Multi-head attention, feed-forward, residual connections, LayerNorm, dropout.
6. Scale up; decoder-only vs encoder vs cross-attention; nanoGPT walkthrough.
7. Concept only: pretraining vs finetuning vs RLHF (how ChatGPT differs from the toy).

Two errata from the author: at 0:57 "future" (not past) tokens must not communicate; at 1:20 normalize by `head_size`, not `C`.

### What the video does NOT cover (our "more than usual")
| Gap | Why it matters |
|---|---|
| Real tokenizer (BPE) | Char-level is a toy; vocab/compression drive quality. |
| Modern architecture bits | RoPE, RMSNorm, SwiGLU, fused attention, KV cache. |
| Training hygiene | LR warmup + cosine decay, grad clipping/accumulation, checkpoint-resume, eval loop. |
| Sampling | temperature, top-k, top-p, repetition penalty, stop tokens. |
| Chat behaviour | A base model only continues text; needs chat template + SFT with assistant-only loss masking. |
| Serving | Streaming API + UI so it actually feels like a GPT. |
| Evidence | Ablations proving each upgrade helps (val loss table). |

### Risks / unknowns
- **Python 3.14 is installed locally**; PyTorch wheel support may lag. Pin the project to **Python 3.12 via `uv`**.
- Apple Silicon: use `mps` backend; some ops fall back to CPU. Keep model small (≈10–30M params).
- Dataset licensing for any custom/Hindi/Hinglish corpus: check before committing data (never commit raw data; script downloads + checksums it).

## Goal

> Train a ~10–30M-parameter GPT **from scratch on a laptop**, with our own BPE tokenizer and modern
> architecture upgrades, give it basic chat behaviour via SFT, and serve it with streaming through a CLI,
> HTTP API and a small web UI — with an ablation report proving what each piece bought us.

### Success criteria (definition of done)
- [ ] Val loss on Tiny Shakespeare beats bigram baseline by a wide margin and matches the video's ≈1.5 region (char-level) — documented.
- [ ] Own BPE tokenizer: lossless round-trip on 100% of test corpus, property-tested.
- [ ] Ablation table: baseline → +multihead → +residual/LN → +RoPE → +RMSNorm/SwiGLU, with val loss for each.
- [ ] KV-cache generation is numerically identical (within tolerance) to no-cache generation; measurable speedup.
- [ ] Training resumes from checkpoint with identical loss curve continuation.
- [ ] SFT model follows a chat template and stops at `<|end|>`.
- [ ] `gpt-scratch chat` CLI + `POST /v1/chat` SSE streaming + web UI working end to end.
- [ ] ≥80% test coverage on tokenizer, model, sampling, data code.

### Non-goals
- Competing with real LLMs on quality. RLHF/DPO (read-only concept, maybe a stretch).
- Multi-GPU / distributed training.

## Proposed architecture

```
01-gpt-from-scratch/
  pyproject.toml            # uv, python 3.12
  src/gpt_scratch/
    data/        # download, clean, shard, loader
    tokenizer/   # char.py, bpe.py, special tokens, save/load
    model/       # config, attention, blocks, gpt, rope, kvcache
    train/       # loop, optim, schedule, checkpoint, eval, logging
    sample/      # sampling strategies, generate, stream
    chat/        # template, sft dataset, masking, sft trainer
    serve/       # cli, fastapi app, web ui
  tests/
  experiments/   # ablation configs + results
  notebooks/     # attention visualisations
```

Design rules: immutable config dataclasses (`frozen=True`), small files (<400 lines), no `print` (use `logging`), seeds explicit.

---

## Parallel task board

Legend: **W1** = no dependencies, start all at once · **W2** = needs listed W1 tasks · **W3** = integration.
Format: `ID — task (deps)`. Every task = one microcommit (tests in same or preceding commit).

### Track A · Data
- [ ] **A1 · W1** — Script to download Tiny Shakespeare + SHA-256 verify (none)
- [ ] **A2 · W1** — Text cleaner: unicode normalise, strip control chars, dedupe lines (none)
- [ ] **A3 · W2** — Train/val split + `uint16` memmap shards writer (A1, B1)
- [ ] **A4 · W2** — Random-batch loader returning `(x, y)` shifted by one, with tests for shape/shift/no-overlap-leak (A3)
- [ ] **A5 · W1** — Optional Hinglish/Hindi corpus fetcher + licence note (none; never commit raw data)

### Track B · Tokenizer
- [ ] **B1 · W1** — Char tokenizer: `encode/decode`, vocab save/load (none)
- [ ] **B2 · W1** — Byte-level BPE trainer (merge counting, deterministic tie-break) (none)
- [ ] **B3 · W2** — BPE encode/decode with regex pre-split pattern (B2)
- [ ] **B4 · W2** — Special tokens (`<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) (B3)
- [ ] **B5 · W2** — Property tests: `decode(encode(s)) == s` for random unicode (B3)
- [ ] **B6 · W3** — Compare compression ratio vs `tiktoken` as sanity (B3)

### Track C · Model
- [ ] **C1 · W1** — `GPTConfig` frozen dataclass + validation (none)
- [ ] **C2 · W1** — Bigram baseline model + loss (C1)
- [ ] **C3 · W1** — Single causal self-attention head (C1)
- [ ] **C4 · W2** — Multi-head attention (batched, single matmul) (C3)
- [ ] **C5 · W1** — Feed-forward block (none)
- [ ] **C6 · W2** — Transformer block: pre-LN + residuals + dropout (C4, C5)
- [ ] **C7 · W2** — GPT assembly: token + position embeddings, weight tying, init (C6)
- [ ] **C8 · W1** — Causality test: perturbing token *t* never changes logits at positions < *t* (C3)
- [ ] **C9 · W2** — RoPE positional encoding replacing learned positions (C7)
- [ ] **C10 · W2** — RMSNorm swap (C7)
- [ ] **C11 · W2** — SwiGLU FFN swap (C7)
- [ ] **C12 · W2** — Use `F.scaled_dot_product_attention`; numeric parity test vs manual (C4)
- [ ] **C13 · W3** — KV cache: incremental decode + parity test vs full forward (C9)
- [ ] **C14 · W1** — Param counter + FLOPs estimate util (C1)

### Track D · Training
- [ ] **D1 · W1** — Seeding + device selection (cpu/mps/cuda) helper (none)
- [ ] **D2 · W2** — Minimal training loop (A4, C7)
- [ ] **D3 · W1** — AdamW param groups (no weight decay on norms/biases) (none)
- [ ] **D4 · W1** — LR schedule: linear warmup + cosine decay, unit-tested at boundaries (none)
- [ ] **D5 · W2** — Grad clipping + gradient accumulation (D2)
- [ ] **D6 · W2** — Checkpoint save/resume incl. optimizer + RNG + step (D2)
- [ ] **D7 · W2** — Eval loop (val loss every N steps, no-grad, averaged) (D2)
- [ ] **D8 · W2** — CSV logger + loss-curve plot script (D2)
- [ ] **D9 · W2** — Overfit-one-batch sanity test (loss → ~0) (D2)
- [ ] **D10 · W3** — Resume-equivalence test: 100 steps vs 50+resume+50 match (D6)

### Track E · Sampling
- [ ] **E1 · W1** — Greedy + temperature (none)
- [ ] **E2 · W1** — top-k and top-p filters, tested on hand-made logits (none)
- [ ] **E3 · W1** — Repetition penalty (none)
- [ ] **E4 · W2** — `generate()` with stop tokens + max length (C7, E1)
- [ ] **E5 · W2** — Streaming generator yielding decoded text deltas (E4, B3)

### Track F · Chat / SFT
- [ ] **F1 · W1** — Chat template spec + renderer (none)
- [ ] **F2 · W2** — SFT dataset builder from JSONL conversations (F1, B4)
- [ ] **F3 · W2** — Loss masking so only assistant tokens contribute; tested (F2)
- [ ] **F4 · W3** — SFT trainer reusing train loop (F3, D2)
- [ ] **F5 · W1** — Curate tiny instruction set (hand-written + licensed) (none)

### Track G · Serving
- [ ] **G1 · W2** — `gpt-scratch chat` CLI with streaming (E5)
- [ ] **G2 · W2** — FastAPI `POST /v1/chat` SSE endpoint + schema validation (E5)
- [ ] **G3 · W3** — Static web UI (stop/regenerate, markdown) (G2)
- [ ] **G4 · W3** — Checkpoint selector (switch models without restart) (G2)

### Track H · Quality & write-up
- [ ] **H1 · W1** — `pyproject.toml` (uv, py3.12), ruff, pytest, coverage config (none)
- [ ] **H2 · W1** — Makefile/justfile: `test`, `lint`, `train-*` targets (none)
- [ ] **H3 · W3** — Ablation runner + results table (D7, C9, C10, C11)
- [ ] **H4 · W3** — Attention heatmap notebook (C4)
- [ ] **H5 · W3** — README: how to train, sample, serve; ablation table (all)

**First wave to launch together:** A1 A2 A5 B1 B2 C1 C3 C5 C14 D1 D3 D4 E1 E2 E3 F1 F5 H1 H2.

---

## Track Z · Demo & LinkedIn (do last, but capture assets as you go)

**Story in one line:** "I trained my own GPT on a laptop — here is it going from gibberish to Shakespeare to a chatbot."

**Shot list (target 3–4 min):**
1. Hook: step-0 sample (pure noise) next to the final sample. *(Capture raw samples at step 0, mid, end during training.)*
2. The loss curve drawing live (terminal or plot) — point at where it "clicks".
3. 20-second attention-heatmap view: "this is what the model looks at".
4. Web chat UI streaming a reply; hit stop/regenerate.
5. Ablation table: each upgrade (multi-head → residual/LN → RoPE → SwiGLU) and the val-loss drop.
6. Close: what surprised me + repo link.

**Talking points for you:** why causal masking matters · what attention actually computes (Q/K/V in plain words) · one thing that broke and how you found it · honest limits (tiny model, Shakespeare-flavoured).

- [ ] **Z1** — Save step-0 / mid / final sample outputs automatically during training (D8)
- [ ] **Z2** — One-command demo script `make demo` that replays a pre-trained checkpoint (G2)
- [ ] **Z3** — Export loss-curve + ablation chart as PNG (H3)
- [ ] **Z4** — Write `demo/SCRIPT.md` (shot list + talking points above, timed)
- [ ] **Z5** — Screen recording (QuickTime / `Cmd+Shift+5`) following the script
- [ ] **Z6** — LinkedIn post draft + 1 hero image (before/after samples) in `demo/LINKEDIN.md`
