# What the KV cache buys

Greedy decoding of N new tokens from the prompt `ROMEO:` with the flagship model (10.7M parameters, context 256),
once recomputing everything each step and once with the cache. Same tokens in both cases (checked on every run).
Apple M-series laptop, one run each after an untimed warm-up.

```bash
gpt-scratch bench-cache --run runs/flagship-best --prompt "ROMEO:" --new-tokens 240 --device cpu   # or mps
```

| Device | New tokens | With cache | Without | Speedup |
|---|---:|---:|---:|---:|
| CPU | 50 | 0.066 s | 0.123 s | 1.9x |
| CPU | 240 | 0.337 s | 1.333 s | 4.0x |
| MPS (GPU) | 50 | 0.132 s | 0.171 s | 1.3x |
| MPS (GPU) | 240 | 0.497 s | 1.140 s | 2.3x |

Reading it:

- Without the cache every step re-processes the whole sequence, so the work per token grows with length; with the
  cache each step processes one token. The speedup therefore **grows with length** (1.9x -> 4.0x on CPU).
- On the GPU the gain is smaller. At this size a step is dominated by fixed per-call overhead, not arithmetic,
  so there is less to save. A larger model or longer context would favour the cache more.
- Single runs on a shared laptop: expect a few tens of percent of run-to-run variation. The first version of this
  benchmark had no warm-up and reported 3.8x on MPS for 200 tokens; that number was inflated by one-off GPU setup.
