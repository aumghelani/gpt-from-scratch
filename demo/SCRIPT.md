# GPT From Scratch — demo script (target 3:30)

Narration is **bullets for you to say in your own words**, not a script to read. Every command below was run to
produce the repo's results; numbers are from `README.md`. Record in short takes, one per beat.

Setup before recording: `cd 01-gpt-from-scratch`, venv active, terminal font ≥ 18pt, notifications off, no `.env`
or email visible. Pre-run `gpt-scratch serve --run runs/flagship-best --device cpu` in a second tab (port 8000).

| # | Time | ON SCREEN | YOU SAY (bullets) |
|---|---|---|---|
| 0 | 0:00 | `demo/assets/samples/step-000000-untrained.txt` then `step-001500-best.txt` side by side | • "Same network. Left: before training. Right: after about 20 minutes of training on my laptop." • Say it before explaining anything |
| 1 | 0:20 | Title card / README top | • Built it following Karpathy's "Let's build GPT", then added things the video doesn't cover • 10.7M parameters, trained from scratch, tiny Shakespeare |
| 2 | 0:45 | `src/gpt_scratch/model/attention.py` (the `forward`) | • Attention = every token asks "who is relevant to me?" • Q/K/V in one sentence each • The mask is why it can only look backwards |
| 3 | 1:15 | `gpt-scratch attention --run runs/flagship-best --text "ROMEO: But soft, what light" --layer 3 --out /tmp/a.svg` → open `demo/assets/attention-layer3-mean.svg` | • Rows ask, columns answer • Lower-triangular = causal • Mostly looks at the last few characters and word starts |
| 4 | 1:45 | `demo/assets/flagship-val-loss.svg` | • Validation loss goes down… then up • That's overfitting — 10M parameters, 1 MB of text • I ship the step-1500 checkpoint, not the last one |
| 5 | 2:15 | `demo/assets/ablation-delta.svg` | • I ablated each upgrade: RoPE, RMSNorm, SwiGLU • RoPE helps most; RMSNorm is a wash on quality (cheaper, not better) • One seed — I'd call gaps under 0.01 noise |
| 6 | 2:45 | Browser `http://127.0.0.1:8000`, type `ROMEO:` → **Send**, then **Stop**, then **Redo** | • Streaming over SSE • It's a text-completer: a char-level Shakespeare model, not an assistant • (Stop and Redo both work; Redo had a history bug I found by reading my own UI code) |
| 7 | 3:05 | `README.md` → "Engineering notes (bugs the tests caught)" | • Property test found my tokenizer silently dropped underscores • Rendering a chart found a label overlapping the row name • A "passing" test was wrong, not the code (Adam vs gradients) |
| 8 | 3:20 | Repo link | • What I'd do next: more data, a real tokenizer vocab at scale • One real question to the audience (see LINKEDIN.md) |

## Gotchas to avoid on camera

- **Don't claim the model "understands" Shakespeare.** The sample reads like verse; the content is nonsense. Show the gap.
- Don't run `train` live (it takes ~45 min); show the log chart.
- If the server tab shows a port error: `lsof -i :8000`, kill, restart.
- The browser UI needs `gpt-scratch serve` on **127.0.0.1** only; do not bind `0.0.0.0` on a shared network.

## Recording checklist

- [ ] Dry run, timed (cut any beat > 45 s)
- [ ] Captions on (LinkedIn autoplays muted)
- [ ] Record each beat as its own take
- [ ] Final cut 60–120 s for the feed; this 3:30 version for the repo / YouTube
