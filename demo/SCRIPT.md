# GPT From Scratch - demo script (target 5:00)

Narration is **bullets for you to say in your own words**, not a script to read. Every command below was run to
produce the repo's results. If you can answer the 15 questions at the end of
[docs/HOW-IT-WORKS.md](../docs/HOW-IT-WORKS.md), you can explain everything here. Record in short takes, one per beat.

Setup: `cd gpt-from-scratch`, venv active, terminal font 18pt or larger, notifications off, no `.env` or email visible.
Checkpoints must exist (`scripts/reproduce.sh` rebuilds them). Pre-run in a second tab:
`gpt-scratch serve --run runs/flagship-best runs/chat-who --device cpu`.

| # | Time | ON SCREEN | YOU SAY (bullets) |
|---|---|---|---|
| 0 | 0:00 | `demo/assets/samples/step-000000-untrained.txt` beside `step-001500-best.txt` | - "Same network. Left: before training. Right: after about 17 minutes of training on my laptop." - Say it before explaining anything |
| 1 | 0:20 | README top | - Built it following Karpathy's "Let's build GPT", then added what the video doesn't cover - 10.7M parameters, from scratch, tiny Shakespeare |
| 2 | 0:40 | `scripts/demo.sh tour` - scroll slowly through sections 1-4 | - Text becomes integers (show the token list) - Most of the model is feed-forward layers, a third is attention - An untrained model's loss is ln(vocabulary) = 4.17; mine reached 1.46 - It outputs a probability for every next token. That is the whole idea |
| 3 | 1:30 | Same output, sections 6-7, then open `src/gpt_scratch/model/attention.py` at the `forward` | - Query, key, value in one sentence each - The mask is why it only looks backwards (the first token attends 100% to itself) - KV cache: after step 1 only ONE new token is fed in |
| 4 | 2:10 | `scripts/demo.sh bench` | - Same tokens with and without the cache - 4x faster on CPU for 240 tokens, 2.3x on the GPU - My first number was wrong: it included GPU warm-up. I fixed it and kept the fix in the history |
| 5 | 2:40 | `demo/assets/flagship-val-loss.svg` | - Validation loss goes down, then up: overfitting. 10M parameters, 1 MB of text - I ship the step-1,500 checkpoint, not the last - The same happened to the BPE model and to chat fine-tuning |
| 6 | 3:10 | `demo/assets/ablation-delta.svg` | - RoPE helped most; RMSNorm is a wash (cheaper, not better); one seed, so gaps under 0.01 are noise |
| 7 | 3:30 | Browser `127.0.0.1:8000`: pick the model, type `ROMEO:`, press **Enter**; switch to the chat model, `Speak as JULIET.` | - Streaming over server-sent events - The dropdown switches models - Chat model stops by itself at the end token. Verse-shaped, but not meaningful |
| 8 | 4:10 | `experiments/chat-tasks.md` results table | - I scored it: "who says this?" 44% on questions it trained on, **3% on unseen scenes** - Guessing the commonest speaker gets 25%, so on new scenes it is *worse than guessing* - That is memorisation. The fix is more data, not more steps |
| 9 | 4:40 | README "Engineering notes" | - A crash lost the whole run: resume existed but nothing was saved until the end - found by a test that actually crashes - The UI didn't send on Enter; found by using it in a browser |
| 10 | 4:55 | Repo link | - One real question to the audience (see LINKEDIN.md) |

## Gotchas to avoid on camera

- **Don't claim the model understands Shakespeare or that it is a chatbot.** It reads like verse; the content is nonsense.
- Don't run `train` live. The flagship's training steps take about 29 minutes (17 to reach the best checkpoint).
- If the server tab shows a port error: `lsof -i :8000`, kill, restart.
- Serve on **127.0.0.1** only; do not bind `0.0.0.0` on a shared network.
- The `tour` output scrolls; run it once beforehand and know where sections 4 and 7 are.

## Recording checklist

- [ ] Dry run, timed (cut any beat over 45 s)
- [ ] Captions on (LinkedIn autoplays muted)
- [ ] Record each beat as its own take
- [ ] Feed cut 60-120 s (beats 0, 2-short, 5, 8); this 5:00 version for the repo or YouTube
