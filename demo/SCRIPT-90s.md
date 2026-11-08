# GPT From Scratch - 90-second feed cut (LinkedIn)

For the muted autoplay feed: **captions carry the video, your voice is a bonus.** Every line under "CAPTION" is
short enough to read in the time shown. The "YOU SAY" bullets are prompts for your own words, not lines to read out.
The long version (5:00, for the repo or YouTube) is [SCRIPT.md](SCRIPT.md). Setup and recording steps are in
[HOW-TO-RECORD.md](HOW-TO-RECORD.md).

Total: about 90 s = 6 beats of 10-20 s. If a beat runs long, cut words, not beats.

| # | Time | ON SCREEN | CAPTION (on the video) | YOU SAY (bullets) |
|---|---|---|---|---|
| 1 | 0:00-0:12 | Two terminal panes side by side: `demo/assets/samples/step-000000-untrained.txt` (gibberish) and `step-001500-best.txt` (verse-shaped) | "Same network. Before and after 17 minutes of training on my laptop." | - Left is the untrained model, right is the same model after about 17 minutes - Say it before explaining anything |
| 2 | 0:12-0:30 | `make tour`, scroll to section 2 ("What the model is made of") | "10.7 million numbers. Two thirds are feed-forward layers, one third is attention." | - I built it from scratch in PyTorch, following Karpathy's video - Then I added what the video doesn't cover - The tour prints real values from the trained model, nothing is mocked |
| 3 | 0:30-0:45 | `./scripts/demo.sh bench` (the result lines) | "KV cache: identical tokens, 4x faster on CPU. My first number was wrong." | - Same output with and without the cache - My first GPU number included warm-up time; I fixed it and left the fix in the git history |
| 4 | 0:45-1:00 | `demo/assets/flagship-val-loss.svg` in the browser | "It overfits. I ship step 1,500, not the last step." | - Validation loss goes down, then back up - 10M parameters on 1 MB of text - That is why the best checkpoint is separate from the latest |
| 5 | 1:00-1:20 | Browser `127.0.0.1:8000`: type `ROMEO:`, Enter; switch the dropdown to the chat model, send `Speak as JULIET.` | "A chat model too. I scored it: 44% on questions it trained on, 3% on new scenes. Guessing gets 25%." | - It streams token by token - The honest result: it memorised. On new scenes it is worse than guessing - I'd rather show that than hide it |
| 6 | 1:20-1:30 | GitHub repo page, then `git log --oneline --grep='^fix'` | "Every mistake is a fix: commit. Repo in the first comment." | - One real question (see LINKEDIN.md) |

## Numbers you may say out loud (all measured, all in the repo)

| Claim | Number | Where it comes from |
|---|---|---|
| Parameters | 10,670,592 | `make tour`, section 2 |
| Feed-forward / attention share | 66.5% / 33.3% | `make tour`, section 2 |
| Best validation loss | 1.463 at step 1,500; rose to about 1.54 by step 2,500 | `demo/assets/flagship-log.csv`, reproduced on a fresh clone |
| Time to the best checkpoint | about 17 minutes | computed from the training log |
| KV-cache speed-up | 4.0x CPU, 2.3x GPU (240 tokens, one run each) | `./scripts/demo.sh bench` |
| Chat: questions it trained on | 44.3% (baseline 6.0%) | `experiments/chat-tasks.md` |
| Chat: unseen scenes | 3.3% to 4.0% (baseline 24.9%) | `experiments/chat-tasks.md` (two runs) |

## Do not say

- "It understands Shakespeare" or "it's a chatbot". It produces verse-shaped text with no meaning.
- "Built over weeks". The git history is 82 commits made on one day (2026-11-06). If you mention time, say what is true.
- Any speed-up number you have not just seen on your own screen. Your laptop will differ.
