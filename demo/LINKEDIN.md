# LinkedIn draft — GPT From Scratch  (DRAFT: edit into your own voice before posting)

**Audience:** engineers and ML-curious builders scrolling LinkedIn. **Not posted anywhere.** Approve/edit first.

---

## Post (≈1,100 chars)

I trained a GPT from scratch on my laptop. Here's the same network before and after about 17 minutes of training on my laptop:

Before: `ROMEO:NCaa;suqWP.&&TNUUuIgl,ONNNfvjjXLZ&Kf…`
After: `ROMEO: Call thy repeal up leave at the hardy shore. O, a virtue can I came in with thy hand…`

It learned spelling, line breaks and the rhythm of verse. It did not learn meaning — which is the honest size of a 1 MB corpus.

What I built (10.7M parameters, PyTorch, following Karpathy's "Let's build GPT" and then going further):
→ my own BPE tokenizer
→ RoPE, RMSNorm and SwiGLU, each ablated
→ a KV cache that gives identical tokens, 4x faster on CPU
→ resume-from-checkpoint that reproduces an uninterrupted run
→ a streaming server and web UI

Four things I'd tell past me:
1. It overfits. Validation loss bottomed out at step 1,500 (1.464) and rose to 1.540 while training loss kept falling.
2. RoPE bought most of the gain (−0.073 val loss). RMSNorm was a wash. One seed, so I'm not claiming more.
3. I fine-tuned it to answer "who says this line?" and scored it. 44% on questions it trained on. 3% on unseen scenes, while always guessing the most common speaker gets 25%. Worse than guessing = memorisation.
4. Tests earned their keep, and still missed things: a property test caught my tokenizer dropping underscores, but "resume after a crash" had never been tried with a real crash.

Repo + ablation tables in the first comment.

What's the first thing you'd change to make a model this small actually useful?

#MachineLearning #LLM #PyTorch #BuildInPublic

---

## First comment

Code, tests and the full ablation: https://github.com/aumghelani/gpt-from-scratch
*(repo is private right now — make it public before posting this link)*

## Notes before you post

- Numbers above are copied from `README.md` and `experiments/`; re-check after any re-run. The post is long for LinkedIn:
  cut point 2 or 4 if you want it tighter.
- "About 17 minutes": computed from the logged tokens/sec: 16.7 min of training steps to reach step 1,500; all 2,500 steps
  are 28.6 min of steps (the README's ~46 min is wall-clock including evals and sample snapshots).
- Hook options to A/B: (a) the before/after text, (b) "It overfit after 1,500 steps — here's the chart".
- Hero image: side-by-side of the two samples, or `demo/assets/flagship-val-loss.svg` exported as PNG.
