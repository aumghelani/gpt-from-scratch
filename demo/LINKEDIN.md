# LinkedIn draft — GPT From Scratch  (DRAFT: edit into your own voice before posting)

**Audience:** engineers and ML-curious builders scrolling LinkedIn. **Not posted anywhere.** Approve/edit first.

---

## Post (≈1,100 chars)

I trained a GPT from scratch on my laptop. Here's the same network before and after about 20 minutes of training on my laptop:

Before: `ROMEO:NCaa;suqWP.&&TNUUuIgl,ONNNfvjjXLZ&Kf…`
After: `ROMEO: Call thy repeal up leave at the hardy shore. O, a virtue can I came in with thy hand…`

It learned spelling, line breaks and the rhythm of verse. It did not learn meaning — which is the honest size of a 1 MB corpus.

What I built (10.7M parameters, PyTorch, following Karpathy's "Let's build GPT" and then going further):
→ my own BPE tokenizer
→ RoPE, RMSNorm and SwiGLU, each ablated
→ a KV cache, tested to match full recomputation
→ resume-from-checkpoint that reproduces an uninterrupted run
→ a streaming server and web UI

Three things I'd tell past me:
1. It overfits. Validation loss bottomed out at step 1,500 (1.464) and rose to 1.540 while training loss kept falling.
2. RoPE bought most of the gain (−0.073 val loss). RMSNorm was a wash. One seed, so I'm not claiming more.
3. Tests earned their keep: a property test caught my tokenizer silently dropping underscores.

Repo + ablation tables in the first comment.

What's the first thing you'd change to make a model this small actually useful?

#MachineLearning #LLM #PyTorch #BuildInPublic

---

## First comment

Code, tests and the full ablation: https://github.com/aumghelani/ai-engineering-projects (folder `01-gpt-from-scratch`)
*(repo is private right now — make it public before posting this link)*

## Notes before you post

- Numbers above are copied from `README.md`; re-check after any re-run.
- "About 20 minutes": 1,500 steps × ~0.75 s/step at ~11k tokens/s ≈ 19 min of compute to the best checkpoint. The full
  2,500-step run took ~46 min wall-clock (other work shared the machine), but the best weights come from step 1,500.
- Hook options to A/B: (a) the before/after text, (b) "It overfit after 1,500 steps — here's the chart".
- Hero image: side-by-side of the two samples, or `demo/assets/flagship-val-loss.svg` exported as PNG.
