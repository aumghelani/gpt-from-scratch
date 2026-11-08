# Chat fine-tuning: what a 10M-parameter Shakespeare model can and cannot be taught

The base model is the BPE run (vocab 1,024, 11.0M parameters, RoPE/RMSNorm/SwiGLU) at its **best checkpoint, step 400**
(validation loss 3.415; later steps overfit). Two instruction tasks are derived from the speaker labels in Tiny
Shakespeare, so every answer is checkable:

- **who** - `Who says this?` + the speech's opening lines -> the speaker's name (168 speakers with >= 5 training speeches)
- **speak** - `Speak as ROMEO.` -> one of that character's speeches

Hold-out = the last 10% of the text (the part of the corpus the base model also never trained on): 273 questions,
11 speakers (scenes from *The Taming of the Shrew*). Always answering the most common speaker scores **24.9%**.

```bash
gpt-scratch make-tasks --text data/shakespeare-bpe/raw/tinyshakespeare.txt --out-dir data/tasks-who \
    --min-speeches 5 --kinds who --who-lines 4
gpt-scratch finetune --run runs/bpe-base --checkpoint best --conversations data/tasks-who/train.jsonl \
    --out runs/chat-who --steps 800 --batch-size 16 --lr 3e-4 --device mps
gpt-scratch eval-chat --run runs/chat-who --tasks data/tasks-who/heldout.jsonl
```

## Results (who-says-this, exact match, greedy)

| Run | Questions | Accuracy | Majority baseline |
|---|---|---:|---:|
| mixed who+speak SFT, **training** questions | 200 | 5.0% | 5.0% |
| who-only SFT, **training** questions | 300 | **44.3%** | 6.0% |
| who-only SFT, **held-out** scenes | 273 | **3.3%** (4.0% on a fresh-clone rerun) | 24.9% |

## What happened, in order

1. **First attempt learned nothing.** Training on both tasks gave 5.0% on its own training questions, exactly the
   majority baseline. The long "speak as" targets dominated the loss, validation loss bottomed at step 100 on that
   signal, and the two-to-four-token name answers barely registered. A sanity check on *training* questions is what
   showed the evaluation was fine and the training was the problem.
2. **Training the task on its own signal worked on seen material:** 44.3% against a 6.0% baseline.
3. **It does not generalise.** On unseen scenes it scores 3.3%, far *below* the 24.9% baseline. The held-out
   characters appear only a handful of times in training, so the model falls back on frequent names
   (`KING EDWARD IV`, `BUCKINGHAM`, ...). This is memorisation rather than understanding, which is what 1 MB of text
   and an already-overfitting base model predict.
4. Famous lines do not rescue it. Asked who says "But, soft! what light through yonder window breaks?" it answers
   `KING RICHARD II` (the answer is ROMEO).

## "Speak as X" (qualitative; no automatic score)

From `runs/chat` (mixed SFT), temperature 0.7, top-k 40, seed 3:

```
Speak as ROMEO.     O, salut, lest with little infety,
                    More patient as I get a curst in tears
                    With triumphings, and that file-blood ...
Speak as MENENIUS.  O, sir, she says him, little more than I loved
                    Than she is the dish, I'll find you to the
                    sight of my teeth.
```

The mechanism works - the persona prompt is accepted, output is verse-shaped, and generation stops by itself at
`<|end|>` - but the words are not meaningful. Nothing here is instruction following in the useful sense.

## Takeaways

- The chat *plumbing* (template, special tokens, assistant-only loss, stopping) is verified by tests and by this run.
- The *capability* is limited by data: a model that overfits 1 MB of text cannot learn a task that needs knowledge of
  the plays. A bigger corpus, not more training steps, is the fix.
- The evaluation is the useful part: it separates "learned the task" (44% vs 6%) from "learned it in a way that
  transfers" (3% vs 25%), and the training-set check shows when a low score is a training problem.
