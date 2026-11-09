# GPT From Scratch - explainer script for a beginner audience (about 4.5 minutes)

Use this when you want to **teach** the project, not just show it. The spoken lines are plain and short so you can
say them in your own words. Change any sentence that does not sound like you. Pair it with
[HOW-TO-RECORD.md](HOW-TO-RECORD.md) for setup and recording. Shorter feed cut: [SCRIPT-90s.md](SCRIPT-90s.md).

**Before you press record:** terminal in the repo folder, font 18 pt or larger, server running in a second tab
(`.venv/bin/gpt-scratch serve --run runs/flagship-best runs/chat-who --device cpu`), notifications off.
Run `make tour` once beforehand so you know where each section is.

Each beat: **DO** is what you do on screen, **SAY** is what you say.

---

### Beat 1 - Hook (0:00 to 0:20)

**DO:** Show two files side by side: `demo/assets/samples/step-000000-untrained.txt` and
`demo/assets/samples/step-001500-best.txt`.

**SAY:** "This is the same neural network twice. On the left, before training: random characters. On the right, after
about 17 minutes of training on my laptop: text that looks like a play. I'll show you how it works, step by step."

### Beat 2 - What it is (0:20 to 0:50)

**DO:** Open the README in the browser (the GitHub page) and stay on the top.

**SAY:** "It's a small GPT, with about ten and a half million adjustable numbers. It plays one game: look at some text
and guess the next character. I trained it on Shakespeare. I built it from scratch in PyTorch, following Andrej
Karpathy's lecture, and then added my own tokenizer, a speed-up called a KV cache, a chat mode and a web server."

*Optional, and your call:* "I used an AI coding assistant to move fast, and I reviewed all of it. I can explain every
part, which is what the rest of this video is."

### Beat 3 - Text becomes numbers (0:50 to 1:15)

**DO:** In the terminal run `make tour`. Stop at **section 1**. Point at the two `O` lines with id 27 and the space with id 1.

**SAY:** "A computer can't read letters, so the first step is to turn text into numbers. 'ROMEO: But soft' is fifteen
characters, so it becomes fifteen numbers. See the letter O? It's twenty-seven both times. Same character, same number.
This model knows sixty-five different characters, and that's its whole vocabulary."

### Beat 4 - Attention and feed-forward (1:15 to 2:00)

**DO:** Scroll to **section 2**. Pause on the table with the two percentages.

**SAY:** "Inside there are six identical blocks, and each block does two jobs. First, attention: every token looks back
at the earlier tokens and gathers what it needs. That's the only place the tokens talk to each other. Second,
feed-forward: each token goes off alone and processes what it just heard. Think of a meeting, then desk work, six times.
One third of the model's numbers are attention. Two thirds are the feed-forward part, so it's the bigger calculator."

### Beat 5 - Knowing nothing, and the answer (2:00 to 2:40)

**DO:** Scroll to **section 3**, then **section 4**. Then run
`.venv/bin/gpt-scratch tour --run runs/flagship-best --prompt "JULIET: O" --layer 3` and scroll to section 4.

**SAY:** "How do we know it learned anything? A model that knows nothing gives all sixty-five characters the same chance.
Its error, the loss, is 4.17. Mine reached 1.46. And the model's whole answer is just a probability for every next
character. After 'JULIET: O', it puts seventy-five percent on a comma. In Shakespeare, 'O,' is a very common pattern,
and it learned that."

### Beat 6 - Temperature and generating (2:40 to 3:05)

**DO:** Scroll to **section 5** briefly, then **section 8**.

**SAY:** "Temperature controls how bold the pick is. Low is safe and repetitive. High is creative and messy. Greedy always
takes the top choice. Sampling draws from the probabilities, so you get different text each time."

### Beat 7 - The speed-up (3:05 to 3:30)

**DO:** Run `./scripts/demo.sh bench`. Read the result lines on screen.

**SAY:** "Generating text one token at a time repeats a lot of work. The KV cache remembers the earlier work. Same
tokens out, but about two times faster on my GPU and about four times on the CPU. The first number I measured was
wrong, because it counted GPU warm-up time. I fixed it, and the fix is in the git history."

### Beat 8 - Overfitting (3:30 to 3:55)

**DO:** Open `demo/assets/flagship-val-loss.svg` in the browser.

**SAY:** "Here's the honest part. The error on text it hasn't seen goes down, then back up. That's overfitting: with ten
million numbers and one megabyte of text, it starts memorizing. So I ship the checkpoint from step 1,500, not the last one."

### Beat 9 - The web UI and the limit (3:55 to 4:25)

**DO:** Open `http://127.0.0.1:8000`. Choose the flagship model, type `ROMEO:`, press **Enter**. Then choose the chat
model and send `Speak as JULIET.`

**SAY:** "I wrapped it in a small server so you can watch it generate live. I also fine-tuned a chat version. It looks
like verse, but it doesn't mean anything. I scored it: on questions it trained on, 44 percent. On new scenes, 3 to 4
percent, which is worse than just guessing the most common answer. So it memorized. It's not a chatbot, and I'd rather
show you that than hide it."

### Beat 10 - Close (4:25 to 4:35)

**DO:** Show the GitHub repo page.

**SAY:** "The code, the tests and a guide that explains every part are in the repo, linked below. What would you want
to see me build next?"

---

## If you forget your words

Each beat has one idea. Say the idea in your own words and move on. Fixing a stumble in editing is easier than
starting over, so pause for a second and then repeat the last sentence.

## Do not say

- That it understands Shakespeare, or that it is a chatbot.
- A build time longer than the truth. The repo went from first commit to public in about three days (6 to 8 November 2026).
- A speed-up number you haven't just seen on your own screen. Your machine gives slightly different numbers each run.
