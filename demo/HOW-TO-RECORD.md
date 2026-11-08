# How to show this project on video

A practical walkthrough: what to prepare, what to put on screen, how to record it, and how to add captions.
Scripts: [SCRIPT-90s.md](SCRIPT-90s.md) (feed cut) and [SCRIPT.md](SCRIPT.md) (5-minute version).

## 0. What you are showing

Five things, in order of how well they film:

1. **Before / after text**: the strongest opening, needs no explanation.
2. **The tour** (`make tour`): one prompt walked through the model, with real numbers on screen.
3. **A chart** (`demo/assets/flagship-val-loss.svg`): the overfitting story in one picture.
4. **The web UI**: a live, streaming model with a dropdown. This is the only thing that looks like a product.
5. **The git history**: `fix:` commits prove you found and fixed your own mistakes.

Do not film training. It takes about 17 minutes to reach the best checkpoint, and nothing on screen changes.

## 1. One-time setup (do this before you record)

The trained checkpoints are **not in git** (`runs/` is ignored). On this laptop they already exist. On another machine
create them first with `./scripts/reproduce.sh` (roughly 30 to 45 minutes).

```bash
cd gpt-from-scratch
ls runs/flagship-best runs/chat-who      # both must list a checkpoint file
.venv/bin/gpt-scratch --help             # proves the venv works
make tour | head -5                      # should print "Tour of a 10,670,592-parameter GPT ..."
```

If `.venv` is missing: `python3 -m venv .venv && .venv/bin/pip install -e ".[serve,dev]"`.

## 2. Screen layout

| Item | Setting |
|---|---|
| Display | 1280x720 or 1920x1080, but record a **selected portion** so the text is large |
| Terminal font | 18 pt or larger, high-contrast theme, window about 100 columns wide |
| Browser | One window, zoom 125%, bookmarks bar hidden, one tab per thing you will show |
| Do-not-disturb | On. Close Slack, mail and anything that pops up |
| Hide | `.env` files, your email address, any private repo names, the macOS menu bar clock if you want |

Open these **before** pressing record, each in its own tab or pane, so you only switch, never type paths:

| Tab / pane | Contents |
|---|---|
| Terminal 1 | `cd` into the repo; run `clear` |
| Terminal 2 | `.venv/bin/gpt-scratch serve --run runs/flagship-best runs/chat-who --device cpu` (leave running) |
| Browser tab A | `http://127.0.0.1:8000` (the web UI) |
| Browser tab B | `demo/assets/flagship-val-loss.svg` (open the file in the browser) |
| Browser tab C | The GitHub repo page (README, then Commits) |
| Editor | `src/gpt_scratch/model/attention.py` at `forward` (5-minute version only) |

Serve on 127.0.0.1 only. If you see a port error: `lsof -i :8000`, kill the process, restart.

## 3. Record

macOS has a recorder built in:

1. Press **Cmd + Shift + 5**.
2. Choose **Record Selected Portion** and drag a box over the area you will show.
3. Options: microphone on or off (off is fine; captions carry the feed cut), show mouse clicks on.
4. Click **Record**. Stop with the menu-bar button or **Cmd + Ctrl + Esc**.
5. The file lands on the Desktop as a `.mov`. Do not commit it; videos are git-ignored.

Record **one take per beat**, 10 to 20 seconds each. A fluffed beat costs you 15 seconds to redo, not the whole video.
Pause a second before and after each beat so you have room to cut.

### Beat by beat (what to type or click)

| Beat | Do this |
|---|---|
| 1 Before / after | Show `demo/assets/samples/step-000000-untrained.txt` and `step-001500-best.txt` side by side (two editor panes or `head -c 400 file`) |
| 2 Tour | In Terminal 1 run `make tour`. Scroll slowly to **section 2** and stop for 3 seconds on the percentages |
| 3 Benchmark | Run `./scripts/demo.sh bench`. It takes a short while; cut the wait in editing |
| 4 Chart | Switch to browser tab B. Hover nothing; let the viewer read the curve |
| 5 Web UI | Tab A: pick the flagship model, type `ROMEO:`, press **Enter**. Then choose the chat model in the dropdown and send `Speak as JULIET.` |
| 6 History | Terminal 1: `git log --oneline --grep='^fix'` (14 mistake-and-fix commits), then show the GitHub repo page |

## 4. Edit and add captions

Most LinkedIn video is watched muted, so captions are not optional.

- **Easiest:** import the clips into CapCut, iMovie or Descript, drop them on a timeline in beat order, and add text
  captions using the CAPTION column of [SCRIPT-90s.md](SCRIPT-90s.md). Keep each caption on screen for at least 3 seconds.
- **If you narrate:** use the editor's auto-captions, then correct the numbers by hand (auto-captions mangle "KV", "RoPE",
  "10.7 million").
- **Export:** 1080p, MP4 (H.264), under 90 seconds for the feed cut. Square (1:1) or 4:5 takes more feed space on a
  phone than 16:9.
- Add a title card of 2 seconds at the start only if the first frame is not already interesting. The gibberish-vs-verse
  frame usually is.

## 5. Pre-flight checklist

- [ ] `ls runs/flagship-best runs/chat-who` lists both checkpoints
- [ ] `make tour` prints without errors
- [ ] The server tab responds at `127.0.0.1:8000` and Enter sends a message
- [ ] Notifications off, no secrets or email visible
- [ ] Dry run of every beat once, timed
- [ ] Numbers you will say match the table in `SCRIPT-90s.md`
- [ ] Repo link is ready for the **first comment**, not the post body

## 6. What to be careful about

- Say the failure out loud: the chat model scores **3 to 4% on unseen scenes against a 25% guessing baseline**. That
  honesty is what makes the rest credible.
- Do not call it a chatbot or say it understands Shakespeare.
- The benchmark's speed-up depends on your machine. If your screen shows a different number, say the number on your screen.
- Nothing is posted anywhere until you decide. The LinkedIn draft is in [LINKEDIN.md](LINKEDIN.md); edit it into your own voice.
