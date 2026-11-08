"""Instruction tasks with checkable answers, derived from the speaker labels in a play-formatted corpus.

Two tasks, both with ground truth taken from the text itself:

* ``who``   - "Who says this?" + the opening lines of a speech  ->  the speaker's name
* ``speak`` - "Speak as ROMEO."                                  ->  one of that character's speeches

Because the answers are known, the ``who`` task can be scored (see `evaluate_who`) and compared with a
majority-class baseline, which is what tells you whether the model learned anything.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gpt_scratch.chat.synthetic import Conversation
from gpt_scratch.chat.template import Message

MAX_NAME_CHARS = 40
MAX_NAME_WORDS = 4
WHO_PROMPT = "Who says this?\n"


@dataclass(frozen=True)
class Speech:
    speaker: str
    text: str


def _speaker_name(first_line: str) -> str | None:
    """'First Citizen:' -> 'First Citizen'; None when the line is not a short speaker label."""
    line = first_line.strip()
    if not line.endswith(":") or len(line) > MAX_NAME_CHARS:
        return None
    name = line[:-1].strip()
    if not name or len(name.split()) > MAX_NAME_WORDS:
        return None
    return name


def parse_speeches(text: str) -> tuple[Speech, ...]:
    """Split blank-line separated blocks into speeches; blocks without a speaker label are skipped."""
    speeches: list[Speech] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        name = _speaker_name(lines[0])
        if name is not None:
            speeches.append(Speech(name, "\n".join(lines[1:])))
    return tuple(speeches)


def split_speeches(
    speeches: Sequence[Speech], val_fraction: float = 0.1
) -> tuple[tuple[Speech, ...], tuple[Speech, ...]]:
    """Order-preserving split; the tail is held out (mirrors the pretraining train/val split)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    cut = len(speeches) - max(1, round(len(speeches) * val_fraction))
    return tuple(speeches[:cut]), tuple(speeches[cut:])


def frequent_speakers(speeches: Sequence[Speech], min_count: int) -> frozenset[str]:
    counts = Counter(s.speaker for s in speeches)
    return frozenset(name for name, n in counts.items() if n >= min_count)


def who_conversations(
    speeches: Sequence[Speech], speakers: frozenset[str], min_chars: int = 30, max_lines: int = 2
) -> list[Conversation]:
    out: list[Conversation] = []
    for speech in speeches:
        if speech.speaker not in speakers:
            continue
        opening = "\n".join(speech.text.splitlines()[:max_lines])
        if len(opening) < min_chars:
            continue
        out.append((Message("user", WHO_PROMPT + opening), Message("assistant", speech.speaker)))
    return out


def speak_conversations(
    speeches: Sequence[Speech], speakers: frozenset[str], max_lines: int = 6
) -> list[Conversation]:
    return [
        (
            Message("user", f"Speak as {s.speaker}."),
            Message("assistant", "\n".join(s.text.splitlines()[:max_lines])),
        )
        for s in speeches
        if s.speaker in speakers
    ]


@dataclass(frozen=True)
class WhoReport:
    total: int
    correct: int
    accuracy: float
    majority_baseline: float
    mistakes: tuple[tuple[str, str], ...]  # (expected, got), first few only


def _normalise(answer: str) -> str:
    return answer.strip().casefold()


def evaluate_who(
    conversations: Sequence[Conversation], answer: Callable[[str], str], max_mistakes: int = 10
) -> WhoReport:
    """Ask `answer(user_text)` for every case; exact match after trimming and case-folding."""
    if not conversations:
        raise ValueError("need at least one case to evaluate")
    correct = 0
    mistakes: list[tuple[str, str]] = []
    for user, expected in conversations:
        got = answer(user.content)
        if _normalise(got) == _normalise(expected.content):
            correct += 1
        elif len(mistakes) < max_mistakes:
            mistakes.append((expected.content, got.strip()))
    counts = Counter(_normalise(expected.content) for _, expected in conversations)
    total = len(conversations)
    return WhoReport(total, correct, correct / total, max(counts.values()) / total, tuple(mistakes))
