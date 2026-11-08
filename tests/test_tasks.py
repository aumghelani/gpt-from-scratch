import pytest

from gpt_scratch.chat.tasks import (
    Speech,
    evaluate_who,
    frequent_speakers,
    parse_speeches,
    speak_conversations,
    split_speeches,
    who_conversations,
)

PLAY = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

MENENIUS:
What work's, my countrymen, in hand? Where go you
With bats and clubs? The matter? Speak, I pray you.

First Citizen:
Our business is not unknown to the senate.
"""


def test_parse_speeches_reads_names_and_joins_lines():
    speeches = parse_speeches(PLAY)
    assert [s.speaker for s in speeches] == ["First Citizen", "All", "MENENIUS", "First Citizen"]
    assert speeches[2].text == "What work's, my countrymen, in hand? Where go you\nWith bats and clubs? The matter? Speak, I pray you."


def test_parse_speeches_skips_blocks_that_are_not_speeches():
    text = "ACT I\n\nFirst Citizen:\nHello there, friend.\n\nthis block has a colon at the end of a long sentence that is not a name:\nx"
    assert [s.speaker for s in parse_speeches(text)] == ["First Citizen"]


def test_parse_speeches_ignores_empty_speeches():
    assert parse_speeches("ROMEO:\n\nJULIET:\nHi there my love.") == (Speech("JULIET", "Hi there my love."),)


def test_split_keeps_order_and_puts_the_tail_in_the_held_out_part():
    speeches = tuple(Speech("A", f"line {i}") for i in range(10))
    train, held = split_speeches(speeches, 0.2)
    assert train == speeches[:8] and held == speeches[8:]


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_a_fraction_outside_the_open_interval(fraction):
    with pytest.raises(ValueError):
        split_speeches((Speech("A", "x"),), fraction)


def test_frequent_speakers_uses_a_minimum_count():
    speeches = (Speech("A", "1"), Speech("A", "2"), Speech("B", "3"))
    assert frequent_speakers(speeches, 2) == frozenset({"A"})


def test_who_conversations_ask_for_the_speaker_of_the_opening_lines():
    speeches = (Speech("ROMEO", "But, soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun."),)
    (user, assistant), = who_conversations(speeches, frozenset({"ROMEO"}), max_lines=1)
    assert user.role == "user" and assistant.role == "assistant"
    assert user.content == "Who says this?\nBut, soft! what light through yonder window breaks?"
    assert assistant.content == "ROMEO"


def test_who_conversations_skip_other_speakers_and_short_lines():
    speeches = (Speech("ROMEO", "Hi."), Speech("JULIET", "A long enough line to be worth asking about."))
    assert who_conversations(speeches, frozenset({"ROMEO"}), min_chars=10) == []
    assert len(who_conversations(speeches, frozenset({"JULIET"}), min_chars=10)) == 1


def test_speak_conversations_ask_for_a_speech_in_a_characters_voice():
    speeches = (Speech("ROMEO", "one\ntwo\nthree\nfour"), Speech("OTHER", "skipped line here"))
    (user, assistant), = speak_conversations(speeches, frozenset({"ROMEO"}), max_lines=2)
    assert user.content == "Speak as ROMEO."
    assert assistant.content == "one\ntwo"


def test_evaluate_who_scores_exact_matches_against_a_majority_baseline():
    def convo(speaker):
        return who_conversations([Speech(speaker, "a sufficiently long line of text")], frozenset({speaker}))[0]

    cases = [convo("A"), convo("A"), convo("A"), convo("B")]
    answers = iter(["A", "a ", "B", "B"])  # case and surrounding space are normalised
    report = evaluate_who(cases, lambda prompt: next(answers))
    assert (report.total, report.correct) == (4, 3)
    assert report.accuracy == 0.75 and report.majority_baseline == 0.75
    assert report.mistakes == (("A", "B"),)


def test_evaluate_who_needs_at_least_one_case():
    with pytest.raises(ValueError):
        evaluate_who([], lambda prompt: "")
