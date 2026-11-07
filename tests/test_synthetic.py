from gpt_scratch.chat.synthetic import passages_to_conversations
from gpt_scratch.chat.template import Message

TEXT = """HAMLET:
To be, or not to be,
that is the question.

short one
line two

KING:
Now is the winter
of our discontent
made glorious summer
by this sun of York.
"""


def test_builds_continue_the_passage_conversations():
    convs = passages_to_conversations(TEXT, min_lines=3)
    assert len(convs) == 2
    user, assistant = convs[0]
    assert user.role == "user" and assistant.role == "assistant"
    assert user.content.endswith("HAMLET:")
    assert assistant.content == "To be, or not to be,\nthat is the question."


def test_short_passages_skipped():
    assert passages_to_conversations("a\nb\n\nc\nd\n", min_lines=3) == []


def test_limit_and_determinism():
    big = "\n\n".join(f"P{i}:\nline one\nline two\nline three" for i in range(50))
    assert len(passages_to_conversations(big, min_lines=3, limit=7)) == 7
    assert passages_to_conversations(big, 3) == passages_to_conversations(big, 3)


def test_returns_immutable_message_tuples():
    conv = passages_to_conversations(TEXT, min_lines=3)[0]
    assert isinstance(conv, tuple) and all(isinstance(m, Message) for m in conv)
