from gpt_scratch.data.clean import clean_text


def test_normalizes_unicode_to_nfc():
    decomposed = "café"  # e + combining acute
    assert clean_text(decomposed) == "café"


def test_normalizes_line_endings():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_strips_control_chars_but_keeps_tab_and_newline():
    assert clean_text("a\x00b\x07c\td\ne") == "abc\td\ne"


def test_collapses_runs_of_blank_lines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_strips_trailing_spaces_per_line():
    assert clean_text("a   \nb\t \nc") == "a\nb\nc"


def test_is_idempotent():
    messy = "x\r\n\x00  \n\n\n\ný   "
    once = clean_text(messy)
    assert clean_text(once) == once


def test_does_not_mutate_input_and_returns_new_string():
    s = "a\r\nb"
    out = clean_text(s)
    assert s == "a\r\nb" and out == "a\nb"
