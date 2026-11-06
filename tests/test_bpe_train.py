from gpt_scratch.tokenizer.bpe import BPETokenizer


def test_classic_worked_example_merges():
    # aaabdaaabac -> Z=aa, Y=ab, X=ZY (standard BPE walkthrough)
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=259)
    assert tok.merges == ((97, 97), (97, 98), (256, 257))
    assert tok.vocab_size == 259


def test_training_is_deterministic():
    text = "the quick brown fox jumps over the lazy dog " * 20
    assert BPETokenizer.train(text, 280).merges == BPETokenizer.train(text, 280).merges


def test_vocab_size_below_256_rejected():
    import pytest

    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=200)


def test_stops_early_when_no_pair_repeats():
    tok = BPETokenizer.train("abcdef", vocab_size=300)
    assert tok.merges == ()
    assert tok.vocab_size == 256


def test_decode_base_bytes_and_merged_tokens():
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=259)
    assert tok.decode([97, 98]) == "ab"
    assert tok.decode([256]) == "aa"
    assert tok.decode([258, 100]) == "aaabd"


def test_decode_handles_partial_utf8_without_crashing():
    tok = BPETokenizer.train("héllo", vocab_size=256)
    # first byte of a 2-byte char alone is invalid utf-8; must not raise
    assert isinstance(tok.decode([0xC3]), str)
