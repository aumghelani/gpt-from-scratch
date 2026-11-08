import math

import pytest
import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer
from gpt_scratch.tour.facts import (
    attention_links,
    cache_growth,
    entropy_bits,
    next_token_candidates,
    param_breakdown,
    random_guess_loss,
    temperature_entropies,
    token_pieces,
)

TEXT = "to be or not to be that is the question"


@pytest.fixture
def char_tok():
    return CharTokenizer.train(TEXT)


@pytest.fixture
def model(char_tok):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=len(char_tok.chars), block_size=32, n_layer=2, n_head=2, n_embd=16, pos_encoding="rope")
    return GPT(cfg).eval()


def test_token_pieces_pair_each_id_with_its_text(char_tok):
    pieces = token_pieces(char_tok, "to be")
    assert [text for _, text in pieces] == ["t", "o", " ", "b", "e"]
    assert [i for i, _ in pieces] == char_tok.encode("to be")


def test_token_pieces_show_multi_character_bpe_tokens():
    bpe = BPETokenizer.train(TEXT * 20, 280)
    pieces = token_pieces(bpe, "to be or not")
    assert len(pieces) < len("to be or not")
    assert "".join(text for _, text in pieces) == "to be or not"


def test_param_breakdown_accounts_for_every_parameter(model):
    parts = param_breakdown(model)
    assert sum(parts.values()) == model.num_params()
    assert parts["attention"] > 0 and parts["feed-forward"] > 0 and parts["token embedding"] > 0


def test_param_breakdown_counts_a_learned_position_table_when_present(char_tok):
    cfg = GPTConfig(vocab_size=len(char_tok.chars), block_size=8, n_layer=1, n_head=2, n_embd=8)
    parts = param_breakdown(GPT(cfg))
    assert parts["position embedding"] == 8 * 8


def test_random_guess_loss_is_log_vocab():
    assert random_guess_loss(1024) == pytest.approx(math.log(1024))


def test_next_token_candidates_are_sorted_probabilities(model, char_tok):
    ids = char_tok.encode("to be")
    top = next_token_candidates(model, char_tok, ids, k=5)
    probs = [p for _, p in top]
    assert len(top) == 5 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0 + 1e-6


def test_entropy_is_zero_for_certainty_and_log2_n_for_uniform():
    assert entropy_bits(torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy_bits(torch.full((8,), 1 / 8)) == pytest.approx(3.0)


def test_higher_temperature_flattens_the_distribution():
    logits = torch.tensor([4.0, 2.0, 1.0, 0.0])
    h = temperature_entropies(logits, (0.3, 1.0, 2.0))
    assert h[0.3] < h[1.0] < h[2.0]


def test_attention_links_only_look_backwards(model, char_tok):
    ids = char_tok.encode("to be or")
    links = attention_links(model, char_tok, ids, layer=0)
    assert len(links) == len(ids)
    assert all(link.key_position <= link.query_position for link in links)
    assert all(0 < link.weight <= 1.0 + 1e-6 for link in links)


def test_attention_links_reject_a_layer_out_of_range(model, char_tok):
    with pytest.raises(ValueError, match="layer"):
        attention_links(model, char_tok, char_tok.encode("to"), layer=5)


def test_cache_growth_adds_one_position_per_generated_token(model, char_tok):
    ids = char_tok.encode("to be")
    rows = cache_growth(model, ids, steps=3)
    assert [r.cached_positions for r in rows] == [5, 6, 7, 8]
    assert [r.tokens_fed for r in rows] == [5, 1, 1, 1]
    assert rows[0].key_shape == (1, 2, 5, 8)
