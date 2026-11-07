import torch

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.gpt import GPT
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer
from gpt_scratch.train.samples import make_sample_hook

CFG = GPTConfig(vocab_size=20, block_size=32, n_layer=1, n_head=2, n_embd=16)


def test_writes_padded_step_file_with_requested_length(tmp_path):
    tok = CharTokenizer.train("abcdefghij" * 3 + "klmnopqrst")
    hook = make_sample_hook(tok, tmp_path, n_tokens=25, device="cpu")
    torch.manual_seed(0)
    hook(7, GPT(CFG))
    text = (tmp_path / "samples" / "step-000007.txt").read_text()
    assert len(text) == 25


def test_same_seed_gives_same_sample_for_same_weights(tmp_path):
    tok = CharTokenizer.train("abcdefghij" * 3 + "klmnopqrst")
    torch.manual_seed(0)
    model = GPT(CFG)
    hook = make_sample_hook(tok, tmp_path, n_tokens=20, device="cpu", seed=5)
    hook(1, model)
    hook(2, model)
    a = (tmp_path / "samples" / "step-000001.txt").read_text()
    b = (tmp_path / "samples" / "step-000002.txt").read_text()
    assert a == b


def test_works_with_bpe_tokenizer(tmp_path):
    tok = BPETokenizer.train("hello world " * 30, vocab_size=270)
    cfg = CFG.with_changes(vocab_size=tok.vocab_size)
    make_sample_hook(tok, tmp_path, n_tokens=10, device="cpu")(0, GPT(cfg))
    assert (tmp_path / "samples" / "step-000000.txt").exists()
