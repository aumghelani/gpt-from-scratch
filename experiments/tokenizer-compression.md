# Tokenizer compression: own BPE vs GPT-2's tokenizer

Bytes of text per token (higher = fewer tokens for the model to process). Merges are learned on the first 90% of
Tiny Shakespeare and every tokenizer is measured on the **unseen last 10%** (111,539 bytes).

```bash
gpt-scratch compare-tokenizer --text data/shakespeare-bpe/raw/tinyshakespeare.txt \
    --vocab-sizes 512 1024 2048 4096 --tiktoken gpt2        # needs: pip install -e ".[compare]"
```

| Tokenizer | Vocab | Tokens | Bytes/token |
|---|---:|---:|---:|
| own BPE | 512 | 59,400 | 1.88 |
| own BPE | 1,024 | 49,419 | 2.26 |
| own BPE | 2,048 | 43,558 | 2.56 |
| own BPE | 4,096 | 38,424 | 2.90 |
| tiktoken `gpt2` | 50,257 | 36,058 | 3.09 |

How to read it:

- Each doubling of the vocabulary buys less: +0.38, +0.30, +0.34 bytes/token. Bigger vocabularies also mean a
  bigger embedding table, so there is a real trade-off, not a free win.
- **This is not a fair fight in GPT-2's favour or ours.** My merges were learned on Shakespeare and tested on
  Shakespeare (in-domain); GPT-2's were learned on web text. On this text a 4k vocabulary gets within ~6% of a
  50k one. On code, other languages or modern web text the gap would be much larger.
- It shows my trainer and encoder compress sensibly. It is not a claim of parity with tiktoken: the pre-split regex
  here is simplified and the training corpus is ~1 MB.
