# gigatrain

Fast BPE tokenizer training with byte-exact HuggingFace `tokenizers` parity.

Trains a 32k vocabulary on 12.9 GB of FineWeb in 38 seconds against
HuggingFace's 257, same ByteLevel pretokenizer, on 16 cores; at 19.4 GB it
uses 2.9 GB of RAM where HuggingFace needs 36.3 GB. Output is byte-identical to
`tokenizers.trainers.BpeTrainer`: the per-commit CI gate diffs merge lists on
corpora up to 4.9 MB across seven configurations, and merge lists have
separately been diffed against HuggingFace at 100 MB, 1 GB and 12.9 GB of
FineWeb in manual runs.

```bash
pip install gigatrain
```

Wheels are published for CPython 3.9–3.14 on Linux (x86_64, aarch64), macOS
(arm64, x86_64) and Windows (x64). Any other platform builds from the sdist,
which needs a Rust toolchain.

```python
import gigatrain

# Write a tokenizer.json that tokenizers.Tokenizer.from_file() can load.
gigatrain.train_tokenizer(
    ["corpus.txt"], vocab_size=32000, output="tokenizer.json",
    pretokenizer="bytelevel", special_tokens=["<|endoftext|>"],
)

# Or get the vocab and merges directly.
vocab, merges = gigatrain.train_bpe(["corpus.txt"], vocab_size=32000)
```

Options: `special_tokens`, `min_frequency`, `max_token_length`,
`limit_alphabet`, `pretokenizer` (`"whitespace"` or `"bytelevel"`), `threads`.

Full documentation, benchmarks, and the parity specification:
https://github.com/Abraham-y/gigatrain
