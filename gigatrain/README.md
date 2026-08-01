# gigatrain

Fast BPE tokenizer training with byte-exact HuggingFace `tokenizers` parity.

Trains a 32k vocabulary on 12.9 GB of FineWeb in 44 seconds using 2.7 GB of
RAM on a 64-core Linux box. Output is byte-identical to
`tokenizers.trainers.BpeTrainer`, verified by CI on corpora up to 1 GB.

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
