# Benchmarks

These numbers are from one local run and should be treated as a reference point, not a universal guarantee.

Hardware/software:

- Apple Silicon arm64
- Go 1.25.6
- HuggingFace `tokenizers` 0.23.1 through Python `train_from_iterator`
- JSONL input, first 846,882 non-empty `text` rows
- total vocab size 32,779, including 11 special tokens
- `min_frequency=2`
- `ByteLevel(add_prefix_space=false)`

Result:

| Trainer | Output | Time |
| --- | --- | ---: |
| HuggingFace Python binding | `tokenizer.json` | 4406.97s |
| fast-bytelevel-bpe-go | `tokenizer.json` | 739.59s |

Parity check:

```text
hf_vocab=32779 go_vocab=32779
hf_merges=32512 go_merges=32512
vocab: SAME
merges: SAME
```

The speedup mostly comes from the specialized JSONL reader, the narrow ByteLevel implementation, and parallel affected-word merge updates. It should not be read as "Go is faster than Rust"; HuggingFace `tokenizers` is a general-purpose library with a broader API surface.
