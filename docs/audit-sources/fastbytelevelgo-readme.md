# fast-bytelevel-bpe-go

[简体中文](README.zh-CN.md) | English

Train HuggingFace-compatible ByteLevel BPE tokenizers about 6x faster than HuggingFace's Python training path for the supported JSONL setup.

`fast-bytelevel-bpe-go` is a focused Go trainer that produces a HuggingFace-compatible `tokenizer.json` while aggressively optimizing the common large-JSONL ByteLevel BPE path.

In one local benchmark, it trained the same 32,779-token tokenizer in 739.59s vs 4406.97s for HuggingFace `tokenizers` through Python `train_from_iterator`, with identical vocab and merges.

See [docs/benchmarks.md](docs/benchmarks.md) for benchmark details and caveats.

This project intentionally targets a narrow, performance-focused setup:

- JSONL input
- one string field, default `text`
- `ByteLevel(add_prefix_space=false)`
- BPE model
- `initial_alphabet=ByteLevel.alphabet()`
- deterministic HuggingFace-compatible pair tie-breaks

It is not a full replacement for HuggingFace `tokenizers`. It intentionally specializes the training path to make large JSONL tokenizer training faster.

## Install

```bash
go install github.com/meipian/fast-bytelevel-bpe-go/cmd/fastbpe@latest
```

For local development:

```bash
go build -o bin/fastbpe ./cmd/fastbpe
```

## Usage

```bash
fastbpe \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 32768 \
  --out tokenizer.json
```

With special tokens:

```bash
fastbpe \
  --input data.jsonl \
  --field text \
  --vocab-size 32779 \
  --special-token '<|bos|>' \
  --special-token '<|eos|>' \
  --special-token '<|pad|>' \
  --out tokenizer.json
```

`--vocab-size` is the total vocabulary size, including special tokens. This matches HuggingFace `BpeTrainer(vocab_size=...)`.

## Validate Against HuggingFace

Install Python dependencies:

```bash
python -m pip install tokenizers
```

Train HuggingFace golden output:

```bash
python scripts/train_hf_tokenizer.py \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 512 \
  --output-dir /tmp/hf_tokenizer
```

Train with Go:

```bash
go run ./cmd/fastbpe \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 512 \
  --out /tmp/go_tokenizer.json
```

Compare:

```bash
python scripts/compare_tokenizers.py \
  --hf /tmp/hf_tokenizer/tokenizer.json \
  --go /tmp/go_tokenizer.json
```

Expected:

```text
vocab: SAME
merges: SAME
```

## Why It Can Be Faster

The trainer is specialized:

- reads JSONL directly in Go
- extracts only the configured string field instead of fully unmarshalling every record
- implements only the GPT-2/HF ByteLevel regex behavior needed by this setup
- stores pair keys as packed `uint64`
- keeps `pair -> affected words` indexes
- updates only affected words after each merge
- parallelizes large affected-word merge batches

## Current Limitations

This tool currently does not support:

- arbitrary normalizers
- arbitrary pre-tokenizers
- `add_prefix_space=true`
- WordPiece or Unigram
- byte fallback
- continuing subword prefix / end of word suffix
- non-JSONL input

The output file is a HuggingFace-compatible `tokenizer.json`, so it can be loaded with:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

## License

MIT
