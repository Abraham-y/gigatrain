# gigatrain

Fast BPE tokenizer **training** with byte-exact HuggingFace `tokenizers`
parity.

Trains a 32k vocabulary on **12.9 GB of FineWeb in 44 seconds** using 2.7 GB
of RAM, producing a merge list byte-identical to
`tokenizers.trainers.BpeTrainer`. On the same machine and corpus HuggingFace
takes 12.6 minutes and 29.8 GB; SentencePiece segfaults.

Whitespace, **ByteLevel (GPT-2 regex)** and WordPiece-style pretokenization.
Zero runtime dependencies. Rust, with Python bindings.

## Why

Training is the one stage of the data pipeline that has no fast, exact tool.
The incumbents run out of memory before they run out of time:

- [tokenizers #1681](https://github.com/huggingface/tokenizers/issues/1681):
  20 GB corpus OOMs on 1.5 TB and 2 TB machines. Closed on a workaround that
  commenters showed does not apply to training.
- [tokenizers #1795](https://github.com/huggingface/tokenizers/issues/1795),
  [#1824](https://github.com/huggingface/tokenizers/issues/1824): open. 100 GB
  of RAM supports ~1.5 GB of Chinese JSONL; a 131k vocab exceeded 750 GB.
- [sentencepiece #1021](https://github.com/google/sentencepiece/issues/1021):
  31.2 GB corpus, vocab 4096, **1.8 TB** of memory, unfinished at 24 hours.

The universal workaround is to sample the corpus down. That has a real cost:
Reddy et al. ([arXiv:2502.20273](https://arxiv.org/abs/2502.20273)) trained
396 tokenizers from 1 GB to 900 GB and found vocabulary composition does not
reach 90% overlap with the 900 GB tokenizer until **150–180 GB** — far above
what anyone trains on. They could not use HuggingFace and wrote their own
trainer.

**This is not a new algorithm.** Incremental pair counts with an inverted
index and a lazy heap is standard, formalized by
[Zouhar et al. 2023](https://arxiv.org/abs/2306.16837) and already implemented
in `tokenizers`, SentencePiece and rustbpe. The contribution is phase-1
architecture, memory layout, and an unusually thorough parity contract.

## Results

**12.9 GB FineWeb, vocab 32000, 64-core x86-64 Linux, 192 GiB, glibc:**

| trainer | wall | peak RSS | vs gigatrain | outcome |
|---|---|---|---|---|
| **gigatrain** (ByteLevel) | **43.8 s** | **2.7 GB** | — | ok |
| gigatrain (whitespace) | 129.4 s | 6.7 GB | 3.0x | ok |
| SentencePiece v0.2.2 | 135.0 s | 20.0 GB | — | **SIGSEGV** |
| HuggingFace 0.22.2 | 754.9 s | 29.8 GB | 17.2x | ok |
| rustbpe | 975.4 s | 5.3 GB | 22.3x | ok |

**1 GB, same machine:** gigatrain 7.4 s · gigatoken 18.8 s · ffbpe 65.4 s ·
rustbpe 88.2 s · SentencePiece 112.7 s · HuggingFace 244.4 s.

### HuggingFace gets slower as you add cores

The same 100 MB corpus, same HF version: **9.7 s on 10 cores, 181 s on 64**.
At 1 GB, 61.2 s becomes 244.4 s. Its rayon-parallel pair counting reduces
per-thread hash maps, so more cores means more merging work, not less.

This is the pathology behind
[#1313](https://github.com/huggingface/tokenizers/issues/1313) — 13 GB on
256 threads, unfinished after 10 hours — closed as stale in 2023 without
diagnosis. It is independently triangulated: published HF figures for 1 GB
rise monotonically with core count across three unrelated sources (59 s at
unstated cores, 97.7 s at 36, 244.4 s at 64).

gigatrain scales the other way: 14.5 s at 1 thread to 4.7 s at 48, flat
thereafter.

Full methodology, per-stage memory profiles, and the designs that were
measured and rejected are in [BENCHMARKS.md](BENCHMARKS.md). Reproduce with
`scripts/run_full_benchmark.sh` or `modal run scripts/modal_benchmark.py`.

## Parity

Output must match `tokenizers` merge-for-merge including tie-breaking, or this
is a demo rather than a drop-in. [PARITY.md](PARITY.md) specifies HF's exact
semantics as read from its source — the tie-break rule, stale-heap handling,
and several behaviours that look like bugs but must be reproduced. That
document does not exist anywhere else, including HuggingFace's own docs.

`scripts/run_parity_ci.sh` gates every commit:

- nine corpus configurations: 32k vocab, special tokens including ones that
  collide with merge strings, `max_token_length`, `min_frequency`,
  `limit_alphabet`, English + Chinese, and ByteLevel
- the ByteLevel pretokenizer diffed against HF over every non-surrogate BMP
  codepoint in 8 contexts (~508k cases), plus real corpora
- 1000 randomized fuzz trials biased toward count ties and same-char runs
- output identical across 1, 2, 3, 7 and 16 threads, in every mode
- the Python bindings' `tokenizer.json` round-tripped through
  `tokenizers.Tokenizer.from_file()` and checked to encode identical ids

CI runs the unit tests on Linux, macOS and Windows, and the parity gate on
Linux/glibc.

## Usage

### Python

```bash
pip install maturin
maturin build --release --features python --manifest-path gigatrain/Cargo.toml
pip install --find-links gigatrain/target/wheels gigatrain
```

```python
import gigatrain

# Writes a tokenizer.json that tokenizers.Tokenizer.from_file() loads and
# that encodes identically to a HuggingFace-trained tokenizer.
gigatrain.train_tokenizer(
    ["corpus.txt"], vocab_size=32000, output="tokenizer.json",
    pretokenizer="bytelevel", special_tokens=["<|endoftext|>"],
)

# Or get the vocab and merges directly.
vocab, merges = gigatrain.train_bpe(["corpus.txt"], vocab_size=32000)
```

Keyword arguments mirror `BpeTrainer`: `special_tokens`, `min_frequency`,
`max_token_length`, `limit_alphabet`, plus `pretokenizer` and `threads`.

### CLI

```bash
cargo build --release --manifest-path gigatrain/Cargo.toml
GT=./gigatrain/target/release/gigatrain

# Merges go to stdout in tokenizer.json order.
$GT --vocab-size 32000 corpus.txt

# ByteLevel (GPT-2 style), what production tokenizers use.
$GT --vocab-size 32000 --pretokenizer bytelevel corpus.txt

# WordPiece-style; equivalent to HF's WordPieceTrainer.
$GT --vocab-size 32000 --wordpiece corpus.txt

# Or from a precomputed word<TAB>count table.
$GT --vocab-size 32000 --words-tsv counts.tsv
```

Options: `--min-frequency`, `--special` (repeatable, order-significant),
`--max-token-length`, `--limit-alphabet`, `--threads`, `--pretokenizer`,
`--continuing-subword-prefix`, `--end-of-word-suffix`, `--wordpiece`.

`GIGATRAIN_STATS=1` prints stage-boundary RSS, structure sizes, and phase-2
sub-stage timings.

## How

Two phases with opposite characters, described fully in
[ARCHITECTURE.md](ARCHITECTURE.md).

**Phase 1** (parallel, I/O bound): parallel range readers → scanners that
split and hash → disjoint shard owners that count. Sharding by hash means each
unique word is stored exactly once machine-wide and the combine is a
concatenation rather than a merge.

**Phase 2** (sequential, memory bound): pair counts maintained incrementally
with an inverted index and a lazy max-heap, so a merge costs O(affected
occurrences) rather than O(corpus). Words live in one flat arena of `u32`
token ids; a merge only ever shrinks a word, so slice starts are fixed for the
whole run and the arena never reallocates or compacts.

Both major memory wins came from profiling rather than intuition. Phase 1's
`HashMap<String, u64>` accumulators held 1.7 GB of a 2.26 GB peak, while the
pair index everyone expects to be the hazard was 1 MB. And once ByteLevel
landed, phase 1 — not the merge loop — became ~83% of runtime.

## Caveats

- **Performance is measured on two machines**, a 10-core M-series laptop and a
  64-core Linux box. Correctness is CI-verified on Linux, macOS and Windows.
  Nothing is validated on ARM Linux or on more than 64 cores.
- **Peak RSS depends on the allocator.** macOS libmalloc is slow to return
  freed pages; identical work measured 419 MB there and 277 MB on Linux.
- **HuggingFace is non-reproducible with `##`.** Three runs over an identical
  corpus give three different merge lists and vocabularies, because decorated
  token ids come from hash-map order and feed the tie-break
  ([#2066](https://github.com/huggingface/tokenizers/issues/2066)). This makes
  `WordPieceTrainer` non-reproducible by default. gigatrain registers those
  tokens in sorted order and *is* reproducible; agreement with any single HF
  run is ~99.6% of merges, which is as close as a deterministic trainer can
  get to a moving target. Byte-exact parity is claimed only for the
  undecorated modes.
- **Scope.** No Unigram/SentencePiece model. Wheels are built for Linux,
  macOS and Windows by `.github/workflows/release.yml` on a version tag, but
  nothing is published to PyPI yet, so installation means a local
  `maturin build`.
- **Parity is against `tokenizers` 0.22.2**, pinned in CI. HF also has an open
  `i32` count-overflow bug
  ([#2058](https://github.com/huggingface/tokenizers/issues/2058)); this crate
  uses `i64`, so parity holds wherever HF itself has not overflowed.

## Prior art

Read [PRIOR_ART.md](PRIOR_ART.md) before citing anything here. It leads with
the findings that undercut this project. In short:

- **[gigatoken](https://github.com/marcelroed/gigatoken) already ships a BPE
  trainer** with a HuggingFace tie-breaking mode and a CI test asserting
  identical merge lists. It is the closest competitor by a wide margin —
  measured at 18.8 s against 10.2 s on 1 GB, so ~1.8x, not an order of
  magnitude. It is undocumented, and its parity test runs at ~120 KB.
  Measured here, its merges match HF exactly on LF corpora but diverge from
  rank 0 on CRLF corpora, because it does not replicate HF's line-at-a-time
  file feeding.
- **[ffbpe](https://github.com/tokn-ai/ffbpe)** (July 2026) advertises 1 GiB
  in 5.58 s, but at vocab 10k on Chinese text from a precomputed bigram
  inventory. Measured end-to-end from raw text at vocab 32k it is 65.4 s.
- **[fast-bytelevel-bpe-go](https://github.com/yunnian/fast-bytelevel-bpe-go)**
  is the only other project verifying byte-exact HF parity.
- **SentencePiece got ~20x faster in v0.2.2** (July 2026). That is against its
  own past, not the field; it remains slower than HuggingFace here.
- **GPU BPE *training* does not exist.** Everything branded that way encodes
  with a pre-trained merge table.

## License

Apache-2.0. See [NOTICE](NOTICE) for what is derived from HuggingFace
`tokenizers` (test vectors and the behavioural specification).
