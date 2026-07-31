# gigatrain

Fast BPE tokenizer **training** with byte-exact HuggingFace `tokenizers`
parity.

Trains a 32k vocabulary on **12.9 GB of FineWeb in 44 seconds** using 2.7 GB
of RAM (64-core Linux), producing a merge list byte-identical to
`tokenizers.trainers.BpeTrainer`. On the same machine and corpus HuggingFace
takes 12.6 minutes and 29.8 GB, and SentencePiece segfaults.

Supports both whitespace and **ByteLevel (GPT-2 regex)** pretokenization, the
latter verified against HF over every non-surrogate BMP codepoint.

Zero dependencies. Rust.

> **Status: work in progress.** The parity claim is enforced by CI. The
> performance numbers below are from a single machine and are not yet
> validated on other hardware — see [Caveats](#caveats) before relying on
> them.

## Why

Training a BPE vocabulary on a large corpus is memory-bound, and the
incumbents fall over on memory before they fall over on time:

- [tokenizers #1681](https://github.com/huggingface/tokenizers/issues/1681):
  20 GB corpus OOMs on 1.5 TB and 2 TB machines. Closed without an answer.
- [tokenizers #1795](https://github.com/huggingface/tokenizers/issues/1795),
  [#1824](https://github.com/huggingface/tokenizers/issues/1824): still open.
- [sentencepiece #1021](https://github.com/google/sentencepiece/issues/1021):
  31.2 GB corpus, vocab 4096, **1.8 TB** of memory, unfinished at 24 hours.
- [sentencepiece #782](https://github.com/google/sentencepiece/issues/782):
  maintainer's answer is that no workaround exists other than sampling less
  data.

The universal workaround is to sample the corpus down. That has a real cost:
Reddy et al. ([arXiv:2502.20273](https://arxiv.org/abs/2502.20273)) trained
396 tokenizers from 1 GB to 900 GB and found vocabulary composition does not
reach 90% overlap with the 900 GB tokenizer until **150–180 GB** — far above
what anyone actually trains on. They could not use HuggingFace for BPE and
wrote their own trainer.

**gigatrain is not a new algorithm.** Incremental pair counts with an
inverted index and a lazy heap is the standard approach, formalized by
[Zouhar et al. 2023](https://arxiv.org/abs/2306.16837) and already
implemented by `tokenizers` and SentencePiece. The contribution here is
memory layout and a parallel phase 1, plus an unusually thorough parity
contract.

## Results

FineWeb sample-10BT, vocab 32000, one special token.
Apple M-series, 10 cores, 34 GB RAM. Baseline: `tokenizers` 0.22.2
using all 10 cores.

| corpus | pretokenizer | gigatrain | HF `BpeTrainer` | speedup | merge lists |
|---|---|---|---|---|---|
| 100 MB | whitespace | 1.7 s / 419 MB | 9.7 s / 1.0 GB | 5.8x | identical |
| 1 GB | whitespace | 9.4 s / 1.3 GB | 61.2 s / 4.7 GB | 6.5x | identical |
| 100 MB | ByteLevel | 1.2 s / 224 MB | 61.2 s | ~50x | identical |
| 1 GB | ByteLevel | 8.5 s / 725 MB | — | — | identical |
| 12.9 GB | ByteLevel | 85 s / 2.2 GB | did not finish in 60 min | — | — |

Against the other trainers, same machine and corpora (speed and memory only —
neither produces a HuggingFace-compatible merge list, so there is nothing to
diff):

| corpus | gigatrain | rustbpe | SentencePiece v0.2.2 |
|---|---|---|---|
| 100 MB | 1.2 s / 224 MB | 9.9 s / 343 MB | 13.7 s / 539 MB |
| 1 GB | 8.5 s / 725 MB | 78.9 s / 1.20 GB | 112.7 s / 3.0 GB |

At 12.9 GB HuggingFace was killed by a watchdog after an hour, having driven
the machine to 12.5 GB of swap — the #1681 failure mode. That is a memory
result, not a clean timing comparison, and is reported as such.

Full methodology, a per-stage memory profile, and the rejected designs are in
[BENCHMARKS.md](BENCHMARKS.md).

## Parity

Output must match `tokenizers` merge-for-merge, including tie-breaking, or
the tool is a demo rather than a drop-in. [PARITY.md](PARITY.md) documents
HF's exact semantics as read from its source at v0.22.2 — the tie-break rule,
the stale-heap-entry handling, and several behaviors that look like bugs but
must be reproduced.

`scripts/run_parity_ci.sh` is the gate, and no change lands without it:

- seven corpus configurations (32k vocab, special tokens including ones that
  collide with merge strings, `max_token_length`, `min_frequency`,
  `limit_alphabet`, English + Chinese, and ByteLevel)
- the ByteLevel pretokenizer diffed against HF over every non-surrogate BMP
  codepoint in 8 contexts (~508k cases) plus real corpora
- 1000 randomized fuzz trials biased toward count ties and same-char runs
- output identical across 1, 2, 3, 7, and 16 threads, under both
  pretokenizers

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

# Train from raw text; merges go to stdout in tokenizer.json order.
./gigatrain/target/release/gigatrain --vocab-size 32000 corpus.txt

# Or from a precomputed word<TAB>count table.
./gigatrain/target/release/gigatrain --vocab-size 32000 --words-tsv counts.tsv

# ByteLevel (GPT-2 style), what production tokenizers use.
./gigatrain/target/release/gigatrain --vocab-size 32000 \
    --pretokenizer bytelevel corpus.txt
```

Options: `--min-frequency`, `--special` (repeatable, order-significant),
`--max-token-length`, `--limit-alphabet`, `--threads`, `--pretokenizer`.

`GIGATRAIN_STATS=1` prints stage-boundary RSS, structure sizes, and phase-2
sub-stage timings.

## How

Two phases with opposite characters, described in full in
[ARCHITECTURE.md](ARCHITECTURE.md).

**Phase 1** (parallel, I/O bound) is a three-stage pipeline: parallel range
readers → scanners that split and hash → disjoint shard owners that count.
Sharding by hash means each unique word is stored exactly once machine-wide
and the combine step is a concatenation rather than a merge.

**Phase 2** (sequential, memory bound) maintains pair counts incrementally
with an inverted index and a lazy max-heap, so each merge costs
O(affected occurrences) rather than O(corpus). Words live in one flat arena
of `u32` token IDs; since a merge only ever shrinks a word, slice starts are
fixed for the whole run and the arena never reallocates or compacts.

The largest single win was not in the merge loop. Profiling showed phase 1's
`HashMap<String, u64>` accumulators held 1.7 GB of a 2.26 GB peak — per-word
allocations and per-worker duplication — while the pair index everyone
expects to be the memory hazard was 1 MB.

## Caveats

- **One machine.** All numbers are from a single 10-core Apple Silicon laptop
  running macOS. The optimizations are structural (no SIMD, no intrinsics, no
  `target-cpu` flags, thread count read at runtime), so they should carry;
  the specific ratios will not. Peak RSS in particular is inflated by macOS
  libmalloc being slow to return freed pages.
- **The 12.9 GB HF baseline is not a clean comparison.** HF exceeded RAM on
  this machine and went deep into swap, so its wall time measures the SSD as
  much as the trainer. It is reported for what it is.
- **Phase 2 is the ceiling.** The merge loop is ~75% of runtime and
  sequential by construction, so more cores do not help it.
- **Scope.** Whitespace and ByteLevel pretokenization; no
  `continuing_subword_prefix` / `end_of_word_suffix`. Wheels are not published
  to PyPI, so installation is a local `maturin build`.

## Prior art

Read [PRIOR_ART.md](PRIOR_ART.md) before citing any claim here. The short
version:

- **[gigatoken](https://github.com/marcelroed/gigatoken) already ships a BPE
  trainer** with a HuggingFace tie-breaking mode and a CI test asserting
  identical merge lists. It is undocumented and validated at ~120 KB / vocab
  500; gigatrain differs in phase-1 architecture, memory layout, and parity
  scope, not in being first.
- **SentencePiece got ~20x faster at BPE training in v0.2.2** (July 2026,
  lazy priority queue). Any comparison against an older version is a
  strawman.
- HuggingFace at 1 GB is around 60 s, not hours — the gap at that size is
  ~6x, not orders of magnitude.
- **[rustbpe](https://github.com/karpathy/rustbpe) is the closest
  competitor**: 5–7x slower with comparable memory.

## License

Apache-2.0. See [NOTICE](NOTICE) for what is derived from HuggingFace
`tokenizers` (test vectors and the behavioral specification).
