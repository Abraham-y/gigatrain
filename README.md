# gigatrain

Fast BPE tokenizer **training** with byte-exact HuggingFace `tokenizers`
parity.

Trains a 32k vocabulary on **12.9 GB of FineWeb in 44 seconds** using 2.7 GB
of RAM. On the same machine and corpus HuggingFace takes 12.6 minutes and
29.8 GB; SentencePiece segfaults.

Output is byte-identical to `tokenizers.trainers.BpeTrainer` — **verified at
12.9 GB in both pretokenization modes**. See [Parity](#parity) for exactly
what is checked.

Whitespace, **ByteLevel (GPT-2 regex)** and WordPiece-style pretokenization.
Zero runtime dependencies. Rust, with Python bindings.

## Why

Training is the one stage of the data pipeline that has no fast, exact tool.
The incumbents run out of memory before they run out of time:

- [tokenizers #1681](https://github.com/huggingface/tokenizers/issues/1681):
  20 GB corpus OOMs on 1.5 TB and 2 TB machines. Closed on a workaround that
  commenters showed does not apply to training.
- [tokenizers #1795](https://github.com/huggingface/tokenizers/issues/1795)
  (open): 100 GB of RAM supports ~1.5 GB of Chinese JSONL.
  [#1824](https://github.com/huggingface/tokenizers/issues/1824) (closed same
  day it was filed): a 131k vocab exceeded 750 GB.
- [sentencepiece #1021](https://github.com/google/sentencepiece/issues/1021):
  31.2 GB, vocab 4096, **1.8 TB** of memory, unfinished at 24 hours — though
  note its log shows `Alphabet size=4` (genomic data), the same degenerate
  low-alphabet shape this repo declines to over-read elsewhere.

The universal workaround is to sample the corpus down. Reddy et al.
([arXiv:2502.20273](https://arxiv.org/abs/2502.20273)) trained BPE, UnigramLM
and WordPiece tokenizers on English data from 1 GB to **900 GB**, and report
diminishing returns only beyond roughly 150 GB — far above what anyone
actually trains on. They used HuggingFace for UnigramLM and WordPiece, but
built their BPE trainer on `minbpe` rather than using HF's.

**This is not a new algorithm.** Incremental pair counts with an inverted
index and a lazy heap is standard, formalized by
[Zouhar et al. 2023](https://arxiv.org/abs/2306.16837) and already implemented
in `tokenizers`, SentencePiece and rustbpe. The contribution is phase-1
architecture, memory layout, and an unusually thorough parity contract.

## Results

**12.9 GB FineWeb, vocab 32000, 64-core x86-64 Linux, 192 GiB, glibc:**

| trainer | pretokenizer | wall | peak RSS | outcome |
|---|---|---|---|---|
| **gigatrain** | ByteLevel | **43.8 s** | 2.7 GB | ok |
| **gigatrain** | whitespace | **129.4 s** | 6.7 GB | ok |
| SentencePiece v0.2.2 | its own | 135.0 s | 20.0 GB | **SIGSEGV** |
| HuggingFace 0.22.2 | whitespace | 754.9 s | 29.8 GB | ok |
| rustbpe | GPT-4 regex | 975.4 s | 5.3 GB | ok |

**The like-for-like comparison is whitespace against whitespace: 129.4 s vs
754.9 s, i.e. 5.8x**, on identical output. gigatrain's ByteLevel mode is
faster still (43.8 s, 17.2x HF's time) but produces a different tokenizer, so
quoting 17.2x as a speedup would break this repo's own rule that a speedup
with different output is not a speedup.

**1 GB on the same 64-core box:** gigatrain 7.4 s (ByteLevel) · rustbpe 88.2 s ·
SentencePiece 112.7 s · HuggingFace 244.4 s. (gigatrain's whitespace mode was
not measured at 1 GB on this box; the trainers it is listed against use their
own pretokenizers, so none of these is a like-for-like parity comparison.)

**19.4 GB on 16 cores / 64 GiB** — a deliberately modest box, because
[#1681](https://github.com/huggingface/tokenizers/issues/1681) is about OOM at
this corpus size: gigatrain 47.3 s / **2.9 GB** (ByteLevel) and 137.4 s /
7.2 GB (whitespace), against HuggingFace 730.9 s / **36.3 GB** (whitespace).
Like-for-like that is 5.3x faster on 5.0x less memory. HF needing 1.9x the
corpus size in RAM is the mechanism behind #1681; gigatrain needs 0.15x.
gigatoken (18.8 s) and ffbpe (65.4 s) were measured on the 10-core laptop
instead, where gigatrain is 10.2 s — see [PRIOR_ART.md](PRIOR_ART.md) for
those same-machine pairings.

### HuggingFace gets slower as you add cores

The same 100 MB corpus, same HF version: **9.7 s on 10 cores, 181 s on 64**.
At 1 GB, 61.2 s becomes 244.4 s. Its rayon-parallel pair counting reduces
per-thread hash maps, so more cores means more merging work, not less.

This is plausibly a contributing factor in
[#1313](https://github.com/huggingface/tokenizers/issues/1313) — 13 GB on
256 threads, unfinished after 10 hours, closed as stale in 2023. It is not a
reproduction of that issue, which used `vocab_size=512` on unsegmented data
and was diagnosed in-thread by a maintainer as degenerate pretokenization; see
the retraction in [BENCHMARKS.md](BENCHMARKS.md).

**Caveat: this is not a controlled core-count sweep.** The two machines differ
in ISA, OS and allocator as well as core count, and no HF thread scan has been
run on a single box. Core count is the most likely cause and the mechanism is
visible in HF's source, but the experiment that would prove it has not been
run.

gigatrain scales the other way on the 64-core box: 8.4 s at 1 thread to 4.7 s
at 48, flat through 96.

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

- seven corpus configurations: 32k vocab, special tokens including ones that
  collide with merge strings, `max_token_length`, `min_frequency`,
  `limit_alphabet`, English + Chinese, and ByteLevel
- the ByteLevel pretokenizer diffed against HF over every non-surrogate BMP
  codepoint from U+0020 up, in 8 contexts (~508k cases), plus real corpora
- 1000 randomized fuzz trials biased toward count ties and same-char runs
- **vocabularies** diffed alongside merge lists — ids, specials and the
  alphabet can drift while every merge stays identical
- output identical across 1, 2, 3, 7 and 16 threads for whitespace and
  ByteLevel; the decorated (`##`, `</w>`) modes are checked for reproducibility
  at 1, 4 and 16 threads, against themselves rather than against HF
- parallel range readers exercised on a small corpus via
  `GIGATRAIN_MIN_RANGE`, which lowers the 64 MiB-per-reader threshold so the
  chunk-boundary skip/overshoot rules actually execute in CI

A separate CI job builds the wheel and round-trips the Python bindings'
`tokenizer.json` through `tokenizers.Tokenizer.from_file()`, checking it
encodes identical ids. CI runs the unit tests on Linux, macOS and Windows,
and the parity gate on Linux/glibc.

**Scale of verification.** The parity gate runs on every commit and its
largest corpus is 4.9 MB (synthetic, vocab 32k) — CI cannot download 13 GB
per push. Merge lists have separately been diffed against HF at 100 MB, 1 GB
and **12.9 GB** of FineWeb, the last by
`modal run scripts/modal_benchmark.py::parity --size-mb 13000`:

| corpus | pretokenizer | merges | gigatrain | HF | result |
|---|---|---|---|---|---|
| 12.9 GB | ByteLevel | 31,790 | 38.0 s | 257.1 s | identical |
| 12.9 GB | whitespace | 16,969 | 107.8 s | 496.9 s | identical |
| 1 GB | whitespace | 25,168 | — | — | identical |
| 1 GB | ByteLevel | — | — | — | identical |
| 100 MB | ByteLevel | 31,800 | — | — | identical |
| 100 MB | whitespace | 29,298 | — | — | identical |

The 12.9 GB rows are at 16 cores, where HF is at its best; at 64 cores HF
takes 754.9 s on the same file.

**Two known divergences**, both documented in [PARITY.md](PARITY.md):
HF's `i32` pair counter wraps past 2^31 occurrences of one pair and silently
emits fewer merges (reachable around 120–150 GB of English text, where
gigatrain's output is arguably the correct one but is not HF's); and the
decorated `##` / `</w>` modes, where HF is usually nondeterministic and we
differ even in the cases where it is not.

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
`--continuing-subword-prefix`, `--end-of-word-suffix`, `--wordpiece`,
`--vocab-out PATH` (writes the vocabulary as a JSON array in id order, which
is what the parity harness diffs).

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
  token ids come from hash-map order and feed the tie-break. This makes
  `WordPieceTrainer` non-reproducible by default. An open PR fixes it
  ([#2066](https://github.com/huggingface/tokenizers/pull/2066), unmerged),
  and the bug is still present in 0.23.1. gigatrain registers those tokens in
  sorted order and *is* reproducible; agreement with any single HF run is
  ~99.6% of merges. That is **not** the best a deterministic trainer could do —
  in the minority of configurations where HF happens to be deterministic,
  gigatrain still differs on some of them, because it picks a different
  registration order rather than replicating HF's. See PARITY.md for the
  numbers. Byte-exact parity is claimed only for the undecorated modes.
- **A corpus with no cut points loses phase-1 parallelism.** The reader buffers
  until it finds a cut point (whitespace, or a newline under
  `--pretokenizer bytelevel`). On input with none, seven of eight reader ranges
  find no boundary and retire, and one thread buffers and scans everything.
  Measured on a 2.0 GB single-line JSON file against the same 2.0 GB with
  newlines every 1000 bytes: **155 s vs 9.3 s, and 2.2 GB vs 1.0 GB peak**. The
  memory side is mild (1.1x the file); the 16.7x slowdown is the real cost.
  Normal web text is unaffected. See
  [docs/degenerate-results.md](docs/degenerate-results.md).
- **Input must be a regular file.** Ranges come from the stat size and the
  readers seek, so pipes and process substitution are rejected with an error
  rather than silently producing an empty tokenizer.
- **A single pretoken must fit in 4 GiB.** Batch offsets are `u32`, so one
  "word" larger than that aborts (exit 101) rather than corrupting output.
  Under `--pretokenizer bytelevel` the cut rule is newline-only, so this means
  a single 4 GiB *line* — reachable with a one-line JSON dump or a file using
  `\r`-only line endings. Until recently this deadlocked instead of aborting
  whenever the run had only one scanner thread (`--threads` 1–3).
- **Merge output cannot represent a token containing a space.** The CLI prints
  `left<space>right`, so `--continuing-subword-prefix`, `--end-of-word-suffix`
  and `--words-tsv` reject values containing spaces. The Python API returns
  pairs and is unaffected.
- **Scope.** No Unigram/SentencePiece model. Wheels are built for Linux,
  macOS and Windows by `.github/workflows/release.yml` on a version tag, but
  nothing is published to PyPI yet, so installation means a local
  `maturin build`.
- **Parity is against `tokenizers` 0.22.2**, pinned in CI. HF also has an open
  `i32` count-overflow bug
  ([#2058](https://github.com/huggingface/tokenizers/issues/2058)); this crate
  uses `i64`, so parity holds wherever HF itself has not overflowed.

## What it was used for

The trainer made a large tokenizer-design sweep affordable — 36 vocabularies
across three corpus compositions in minutes rather than days.

**The first round of results has been retracted.** An adversarial audit found
the experimental setup was broken in several independent ways: the
multilingual held-out set turned out to be a single language, the
per-language equity numbers were measured in-sample, the corpora were not
language-balanced, and the "shared head" metric was measuring the UTF-8 byte
alphabet rather than learned merges. Details, in full, in
[docs/sweep-results.md](docs/sweep-results.md).

What survives is only the coarse shape, for English and code: vocabulary
overlap falls with smaller corpora and falls faster for larger vocabularies,
while fertility moves very little. The experiment is being rebuilt before any
number from it is quoted.

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
- **[ffbpe](https://github.com/tokn-ai/ffbpe)** (created December 2025;
  renamed from `unitoken` and released as v0.1.8 in July 2026) reports 1 GiB in
  5.58 s in a bounded-memory table — at vocab 10k on Chinese text from a
  precomputed bigram inventory, not as a headline speed claim. Measured
  end-to-end from raw text at vocab 32k it is 65.4 s.
- **[fast-bytelevel-bpe-go](https://github.com/yunnian/fast-bytelevel-bpe-go)**
  is the only other project verifying byte-exact HF parity.
- **SentencePiece got ~20x faster in v0.2.2** (July 2026). That is against its
  own past, not the field. It is slower than HuggingFace on the 10-core
  laptop, and faster than HF on the 64-core box — because HF degrades with
  core count and SentencePiece's BPE trainer is single-threaded.
- **GPU BPE *training* does not exist.** Everything branded that way encodes
  with a pre-trained merge table.

## License

Apache-2.0. See [NOTICE](NOTICE) for what is derived from HuggingFace
`tokenizers` (test vectors and the behavioural specification).
