# gigabpe

Fast BPE tokenizer **training** with byte-exact HuggingFace `tokenizers`
parity.

**12.9 GB of FineWeb → 32k vocab in 38 s, against HuggingFace's 257 s — same
ByteLevel pretokenizer, all 31,790 merges byte-identical.** At 19.4 GB it uses
2.9 GB of RAM where HuggingFace needs 36.3 GB.

Whitespace, ByteLevel (GPT-2 regex) and WordPiece-style pretokenization. Zero
runtime dependencies. Rust, with Python bindings.

## Why

Training is the one stage of the data pipeline with no fast, exact tool, and
the incumbents run out of memory before they run out of time:

- [tokenizers#1681](https://github.com/huggingface/tokenizers/issues/1681):
  20 GB corpus OOMs on 1.5 TB and 2 TB machines.
- [#1795](https://github.com/huggingface/tokenizers/issues/1795): 100 GB of RAM
  supports ~1.5 GB of Chinese JSONL. [#1824](https://github.com/huggingface/tokenizers/issues/1824):
  a 131k vocab exceeded 750 GB.
- [sentencepiece#1021](https://github.com/google/sentencepiece/issues/1021):
  31.2 GB, vocab 4096, **1.8 TB** of memory, unfinished at 24 hours.

Measured here, HuggingFace needs **1.9x the corpus size in RAM**. That is the
mechanism behind #1681. gigabpe needs 0.15x.

**This is not a new algorithm.** Incremental pair counts with an inverted index
and a lazy heap is standard, formalized by
[Zouhar et al. 2023](https://arxiv.org/abs/2306.16837) and already implemented
in `tokenizers`, SentencePiece and rustbpe. The contribution is phase-1
architecture, memory layout, and an unusually thorough parity contract.

## Results

**12.9 GB FineWeb, vocab 32000, 16 cores — the headline, same pretokenizer
both sides:**

| pretokenizer | merges | gigabpe | HF | identical |
|---|---|---|---|---|
| ByteLevel | 31,790 | **38.0 s** | 257.1 s | yes |
| whitespace | 16,969 | **107.8 s** | 496.9 s | yes |

**19.4 GB on 16 cores / 64 GiB** — a deliberately modest box, since #1681 is
about OOM: gigabpe **47.3 s / 2.9 GB** (ByteLevel), 137.4 s / 7.2 GB
(whitespace); HuggingFace 730.9 s / 36.3 GB; rustbpe 1216.7 s / 5.8 GB;
SentencePiece **SIGSEGV**.

**1 GB, one container, all seven trainers, median of 3** — the only mutually
comparable table: gigabpe **6.7 s** · gigatoken 25.8 s · YouTokenToMe 26.3 s
· HF 46.9 s (byte-identical) · ffbpe 52.2 s · rustbpe 78.0 s · SentencePiece
108.1 s. Like-for-like whitespace with verified identical output: **20.1 s vs
101.1 s (5.0x) on 3.9x less memory.**

**Degenerate corpora** (real genomic FASTA, single-line JSON, minified JS,
CR-only text): across 20 configurations gigabpe completed **all 20**,
HuggingFace **15**, and every one of the 15 was byte-identical.

Full methodology, variance, and the designs that were measured and rejected:
[BENCHMARKS.md](BENCHMARKS.md).

### What not to claim

**"Fastest of the seven BPE trainers that exist, on web text"** is supported —
all seven are in the one-session table above. But keep the caveats: only the
HuggingFace comparison has verified identical output, and **rustbpe is ~15%
faster on single-giant-pretoken corpora**. The strongest claim remains the
byte-identical one, because it needs no caveat at all.

Numbers are good to roughly one significant figure: identical configurations
vary **20–28% between cloud allocations**, against ±2% within a single one.

### HuggingFace gets slower as you add cores

One box, one binary, one corpus, varying only `RAYON_NUM_THREADS`:

| threads | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| HF @ 1 GB | 134.6 s | 79.9 s | **66.0 s** | 83.9 s | 156.8 s |
| gigabpe | 22.2 s | 16.5 s | 14.5 s | 14.5 s | 14.3 s |

U-shaped, with the optimum moving with corpus size (4 threads at 100 MB, 16 at
1 GB). Its rayon-parallel pair counting reduces per-thread hash maps, so more
cores means more merging work. SentencePiece's maintainer measured the same
strategy as ineffective-to-harmful independently
([sentencepiece#366](https://github.com/google/sentencepiece/issues/366)).

## Parity

Output must match `tokenizers` merge-for-merge including tie-breaking, or this
is a demo rather than a drop-in. [PARITY.md](PARITY.md) specifies HF's exact
semantics as read from its source — the tie-break rule, stale-heap handling,
and several behaviours that look like bugs but must be reproduced. That
document does not exist anywhere else, including HuggingFace's own docs.

`scripts/run_parity_ci.sh` gates every commit:

- seven corpus configurations: 32k vocab, special tokens that collide with
  merge strings, `max_token_length`, `min_frequency`, `limit_alphabet`,
  English + Chinese, and ByteLevel
- the ByteLevel pretokenizer diffed against HF over every BMP codepoint from
  U+0020 up (surrogates excluded) in 8 contexts (~508k cases) plus handcrafted
  control-character edge cases, and two real corpora (English + Chinese)
- 1000 randomized fuzz trials biased toward count ties and same-char runs
- **vocabularies** diffed alongside merge lists
- output identical across 1, 2, 3, 7 and 16 threads; decorated modes checked
  for reproducibility at 1, 4, 16
- parallel range readers forced on a small corpus via `GIGABPE_MIN_RANGE`
- CLI guards for inputs the merge format cannot represent

**Scale of verification.** The per-commit gate's largest corpus is 4.9 MB — CI
cannot download 13 GB per push. Merge lists have separately been diffed against
HF at 100 MB, 1 GB and **12.9 GB**.

**Two known divergences**, both in [PARITY.md](PARITY.md): HF's `i32` pair
counter wraps past 2^31 occurrences (reachable around 120–150 GB of English);
and the decorated `##` / `</w>` modes, where HF is usually nondeterministic and
we differ even where it is not. Byte-exact parity is claimed only for the
undecorated modes.

## Usage

```bash
pip install gigabpe
```

Wheels cover CPython 3.9–3.14 on Linux (x86_64, aarch64), macOS (arm64,
x86_64) and Windows (x64). Anything else builds from the sdist and needs a
Rust toolchain. To build from a checkout instead:

```bash
pip install maturin
maturin build --release --features python --manifest-path gigabpe/Cargo.toml
pip install --find-links gigabpe/target/wheels gigabpe
```

```python
import gigabpe

gigabpe.train_tokenizer(
    ["corpus.txt"], vocab_size=32000, output="tokenizer.json",
    pretokenizer="bytelevel", special_tokens=["<|endoftext|>"],
)
vocab, merges = gigabpe.train_bpe(["corpus.txt"], vocab_size=32000)
```

Keyword arguments mirror `BpeTrainer`: `special_tokens`, `min_frequency`,
`max_token_length`, `limit_alphabet`, plus `pretokenizer` and `threads`.

```bash
cargo build --release --manifest-path gigabpe/Cargo.toml
GT=./gigabpe/target/release/gigabpe

$GT --vocab-size 32000 corpus.txt                          # merges to stdout
$GT --vocab-size 32000 --pretokenizer bytelevel corpus.txt # production config
$GT --vocab-size 32000 --wordpiece corpus.txt
$GT --vocab-size 32000 --words-tsv counts.tsv
```

Options: `--min-frequency`, `--special` (repeatable, order-significant),
`--max-token-length`, `--limit-alphabet`, `--threads`, `--pretokenizer`,
`--continuing-subword-prefix`, `--end-of-word-suffix`, `--wordpiece`,
`--vocab-out PATH`. `GIGABPE_STATS=1` prints stage RSS and phase-2 timings.

## How

Two phases with opposite characters, described fully in
[ARCHITECTURE.md](ARCHITECTURE.md).

**Phase 1** (parallel, I/O bound): parallel range readers → scanners that split
and hash → disjoint shard owners that count. Sharding by hash means each unique
word is stored exactly once machine-wide and the combine is a concatenation.

**Phase 2** (sequential, memory bound): pair counts maintained incrementally
with an inverted index and a lazy max-heap, so a merge costs O(affected
occurrences). Words live in one flat arena of `u32` token ids; a merge only
shrinks a word, so slice starts are fixed and the arena never reallocates.

Both memory wins came from profiling rather than intuition: phase 1's
`HashMap<String, u64>` held 1.7 GB of a 2.26 GB peak while the pair index
everyone expects to be the hazard was 1 MB.

## Caveats

- **A corpus with no cut points loses phase-1 parallelism.** Seven of eight
  reader ranges find no boundary and retire. Measured on 2 GB of single-line
  JSON against the same bytes with newlines: **160.9 s vs 80.8 s (2.0x)**, with
  memory unaffected (5.6 vs 5.8 GB). Under *whitespace* the whole file is one
  word and it does not finish in an hour.
- **A single pretoken must fit in 4 GiB.** Batch offsets are `u32`; larger
  aborts cleanly rather than corrupting output.
- **Merge output cannot represent a token containing a space**, so decoration
  flags and `--words-tsv` words reject spaces. The Python API is unaffected.
- **Input must be a regular file** — the readers seek, so pipes are rejected.
- **HuggingFace is non-reproducible with `##`.** Decorated token ids come from
  hash-map order and feed the tie-break, making `WordPieceTrainer`
  non-reproducible by default. An open PR fixes it
  ([#2066](https://github.com/huggingface/tokenizers/pull/2066)). gigabpe
  registers those tokens in sorted order and *is* reproducible; agreement with
  any single HF run is ~99.6% of merges.
- **Peak RSS depends on the allocator**; macOS libmalloc is slow to return
  freed pages.
- **Scope.** No Unigram/SentencePiece model.

## Repository map

| file | what it is |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | every measurement, and which numbers to quote |
| [PARITY.md](PARITY.md) | HF's exact training semantics — the reusable artifact |
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the two phases work |
| [PRIOR_ART.md](PRIOR_ART.md) | competitors, and an audit of what their benchmarks report |
| [docs/CORRECTIONS.md](docs/CORRECTIONS.md) | every claim withdrawn, and why |
| [docs/publishing.md](docs/publishing.md) | ship/blog plan and the decision not to submit a paper |
| [docs/upstream-issues.md](docs/upstream-issues.md) | the phantom-merge bug, filed as [tokenizers#2320](https://github.com/huggingface/tokenizers/issues/2320) |
| [docs/audit-sources/](docs/audit-sources/) | archived, checksummed sources for every quote |

## License

Apache-2.0. See [NOTICE](NOTICE) for what is derived from HuggingFace
`tokenizers` (test vectors and the behavioural specification).
