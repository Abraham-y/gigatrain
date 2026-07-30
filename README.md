# gigatrain

Fast BPE tokenizer **training** with byte-exact HuggingFace `tokenizers`
parity.

Trains a 32k vocabulary on **12.9 GB of FineWeb in 86 seconds** using 6.4 GB
of RAM on a 10-core laptop, producing a merge list byte-identical to
`tokenizers.trainers.BpeTrainer`.

Zero dependencies. Rust.

> **Status: work in progress.** The parity claim is enforced by CI. The
> performance numbers below are from a single machine and are not yet
> validated on other hardware — see [Caveats](#caveats) before relying on
> them.

## Why

Tokenizer *encoding* is a solved speed problem
([gigatoken](https://github.com/marcelroed/gigatoken) reaches ~24 GB/s).
Training is not. HuggingFace `tokenizers` becomes impractical well before
web scale:

- [#1313](https://github.com/huggingface/tokenizers/issues/1313): 13 GB
  corpus, 256 threads, unfinished after 10+ hours.
- [#1681](https://github.com/huggingface/tokenizers/issues/1681): 20 GB
  corpus OOMs during the merge phase on 1.5 TB and 2 TB machines.

The practical workaround everywhere is to sample the corpus down and train on
a fraction of it. That is a real cost: it means nobody studies vocabulary
design empirically at scale, because you cannot afford to train twenty
vocabularies on a terabyte.

## Results

FineWeb sample-10BT, whitespace pretokenization, vocab 32000, one special
token. Apple M-series, 10 cores, 34 GB RAM. Baseline: `tokenizers` 0.22.2
using all 10 cores.

| corpus | gigatrain | HF `BpeTrainer` | speedup | merge lists |
|---|---|---|---|---|
| 100 MB | 1.7 s / 419 MB | 9.7 s / 1.0 GB | 5.8x | identical |
| 1 GB | 9.4 s / 1.3 GB | 61.2 s / 4.7 GB | 6.5x | identical |
| 12.9 GB | 86 s / 6.4 GB | see caveats | — | — |

Full methodology, a per-stage memory profile, and the rejected designs are in
[BENCHMARKS.md](BENCHMARKS.md).

## Parity

Output must match `tokenizers` merge-for-merge, including tie-breaking, or
the tool is a demo rather than a drop-in. [PARITY.md](PARITY.md) documents
HF's exact semantics as read from its source at v0.22.2 — the tie-break rule,
the stale-heap-entry handling, and several behaviors that look like bugs but
must be reproduced.

`scripts/run_parity_ci.sh` is the gate, and no change lands without it:

- five corpus configurations (32k vocab, special tokens including ones that
  collide with merge strings, `max_token_length`, `min_frequency`,
  `limit_alphabet`, English + Chinese)
- 1000 randomized fuzz trials biased toward count ties and same-char runs
- output identical across 1, 2, 3, 7, and 16 threads

## Usage

```bash
cargo build --release --manifest-path gigatrain/Cargo.toml

# Train from raw text; merges go to stdout in tokenizer.json order.
./gigatrain/target/release/gigatrain --vocab-size 32000 corpus.txt

# Or from a precomputed word<TAB>count table.
./gigatrain/target/release/gigatrain --vocab-size 32000 --words-tsv counts.tsv
```

Options: `--min-frequency`, `--special` (repeatable, order-significant),
`--max-token-length`, `--limit-alphabet`, `--threads`.

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
- **Scope.** Whitespace pretokenization only; no
  `continuing_subword_prefix` / `end_of_word_suffix`; no PyO3 bindings yet.

## Prior art

`tokenizers` and SentencePiece are the incumbents, and SentencePiece
shipped a large BPE training speedup in v0.2.2 (July 2026) that any honest
comparison has to account for. A survey of existing fast BPE trainers, with
numbers, is in [PRIOR_ART.md](PRIOR_ART.md).

## License

Apache-2.0. See [NOTICE](NOTICE) for what is derived from HuggingFace
`tokenizers` (test vectors and the behavioral specification).
