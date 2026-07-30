# Benchmarks

All runs via `scripts/benchmark.py`, which reports wall time, peak RSS, and
merge-list parity together (a speedup with different output is not a speedup).
Corpus: FineWeb sample-10BT slices (real web text), whitespace pretokenization,
vocab 32000, one special token.

Machine: Apple M-series Mac, 10 cores, 34 GB RAM, macOS 25.5.
Baseline: `tokenizers` 0.22.2 (Python), rayon across all 10 cores.

## Current results (2026-07-30)

**Whitespace pretokenization**

| corpus | trainer | wall | peak RSS | speedup | parity |
|---|---|---|---|---|---|
| 100 MB | gigatrain | 1.7 s | 419 MB | 5.8x | IDENTICAL |
| 100 MB | HF BpeTrainer | 9.7 s | 1.0 GB | | |
| 1 GB | gigatrain | 9.4 s | 1.3 GB | 6.5x | IDENTICAL |
| 1 GB | HF BpeTrainer | 61.2 s | 4.7 GB | | |

**ByteLevel (GPT-2 regex) — the production configuration**

| corpus | trainer | wall | peak RSS | speedup | parity |
|---|---|---|---|---|---|
| 100 MB | gigatrain | 1.2 s | 224 MB | ~50x | IDENTICAL |
| 100 MB | HF BpeTrainer | 61.2 s (incl. pretok) | — | | |
| 1 GB | gigatrain | 8.5 s | 725 MB | — | IDENTICAL |
| 12.9 GB | gigatrain | **85 s** | **2.2 GB** | — | — |

ByteLevel favours gigatrain more than whitespace does: HF pays for a regex
engine per document, while this is a hand-written state machine. It also
produces far fewer unique pretokens (1.6M vs 4.1M at 1 GB) because
punctuation splits, which shrinks phase 2 substantially.

At 12.9 GB / ByteLevel: phase 1 70.6 s, phase 2 14.8 s, 8.96M unique
pretokens. Whitespace on the same file: 85.7 s total, 6.4 GB peak, 27.4M
unique pretokens.

### Phase 1 became the bottleneck, and was optimized twice

Once ByteLevel landed, the profile inverted: phase 1 was 85% of the 12.9 GB
run (88.9 s of 104 s), not the merge loop. Two fixes, both verified against
the full BMP parity sweep:

| change | phase 1 @ 100 MB | why |
|---|---|---|
| baseline | 1.06–1.21 s | |
| ASCII lookup for `\p{L}`/`\p{N}` | 0.67–0.86 s | a binary search over ~700 ranges ran per character, ~13 billion times on the 13 GB corpus |
| defer byte-to-unicode mapping | 0.45–0.48 s | the map is a bijection, so pieces can be counted raw and mapped only for the ~9M unique words instead of every occurrence |

Together ~2.4x on phase 1, taking 12.9 GB from 104 s to **85 s** and peak RSS
from 2.4 GB to 2.2 GB.

### The HF baseline at 12.9 GB

**HuggingFace did not finish.** Killed by a 60-minute watchdog, having spent
the run between 4 and 8 GB RSS while the machine held 12.5 GB of swap. This
is the #1681 failure mode (memory, not merge-loop time) on a 34 GB laptop.

It is not a clean speedup number and should not be quoted as one: a
swap-bound wall time measures this machine's SSD. What it does establish is
that the same corpus that gigatrain trains in under two minutes inside RAM
drives HF into an hour of thrashing on identical hardware.

### Against SentencePiece v0.2.2

SentencePiece shipped a lazy-priority-queue BPE optimization in v0.2.2
(2026-07-12) with a claimed 20x, so it is the current baseline to beat rather
than the older published numbers. Measured here, vocab 32000, same machine,
with `--max_sentence_length` raised from its 4192-byte default (which
silently drops most FineWeb documents) and `train_extremely_large_corpus`:

| corpus | SentencePiece v0.2.2 | HF 0.22.2 | gigatrain |
|---|---|---|---|
| 100 MB | 13.7 s / 539 MB | 9.7 s / 1.0 GB | 1.7 s / 419 MB |
| 1 GB | 112.7 s / 3.0 GB | 61.2 s / 4.7 GB | 9.4 s / 1.3 GB |

The 20x was against older SentencePiece, not the field: post-optimization it
is still ~2x slower than HuggingFace at these sizes, and ~12x slower than
gigatrain. It is more memory-efficient than HF, though not than gigatrain.
User time barely exceeds wall time (17.2 s vs 14.0 s at 100 MB), consistent
with its maintainer's statement that BPE training is single-threaded.

This is a speed and memory comparison only. SentencePiece BPE produces a
different tokenizer by design — normalization, a character-coverage alphabet,
U+2581 word prefixes, and pieces rather than a merge list — so no
merge-for-merge diff is possible.

### Measurement noise

Numbers taken after the HF run were affected by 12 GB of swap that macOS had
not reclaimed: the same 1 GB whitespace configuration measured 9.4 s / 1.3 GB
on a quiet machine and 11.3–19.4 s / 0.7–0.9 GB while swap was occupied
(slower wall, and lower RSS because the allocator behaves differently under
pressure). Treat single measurements on a loaded machine as indicative only.

## A retraction: this is not HF issue #1313

An earlier version of this file claimed the 12.9 GB run reproduced
[tokenizers #1313](https://github.com/huggingface/tokenizers/issues/1313).
It does not. That issue used `vocab_size=512` on ~13 billion characters of
unsegmented DNA-like data, so its merge loop runs only a couple of hundred
merges and the reported 10+ hours came from degenerate pretokenization, not
merge cost. A 32k-vocab FineWeb run is a different and far merge-heavier
workload.

The genuinely unanswered scale issues are the memory ones: #1681 (20 GB OOM
on 1.5–2 TB machines), #1795, #1824.

## Progression (1 GB corpus)

| stage | wall | peak RSS |
|---|---|---|
| milestone 3 (single-threaded, std HashMap) | 39.3 s | 3.4 GB |
| milestone 4 (parallel phase 1, FxHash, Vec pos) | 15.3 s | 2.4 GB |
| milestone 5 (arena layout, sharded shuffle) | 11.1 s | 1.4 GB |
| + token_chars table (4-byte symbols) | 9.4 s | 1.3 GB |

Phase 2 breakdown at 1 GB after the last step: alphabet 126 ms, tokenize
198 ms, initial pair count 338 ms, merge loop 7.06 s. The merge loop is now
~90% of phase 2 and ~75% of total runtime; it is sequential by construction,
so it is the ceiling on further gains. A sampling profile puts ~6% of it in
allocator churn (position-list growth) and the rest in the scan-and-update
body itself. The remaining structural idea is CLAUDE.md's linked-list
representation with position-indexed occurrences, replacing the full-word
rescan per merge; it is a real win on paper but a parity risk, so it belongs
in its own change with heavy fuzzing.

## Where the memory actually went

Profiled with `GIGATRAIN_STATS=1` (stage RSS + structure sizes). The
prediction in CLAUDE.md was that `pair_where` sets would be the hazard. They
were not — on 1 GB there are only 63k distinct pairs (~1 MB of counts) and
189 MB of position lists. **Phase 1 was the hog**: per-worker
`HashMap<String, u64>` accumulators held 1.7 GB of the 2.26 GB peak, because
every unique word cost a 24-byte header plus its own heap allocation plus
allocator rounding, and frequent words were stored once per worker.

Fixes, in order of effect:

1. Arena-backed word counter, disjoint hash shards (1.7 GB -> 422 MB).
2. Word strings dropped as soon as words are tokenized to symbol IDs.
3. Flat symbol arena instead of a `Vec<Symbol>` per unique word (removes 4.1M
   allocations).

## Phase 1 design: three variants measured

Same 1 GB corpus, 10 cores:

| design | phase 1 wall | phase 1 RSS | scaling 1->10 threads |
|---|---|---|---|
| per-worker maps, merge at end | 2.4 s | 1.3 GB | — |
| broadcast chunks to shard owners | 4.8 s | 422 MB | 9.8 s -> 4.8 s (2.0x) |
| **shuffle: reader -> scanners -> shards** | **1.3 s** | **468 MB** | 6.4 s -> 1.3 s (4.8x) |

The broadcast variant fixes memory but replicates splitting and hashing on
every worker, which is a fixed floor that gets relatively worse as core count
rises. The shuffle divides every stage, which is why it wins on both axes.

## Generalization: what is machine-specific and what is not

**Portable (algorithmic / data-layout).** Everything that produced the gains
above is structural, not tuned to this chip: no SIMD, no intrinsics, no
`target-cpu` flags, no assumptions about cache sizes or core count (thread
count is read at runtime). The wins come from doing less work and allocating
less memory — fewer allocations, smaller records, no replicated scanning, no
single-threaded merge. Those hold on any CPU.

**Ratios that will move.** The specific multipliers are this machine's:

- Apple Silicon has unusually high per-core memory bandwidth, which flatters
  the pointer-chasing merge loop for *both* trainers. On a server CPU with
  more cores but less bandwidth per core, phase 2 (single-threaded, memory-
  bound by construction) should slow for both sides; the ratio may narrow.
- Peak RSS depends on the allocator. macOS libmalloc is slow to return freed
  pages, so measured peak overstates live data; glibc or jemalloc may report
  lower peaks for the same code.
- Only 10 cores were available. Phase 1 scales 4.8x on 10 threads and its
  reader is still single-threaded, so it will plateau on a 64-core box —
  parallel reads across files/offsets is the next fix. Phase 2 is sequential
  by construction and gains nothing from more cores, so at high core counts
  total time approaches phase 2 alone.
- HF's side is a moving target: 0.22.2 is much healthier than the 2023-era
  issues suggest. Their phase 1 reportedly degrades badly at high thread
  counts (issue #1313), so the gap could widen on a many-core box — untested,
  and it should not be claimed without a run.

**Not yet validated anywhere else.** These numbers are one machine, one OS,
one allocator, one corpus. Before publishing: rerun on a many-core Linux
server, and confirm the parity CI on a different architecture (the code has
no endianness assumptions, but the claim should be tested, not reasoned).
