# Benchmarks

All runs via `scripts/benchmark.py`, which reports wall time, peak RSS, and
merge-list parity together (a speedup with different output is not a speedup).
Corpus: FineWeb sample-10BT slices (real web text), whitespace pretokenization,
vocab 32000, one special token.

Machine: Apple M-series Mac, 10 cores, 34 GB RAM, macOS 25.5.
Baseline: `tokenizers` 0.22.2 (Python), rayon across all 10 cores.

## Current results (2026-07-30, after milestone 5)

| corpus | trainer | wall | peak RSS | speedup | parity |
|---|---|---|---|---|---|
| 100 MB | gigatrain | 1.7 s | 419 MB | 5.8x | IDENTICAL |
| 100 MB | HF BpeTrainer | 9.7 s | 1.0 GB | | |
| 1 GB | gigatrain | 11.1 s | 1.4 GB | 5.5x | IDENTICAL |
| 1 GB | HF BpeTrainer | 61.2 s | 4.7 GB | | |

At 1 GB: phase 1 (read + pretokenize + count) 1.3 s, phase 2 (merge loop)
9.8 s. 4.14M unique pretokens, 39.3M symbols.

## The 13 GB case

**gigatrain trains 12.9 GB of FineWeb in 85.7 s (6.4 GB peak RSS)** on 10
cores: phase 1 18.1 s, phase 2 67.4 s, 27.4M unique pretokens, 32k vocab.

**This is not a reproduction of HF issue #1313**, and an earlier version of
this file wrongly said it was. That issue used `vocab_size=512` on ~13
billion characters, so its merge loop runs only a couple of hundred merges;
the reported 10+ hours almost certainly came from degenerate pretokenization
on unsegmented data, not from merge-loop cost. A 32k-vocab FineWeb run is a
different and much merge-heavier workload. The honest open issues about
trainer scale are the memory ones — #1681 (20 GB OOM on 1.5–2 TB machines),
#1795, #1824 — not #1313.

The HF baseline on the identical file exceeded RAM on this machine and drove
it into swap (7+ GB RSS against 12.5 GB of swap in use), which is the failure
mode #1681 describes. Its wall time therefore measures this laptop's SSD as
much as the trainer, and is reported as such rather than as a clean speedup.

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
