# Benchmarks

All runs: `scripts/benchmark.py`, which reports wall time, peak RSS, and
merge-list parity together (a speedup with different output is not a
speedup). Corpus: FineWeb sample-10BT slices (real web text), whitespace
pretokenization, vocab 32000, one special token.

Machine: Apple Silicon Mac, 10 cores, 34 GB RAM, macOS 25.5.
Baseline: `tokenizers` 0.22.2 (Python), rayon on all 10 cores.
gigatrain: **single-threaded** (phase-1 parallelism not yet implemented),
std HashMap (no ahash yet).

## 2026-07-30 (post-milestone-2 baseline, unoptimized)

| corpus | trainer | wall | peak RSS | parity |
|---|---|---|---|---|
| 100 MB | gigatrain | 5.1 s | 568 MB | — |
| 100 MB | HF BpeTrainer | 9.4 s | 1016 MB | IDENTICAL |
| 1 GB | gigatrain | 39.3 s | 3.4 GB | — |
| 1 GB | HF BpeTrainer | 52.5 s | 4.5 GB | IDENTICAL |

gigatrain internal split at 1 GB: phase 1 (read+pretokenize+count) 13.3 s,
phase 2 (merge loop) 25.6 s. Unique pretokens: 807 k @ 100 MB, 4.1 M @ 1 GB.

## 2026-07-30 (after milestone 4 + phase-2 data-structure work)

Changes: parallel streaming phase 1 (32MB whitespace-bounded chunks, worker
threads, first-seen merge), FxHash maps (vendored), position lists as
Vec<u32> instead of HashSet, 8-byte symbols. Parity CI green throughout.

| corpus | trainer | wall | peak RSS | parity |
|---|---|---|---|---|
| 100 MB | gigatrain (10 threads) | 2.5 s | 476 MB | — |
| 100 MB | HF BpeTrainer | 9.5 s | 1.0 GB | IDENTICAL |
| 1 GB | gigatrain (10 threads) | 15.3 s | 2.4 GB | — |
| 1 GB | HF BpeTrainer | 54.9 s | 3.9 GB | IDENTICAL |

gigatrain internal split at 1 GB: phase 1 13.3 s -> 2.9 s, phase 2
25.6 s -> 12.2 s. Speedup 1.3x -> 3.6x at 1 GB.

### Generalization caveats

The optimizations are data-layout/algorithmic and portable (std-only Rust,
no SIMD, no target-cpu flags). The *ratios* are machine-specific: Apple
Silicon's high per-core memory bandwidth flatters single-threaded merge
loops on both sides. Phase 1 currently has a single reader thread and a
single-threaded final map-merge — fine at 10 cores, an Amdahl ceiling at
64+; sharded reduction is the known fix. Validate on a many-core Linux box
before publishing claims (HF's phase-1 scaling reportedly collapses at high
thread counts per issue #1313, so the gap may widen there — unverified).

## Observations

- Parity holds at real-corpus scale: 29,298 and 25,168 merges identical.
- tokenizers 0.22.2 is far healthier at 1 GB than the 2023-era issue #1313
  (13 GB, 10+ h) implies — the dary_heap/ahash/compact_str work landed since.
  The honest claim target is the 13 GB+ regime, not 1 GB.
- **Memory is the binding constraint for the 13 GB headline**, as CLAUDE.md
  predicted: ~3.4 GB RSS per corpus-GB extrapolates to ~40 GB at 13 GB — more
  than this machine's RAM, for HF (~4.5 GB/GB) too. Milestone 5 (arena
  layout, compact pair index, streaming phase 1) gates that run.
- Next speed wins, in order: parallel phase 1 (13.3 s → ~2 s on 10 cores),
  faster hashing in phase 2 (std SipHash → ahash/fxhash), then memory layout.
