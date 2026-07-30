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
