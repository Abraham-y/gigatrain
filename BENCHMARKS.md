# Benchmarks

Every measurement in one place. **Superseded numbers are not kept here** — they
are in [docs/CORRECTIONS.md](docs/CORRECTIONS.md) with the reason they moved.

Harnesses: `scripts/benchmark.py` (local, parity-checking),
`scripts/modal_benchmark.py` (all runs below), `scripts/degenerate_benchmark.py`
(repeats, variance, failure-tolerant), `scripts/real_corpora.py` and
`scripts/degenerate_corpora.py` (corpora).

**A speedup with different output is not a speedup.** Rows where the two sides
run different pretokenizers are time/memory comparisons only and are marked.

---

## Which numbers to quote

| claim | number | where |
|---|---|---|
| **Headline** — same pretokenizer, verified identical output | 12.9 GB in **38.0 s** vs HF **257.1 s** (**6.8x**), 31,790 merges identical | [Parity at scale](#parity-at-scale) |
| Memory at scale | 19.4 GB in **2.9 GB** RAM vs HF **36.3 GB** | [19.4 GB](#194-gb-milestone-5-and-issue-1681) |
| Comparable multi-trainer ratios (all 7) | gigatoken 3.9x · YouTokenToMe 3.9x · HF 7.0x · ffbpe 7.8x · rustbpe 11.6x · SentencePiece 16.1x | [One-session](#one-session-comparison) |
| HF thread pathology | **2.4x** at 1 GB (1.34x at 36 threads); 10.3x at 100 MB | [Core-count sweep](#controlled-core-count-sweep) |
| Measurement noise | **20–28%** between allocations, ±2% within | [Variance](#between-allocation-variance) |

**"Fastest of the seven trainers that exist, on web text"** is now supported —
all seven are in the one-session table. Keep the caveats attached: only the
HuggingFace rows have verified identical output, and rustbpe is ~15% faster on
single-giant-pretoken corpora.

---

## Parity at scale

`modal run scripts/modal_benchmark.py::parity --size-mb 13000`. 12.9 GB FineWeb,
vocab 32000, 16 cores (deliberately not 64 — HF degrades with core count, so
this is the least favourable setting for gigatrain).

| pretokenizer | merges | gigatrain | HF | identical |
|---|---|---|---|---|
| ByteLevel | 31,790 | **38.0 s** | 257.1 s | yes |
| whitespace | 16,969 | **107.8 s** | 496.9 s | yes |

ByteLevel is both faster *and* produces more merges: it yields far fewer unique
pretokens (9.0M vs 27.4M), and the 32k vocabulary is reached with more merges
because its alphabet is 256 bytes rather than every character observed.

Merge lists have also been diffed at 100 MB and 1 GB in both modes. The
per-commit CI gate's largest corpus is 4.9 MB — see README, "Parity".

## 19.4 GB: milestone 5 and issue #1681

CLAUDE.md milestone 5 targets *"20 GB trains without OOM on a normal machine."*
16 cores / 64 GiB — modest on purpose, since #1681 is about OOM.

| trainer | pretokenizer | wall | peak RSS | outcome |
|---|---|---|---|---|
| **gigatrain** | ByteLevel | **47.3 s** | **2.9 GB** | ok |
| **gigatrain** | whitespace | **137.4 s** | 7.2 GB | ok |
| SentencePiece v0.2.2 | its own | 158.3 s | 27.2 GB | **SIGSEGV** |
| HuggingFace 0.22.2 | whitespace | 730.9 s | 36.3 GB | ok |
| rustbpe | GPT-4 regex | 1216.7 s | 5.8 GB | ok |

Like-for-like whitespace: **5.3x on 5.0x less memory.** HF needs **1.9x the
corpus size in RAM**, which is the mechanism behind #1681; gigatrain needs
0.15x. SentencePiece reproduced its 12.9 GB segfault on a different machine and
corpus size, so that crash tracks input scale.

Corpus is 19.4 GB, not 20 — the parquet-yield estimate was wrong (~3.2 GB of
text each, not 4–5). Fixed for future runs; reported at its true size.

## One-session comparison

**The only mutually comparable multi-trainer table here**, and now complete —
all seven trainers that exist. Every other table was assembled from separate
sessions, which is why gigatrain's own 1 GB figure appears three different ways
in this repo's history. One container, one page cache, median of 3 repeats,
16-core Linux, vocab 32000.

**1 GB FineWeb, ByteLevel**

| trainer | pretokenizer | wall | peak RSS | vs gigatrain | parity |
|---|---|---|---|---|---|
| **gigatrain** | ByteLevel | **6.7 s** ±3% | **537 MB** | — | — |
| gigatoken | its own | 25.8 s ±1% | 1.1 GB | 3.9x | — |
| YouTokenToMe | its own | 26.3 s ±2% | **6.4 GB** | 3.9x | — |
| HuggingFace | ByteLevel | 46.9 s ±2% | 1.7 GB | 7.0x | **identical (31,798)** |
| ffbpe | its own | 52.2 s ±4% | 1.2 GB | 7.8x | — |
| rustbpe | GPT-4 regex | 78.0 s ±4% | 1.1 GB | 11.6x | — |
| SentencePiece | its own | 108.1 s ±2% | 3.6 GB | 16.1x | — |

**1 GB whitespace** (like-for-like, verified identical output): gigatrain
**20.1 s** ±8% / 1.1 GB against HF **101.1 s** ±4% / 4.3 GB — **5.0x on 3.9x
less memory**, 25,169 merges identical.

**100 MB FineWeb, ByteLevel**

| trainer | wall | peak RSS | vs gigatrain | parity |
|---|---|---|---|---|
| **gigatrain** | **1.5 s** ±2% | **162 MB** | — | — |
| YouTokenToMe | 6.9 s ±2% | 1.6 GB | 4.6x | — |
| gigatoken | 7.6 s ±6% | 243 MB | 5.1x | — |
| ffbpe | 8.1 s ±5% | 393 MB | 5.4x | — |
| rustbpe | 8.9 s ±5% | 387 MB | 5.9x | — |
| SentencePiece | 13.8 s ±6% | 548 MB | 9.2x | — |
| HuggingFace | 16.1 s ±1% | 546 MB | 10.7x | **identical (31,801)** |

100 MB whitespace: gigatrain 3.3 s / 260 MB vs HF 26.8 s / 1.0 GB (8.1x),
29,299 merges identical.

**gigatrain is fastest of all seven at both sizes.** Three caveats belong with
that: only the HuggingFace rows have verified identical output (the other five
apply their own pretokenization); rustbpe is ~15% *faster* on single-giant-
pretoken corpora (see [Degenerate corpora](#degenerate-corpora)); and this is
one allocation, so the absolute times carry the 20–28% between-allocation
spread while the ratios do not.

**YouTokenToMe's memory is the outlier nobody reports.** 6.4 GB for a 1 GB
corpus — 12x gigatrain's, and the highest of any trainer here — while being
second-fastest. No published trainer benchmark reports memory, so this cost is
invisible in the literature.

### These numbers reproduce YouTokenToMe's own benchmark

Worth stating because it cuts against this repo's own audit. YouTokenToMe's
`benchmark.md` reports **25.4 s** for itself and **97.7 s** for HuggingFace on
1 GB English, on 36 cores. Measured here on 16 cores: **26.3 s** and (whitespace)
**101.1 s**.

Both reproduce closely. PRIOR_ART.md questions whether their HF baseline was
inflated by an unreported thread count; the answer appears to be *not much*,
which is consistent with the 1 GB core-count sweep being shallow (1.34x at 36
threads). Their benchmark holds up better than that criticism implied.

## Controlled core-count sweep

One box, one binary, one corpus, varying only `RAYON_NUM_THREADS`. 64-core
x86-64 Linux, vocab 32000, median of 3.

| threads | HF @100 MB | HF @1 GB | gigatrain @1 GB |
|---|---|---|---|
| 1 | 23.8 s | 134.6 s | 22.2 s |
| 4 | **15.4 s** | 79.9 s | 16.5 s |
| 16 | 17.9 s | **66.0 s** | 14.5 s |
| 32 | 29.2 s | 83.9 s | 14.5 s |
| 36 | — | 88.5 s | 14.4 s |
| 64 | **158.6 s** | 156.8 s | 14.3 s |

HF is U-shaped and **the optimum moves with corpus size** — 4 threads at
100 MB, 16 at 1 GB. Worst/optimum is 10.3x at 100 MB but only **2.4x at 1 GB**,
and at 36 threads (YouTokenToMe's core count, on their headline size) it is
**1.34x**. Peak RSS rises monotonically with threads (764 → 1215 MB at 100 MB).

The mechanism is visible in HF's source: rayon-parallel pair counting reduces
per-thread hash maps, so more cores means more merging work. SentencePiece's
maintainer measured the same strategy as ineffective-to-harmful independently
([sentencepiece#366](https://github.com/google/sentencepiece/issues/366)).

**Quote the 1 GB numbers**, not the 100 MB ones: the 10.3x exists only at a
configuration no published benchmark uses.

## Between-allocation variance

One fixed configuration (100 MB, ByteLevel, vocab 32k) on **8 freshly allocated
containers**, identity verified per probe via `/proc/sys/kernel/random/boot_id`
and `MODAL_TASK_ID` (8 distinct values, ~1 s uptime each).

| | within-run (3 repeats) | across 8 allocations |
|---|---|---|
| gigatrain | ±2% | **20%** (1.19 → 1.46 s) |
| HuggingFace | ±2% | **28%** (16.82 → 21.99 s) |

Repeats inside one allocation measure scheduler jitter, not reproducibility.
**No number here should be read to two significant figures.**

## Degenerate corpora

45 MB each, vocab 32000, 16 cores, median of 3, 1800 s timeout, one container.
Real corpora via `scripts/real_corpora.py` (UCSC hg38 chr21, npm registry
packuments, cdnjs bundles, Project Gutenberg), synthetic via
`scripts/degenerate_corpora.py`.

| corpus | gigatrain (bl) | HF (bl) | parity |
|---|---|---|---|
| dna_real (FASTA) | **13.7 s / 957 MB** | 73.9 s / 3.5 GB | identical |
| dna_real_oneline | **267 s / 800 MB** | **TIMEOUT** | — |
| dna_real_acgt_only | **251 s / 754 MB** | **TIMEOUT** | — |
| json_real_oneline | **4.0 s / 235 MB** | 115.3 s / 4.5 GB | identical |
| minjs_real | **0.6 s / 164 MB** | 20.9 s / 3.7 GB | identical |
| text_real_cjk | **0.9 s / 107 MB** | 21.9 s / 419 MB | identical |
| text_real_cr_only | **2.4 s / 75 MB** | 95.9 s / 4.1 GB | identical |

Across 20 configurations (10 corpora × 2 modes) **gigatrain completed all 20;
HuggingFace completed 15**, and every one of the 15 was byte-identical. rustbpe
is ~15% faster than gigatrain on the two single-giant-pretoken corpora
(235.7 s vs 266.8 s). SentencePiece failed 5 of 7 in an earlier pass — four
refusals ("Vocabulary size too high... set it to a value <= 28") and one
SIGABRT.

**Caveats.** `text_real_cjk` is one Gutenberg book repeated 18x and
`minjs_real` cycles 2.5x, so both have inflated redundancy and carry no claim.
`TIMEOUT` is censored data, not a proof of non-termination.

## Boundary-free input

A corpus with no cut point (no whitespace, or no newline under ByteLevel) forces
the reader to buffer the longest boundary-free run, and seven of eight reader
ranges find no boundary and retire — so phase 1 collapses to one thread. 2 GB
single-line JSON against the same bytes with a newline every 1000, one
container, 3 repeats:

| | wall | peak RSS |
|---|---|---|
| no cut points (ByteLevel) | 160.9 s ±4% | 5.6 GB |
| newline every 1000 B | 80.8 s ±3% | 5.8 GB |
| no cut points (**whitespace**) | **>3600 s TIMEOUT** | — |

**2.0x under ByteLevel, and memory is unaffected** (5.6 vs 5.8 GB). Under
whitespace the whole 2 GB is a single word and it does not finish in an hour.

The fix — cutting a long line at a safe interior pretoken boundary, found by
running the pretokenizer rather than reasoning about the regex — is designed
but unimplemented, because it touches the parity-critical path.

## Where the memory goes

Profiled with `GIGATRAIN_STATS=1`. CLAUDE.md predicted `pair_where` would be
the hazard. It was not: at 1 GB there are 63k distinct pairs (~1 MB of counts)
and 189 MB of position lists. **Phase 1 was the hog** — per-worker
`HashMap<String, u64>` accumulators held 1.7 GB of a 2.26 GB peak, because
every unique word cost a 24-byte header plus its own allocation plus rounding,
and frequent words were stored once per worker.

Fixes, in order of effect: arena-backed counter with disjoint hash shards
(1.7 GB → 422 MB); word strings dropped once tokenized; flat symbol arena
instead of a `Vec` per word (removes 4.1M allocations).

Phase-1 design, same 1 GB corpus, 10 cores:

| design | wall | RSS | scaling 1→10 threads |
|---|---|---|---|
| per-worker maps, merge at end | 2.4 s | 1.3 GB | — |
| broadcast chunks to shard owners | 4.8 s | 422 MB | 2.0x |
| **shuffle: reader → scanners → shards** | **1.3 s** | **468 MB** | **4.8x** |

## Optimizations tried and rejected

| change | rationale | result |
|---|---|---|
| 60/40 scanner/owner split | CPU split measured ~58/42 | no effect |
| byte-table ASCII path in piece scanning | stop decoding chars to classify | **10% slower** |
| zero-copy batching | removes a memcpy pass | slightly worse |
| thread-pool retune @10 cores | apparent oversubscription | inside noise, reverted |
| the same retune @64 cores | — | **worked: 1.46x, 2.2x less memory** |

The ASCII path is the lesson: replacing `char_indices()` with a per-character
`class_at()` that indexes `text[i..]` re-does a UTF-8 boundary check every
character. Faster-looking code, slower code. And the last two rows are the same
change, correctly rejected at 10 cores and correctly accepted at 64 — the
laptop was not a weaker server, it was the wrong instrument.

## Remaining candidates

1. **Interior cut rule** for boundary-free input — 2.0x, parity-critical.
2. **A cheaper routing hash.** Every occurrence is hashed (~240M per GB) and
   instrumentation puts the time there.
3. **Word-at-a-time scanning** with `u64` bit tricks — portable, removes work.
4. **Allocator flag** (mimalloc/jemalloc) — untested, plausibly real on Linux.
5. **Merge loop**: position-indexed occurrences replacing the full-word rescan.
   Now only ~17% of runtime, so the payoff shrank while the parity risk did not.

## Machine notes

Two machines: a 10-core M-series laptop (macOS, libmalloc) and rented 16/64-core
x86-64 Linux (glibc). **All laptop measurements were taken while an unrelated
training job was running** and should be treated as indicative only; everything
quoted above is from isolated cloud containers. Nothing is validated on ARM
Linux or above 64 cores.
