# Benchmarks

Harnesses: `scripts/benchmark.py` (gigatrain vs HF, reports wall time, peak
RSS and merge-list parity together), `scripts/modal_benchmark.py` (all
trainers on a rented many-core Linux box), and per-trainer CLIs under
`scripts/` for rustbpe, SentencePiece, gigatoken and ffbpe. Only
`benchmark.py` verifies parity; the multi-trainer runs discard stdout and
measure time and memory only.

**A speedup with different output is not a speedup.** Where a row compares
different pretokenizers, or a trainer that does not produce an HF-compatible
merge list, it is labelled as such and is a time/memory comparison only.
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
| 100 MB | gigatrain | 1.2 s | 224 MB | — | IDENTICAL |
| 1 GB | gigatrain | 8.5 s | 725 MB | — | IDENTICAL |
| 12.9 GB | gigatrain | **85 s** | **2.2 GB** | — | — |

No HF ByteLevel figure was ever measured on this laptop, so the speedup column
is empty. (A previous version of this table carried a "~50x" against an HF
ByteLevel time of 61.2 s at 100 MB. That number was the 1 GB *whitespace* row
copied one table up; it was never measured. The only real HF ByteLevel timing
in this repo is 257.1 s at 12.9 GB / 16 cores — see
[Parity verified at 12.9 GB](#parity-verified-at-129-gb) — where HF's ByteLevel
run is ~1.9x *faster* than its whitespace run, which is the opposite of what
the retracted 100 MB row implied.)

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

Together ~2.4x on phase 1 **at 100 MB**. The effect is much smaller at scale:
at 12.9 GB phase 1 went 88.9 s → 70.6 s (**1.26x**), taking the whole run from
104 s to **85 s** and peak RSS from 2.4 GB to 2.2 GB. The 100 MB run is
dominated by the per-character work these two changes remove; the 12.9 GB run
is dominated by I/O and hash-map inserts, which they do not touch.

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

All 10-core macOS, whitespace for gigatrain and HF:

| corpus | SentencePiece v0.2.2 | HF 0.22.2 | gigatrain |
|---|---|---|---|
| 100 MB | 13.7 s / 539 MB | 9.7 s / 1.0 GB | 1.7 s / 419 MB |
| 1 GB | 112.7 s / 3.0 GB | 61.2 s / 4.7 GB | 9.4 s / 1.3 GB |

SentencePiece measured 112.7 s at 1 GB on *both* this laptop and the 64-core
Linux box. Its BPE trainer is single-threaded, so both machines do the same
one-core work; peak RSS differs (3.0 vs 3.6 GB), confirming these are separate
runs rather than a transcription error.

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

## Parity verified at 12.9 GB

`modal run scripts/modal_benchmark.py::parity --size-mb 13000` trains both
trainers on the same 12.9 GB corpus keeping their merge lists, and diffs
them. Vocab 32000, 16 cores (deliberately not 64 — HF degrades with core count,
and these runs need it to finish correctly rather than fast, which also makes
them the least favourable comparison for gigatrain):

| pretokenizer | merges | gigatrain | HF | identical |
|---|---|---|---|---|
| ByteLevel | 31,790 | 38.0 s | 257.1 s | yes |
| whitespace | 16,969 | 107.8 s | 496.9 s | yes |

Note ByteLevel is both *faster* and produces *more* merges: it yields far
fewer unique pretokens (9.0M vs 27.4M), so phase 2 does less work, and the
32k vocabulary is reached with more merges because its alphabet is 256 bytes
rather than every character observed.

Until this run, nothing above 1 GB had been diffed: the benchmark harnesses
send trainer stdout to `/dev/null`, so the 12.9 GB headline asserted a parity
result that had never been computed. It has now.

64-core x86-64 Linux, 192 GiB, glibc, vocab 32000. This is the run the laptop
could not do: with enough RAM, HuggingFace and SentencePiece are no longer
swap-bound, so these are real timings rather than "it thrashed".

| trainer | pretokenizer | wall | peak RSS | outcome |
|---|---|---|---|---|
| **gigatrain** | ByteLevel | **43.8 s** | 2.7 GB | ok |
| **gigatrain** | whitespace | **129.4 s** | 6.7 GB | ok |
| SentencePiece v0.2.2 | its own | 135.0 s | 20.0 GB | **SIGSEGV** |
| HuggingFace | whitespace | 754.9 s | 29.8 GB | ok |
| rustbpe | GPT-4 regex | 975.4 s | 5.3 GB | ok |

**Like-for-like is whitespace vs whitespace: 129.4 s against 754.9 s, 5.8x**,
with identical output. The 43.8 s ByteLevel figure is 17.2x HF's wall time but
produces a different tokenizer, so it is not a speedup number. No HF ByteLevel
figure exists at this size because `modal_benchmark.py:159` does not pass
`--pretokenizer` to the HF CLI, which defaults to `WhitespaceSplit` — the CLI
itself supports `bytelevel` (`hf_train_cli.py:19`) and it is used for the
12.9 GB parity runs below. Running it here is a one-line change and would give
the missing like-for-like ByteLevel row.

Three things worth stating plainly.

**HuggingFace does finish, given 192 GiB** — in 12.6 minutes, using 29.8 GB of
resident memory. That is 4.4x gigatrain's whitespace mode (6.7 GB, the
like-for-like comparison) and 11x its ByteLevel mode (2.7 GB). On a 34 GB
laptop the same job never completed in an hour. So the honest claim is not
"HF cannot do this"; it is that HF needs several times the memory and, like
for like, 5.8x the time.

**SentencePiece crashed.** It ran 135 s and died with SIGSEGV (rc=139) at
20 GB resident, having been given 192 GiB. This matches the segfault reported
in sentencepiece#862 on a 98 GB corpus. It is a crash, not a timeout.

**rustbpe is the memory winner.** At 5.3 GB it uses *less* than gigatrain's
whitespace mode, though about twice gigatrain's ByteLevel mode — while taking
22x longer. If memory were the only axis it would be a genuine contender.

Note ByteLevel is 3x faster than whitespace *on this box*, because it yields
far fewer unique pretokens (9.0M vs 27.4M) and phase 2 scales with that. The
gap is machine-dependent, not scale-dependent: on the same 12.9 GB file on the
10-core laptop, ByteLevel is 85.4 s and whitespace 85.7 s — a 1.0x gap. An
earlier version of this line claimed the gap "widens with scale", which the
laptop's own numbers contradict.

## 19.4 GB on a "normal machine" — milestone 5, and issue #1681's size

CLAUDE.md milestone 5 sets the target *"20 GB corpus trains without OOM on a
normal machine, directly answering issue #1681"*. Until now the largest run in
this repo was 12.9 GB, so the target had never been tested.

`modal run scripts/modal_benchmark.py::main --sizes 20000 --cpu 16 --memory 64`
on **x86-64 Linux, 16 cores, 64 GiB**, vocab 32000. The deliberately modest box
is the point: #1681 is about OOM, so a 192 GiB machine would not answer it.

| trainer | pretokenizer | wall | peak RSS | outcome |
|---|---|---|---|---|
| **gigatrain** | ByteLevel | **47.3 s** | **2.9 GB** | ok |
| **gigatrain** | whitespace | **137.4 s** | 7.2 GB | ok |
| SentencePiece v0.2.2 | its own | 158.3 s | 27.2 GB | **SIGSEGV** |
| HuggingFace 0.22.2 | whitespace | 730.9 s | 36.3 GB | ok |
| rustbpe | GPT-4 regex | 1216.7 s | 5.8 GB | ok |

**Like-for-like is whitespace vs whitespace: 137.4 s against 730.9 s (5.3x),
on 7.2 GB against 36.3 GB (5.0x).** That 5.3x sits alongside the 5.8x measured
at 12.9 GB on 64 cores, so the advantage is stable across both scale and core
count.

SentencePiece segfaulted again at 158 s and 27.2 GB resident, reproducing its
12.9 GB behaviour on a different machine, core count and corpus size — so that
crash is a property of the input scale rather than a one-off. rustbpe finished
but took 20 minutes, and is again the memory winner among the baselines at
5.8 GB — less than gigatrain's whitespace mode, more than twice its ByteLevel
mode.

The memory line is the one that answers #1681. HuggingFace needed **36.3 GB of
RAM for 19.4 GB of text — 1.9x the corpus** — which is why a 20 GB corpus OOMs
on machines that look like they should cope, and it would not have fit on a
32 GB box. gigatrain's ByteLevel mode used 2.9 GB, **0.15x the corpus and 12.5x
less than HF**.

**The corpus is 19.4 GB, not 20 GB.** `_prepare_corpora` assumed 4-5 GB of text
per FineWeb parquet; the measured yield is ~3.2 GB, so six parquets gave
19,370 MB and the script warned rather than failing. The estimate is now fixed
(3 GB per parquet plus one), but the numbers above are from the 19.4 GB file
and are reported at that size.

## Validation on 64-core Linux (Modal)

Run with `modal run scripts/modal_benchmark.py --sizes 100,1000 --cpu 64`.
x86-64 Linux, 64 cores, 192 GiB, glibc. Wall time / peak RSS, vocab 32000.

| corpus | gigatrain (ByteLevel) | rustbpe | SentencePiece | HF |
|---|---|---|---|---|
| 100 MB | **2.6 s / 608 MB** | 10.4 s (3.9x) | 15.5 s (5.9x) | 181.0 s (68.8x) |
| 1 GB | **7.4 s / 1570 MB** | 88.2 s (11.9x) | 112.7 s (15.2x) | 244.4 s (33.1x) |

Two things this settles.

**The advantage is not an Apple Silicon artifact.** The ratios hold or widen
on x86-64 Linux: rustbpe 9.5x -> 11.9x at 1 GB, SentencePiece 13x -> 15.2x.

**HuggingFace degrades with core count — now measured under control.**

The claim used to rest on 9.7 s (10-core macOS) against 181 s (64-core Linux),
which varies ISA, OS, allocator and machine as well as core count. That was
retracted as uncontrolled. `modal_benchmark.py::threads` runs the experiment
properly: **one box, one binary, one corpus, varying only `RAYON_NUM_THREADS`**
(64-core x86-64 Linux, 100 MB FineWeb, vocab 32000, median of 3):

| threads | HF | peak RSS | gigatrain |
|---|---|---|---|
| 1 | 23.8 s | 764 MB | 3.18 s |
| 2 | 18.6 s | 781 MB | 3.04 s |
| 4 | **15.4 s** | 827 MB | 2.86 s |
| 8 | 16.0 s | 888 MB | 2.66 s |
| 16 | 17.9 s | 930 MB | 2.98 s |
| 32 | 29.2 s | 1064 MB | 2.86 s |
| 64 | **158.6 s** | 1215 MB | 3.04 s |

HF is U-shaped with a minimum at 4 threads. At 64 it is **10.3x slower than its
own optimum** and 6.7x slower than single-threaded, while peak RSS rises
monotonically 764 MB -> 1215 MB. gigatrain is flat within noise.

So the effect is real and the mechanism (rayon-parallel pair counting reducing
per-thread hash maps) is visible in the source — but **the honest magnitude is
10.3x against its own optimum, not the "19x" this file once claimed.** The
retracted number was inflated by the confounds.

gigatrain being flat here is not evidence of good scaling: at 100 MB the
sequential phase 2 dominates, so phase-1 parallelism has little to show. The
scaling result for gigatrain is the 1 GB thread scan further down.

This is also **not** a reproduction of issue #1313 — see
[the retraction above](#a-retraction-this-is-not-hf-issue-1313). #1313 is a
`vocab_size=512` run on unsegmented DNA-like data, and its maintainer
diagnosed it in-thread as degenerate pretokenization. The anti-scaling
measured here is a separate phenomenon that happens to point the same
direction.

### Thread scaling, and a real bug it exposed

Sizing scanners and owners each at `nthreads` put ~2x the core count on the
CPU. On 10 cores that was inside the noise — an earlier attempt to fix it was
reverted for lack of evidence. On 64 cores it is unmistakable:

| threads | before (scanners=owners=N) | after (split budget) |
|---|---|---|
| 16 | 5.71 s / 604 MB | 5.3 s / 505 MB |
| 32 | 5.51 s / 729 MB | 4.8 s / 534 MB |
| 48 | 6.31 s / 958 MB | 4.7 s / 567 MB |
| 64 | 7.16 s / 1384 MB | **4.9 s / 638 MB** |
| 96 | 8.25 s / 1804 MB | 5.2 s / 760 MB |

Before, throughput peaked at 32 threads and got *worse* with more cores, with
peak RSS climbing to 1.8 GB. After splitting the budget between the two pools,
the curve is flat from 32 to 96. At the default on a 64-core box that is
**1.46x faster and 2.2x less memory**.

`--threads N` now means N workers total, split between the pools, rather than
N of each.

## Remaining optimization candidates

Phase 1 is now ~83% of the ByteLevel runtime (70.6 s of 85 s at 12.9 GB), so
that is where the remaining work is. A sampling profile shows scanner and
owner threads mostly *blocked on their channels* rather than computing, which
means the pipeline is limited by hand-offs or by I/O, not by the split/hash
inner loops that were just optimized.

Tried and rejected: retuning the thread pools. Sizing readers/scanners/owners
each at `nthreads` spawns ~3x the core count, and one measurement suggested
4 threads beat 10. Under a repeated A/B (4 alternating rounds) the difference
was 4.94 s vs 4.87 s — inside the noise — and halving the scanner/owner pools
was clearly worse. Reverted rather than landing complexity for a
non-effect.

### Four optimizations tried and rejected

Phase 1 was instrumented per stage (`GIGATRAIN_STATS=1` now reports summed
CPU-ms for read / scan+hash / send-blocked / recv-blocked / insert). On 1 GB
ByteLevel, 10 threads: **scan+hash 7589 ms across 5 scanners, insert 5538 ms
across 5 owners, and owners blocked 2118 ms waiting on recv**. Scanners are
the critical path — their per-thread ~1518 ms is essentially the 1.63 s wall.

Four changes were implemented against that reading and measured on a clean
64-core box. The first three were reverted; the fourth was rejected at 10
cores, then re-measured at 64 and landed (it is the thread-budget split
tabulated above):

| change | rationale | measured (1 GB, 64 threads) |
|---|---|---|
| 60/40 scanner/owner split | CPU split is ~58/42, so balance the pools | 4.7 s vs 4.7 s — no effect |
| byte-table ASCII path in `next_piece_len` | avoid decoding chars to classify them | **10% slower**; 1 thread 8.4 s -> 12.1 s |
| zero-copy batching (spans into the shared chunk) | removes one memcpy pass over the corpus | 5.4 s vs 4.8 s — slightly worse |
| (earlier) thread-pool retuning at 10 cores | apparent oversubscription | inside noise; later confirmed and landed at 64 cores |

The ASCII path is instructive: replacing `char_indices()` with a per-character
`class_at()` that indexes `text[i..]` re-does a char-boundary check on every
character, which costs more than the iterator it replaced. Faster-looking code
was slower code.

The conclusion from the instrumentation is that phase 1 is bound by the
hashing and hash-map inserts themselves, not by copying, classification, or
pool balance. Reducing it further means changing what work is done — a cheaper
hash, or not hashing every occurrence — not doing the same work more
efficiently.

Still untested, roughly by expected payoff:

1. **A cheaper hash for routing.** Every token occurrence is hashed
   (~240M times per GB). FxHash over short slices is already cheap, but the
   instrumentation says this is where the time is, so it is the first thing to
   attack.
2. **Word-at-a-time scanning.** Processing 8 bytes per step with `u64` bit
   tricks is portable (no intrinsics, identical on ARM and x86). Unlike the
   rejected ASCII path, this removes work rather than reorganising it.
3. **Allocator choice.** A mimalloc/jemalloc feature flag may change peak RSS
   materially, especially on Linux.
4. **Merge loop**: CLAUDE.md's linked-list with position-indexed occurrences,
   replacing the full-word rescan. Now only ~17% of runtime, so the payoff
   shrank while the parity risk did not.

**Measuring these needs a quiet machine.** After the HF and SentencePiece runs
left 14-28 GB of swap occupied, repeated runs of an identical configuration
varied by up to 2x — larger than the effects being chased. The two wins above
were only trustworthy because they were A/B'd with alternating rounds and had
effect sizes (1.4x, 1.7x) well clear of that noise.

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
