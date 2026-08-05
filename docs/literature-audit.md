# What published BPE-trainer benchmarks do and do not report

Compiled 2026-08-05. Every source quoted here is archived verbatim under
[audit-sources/](audit-sources/) with a SHA-256 and retrieval timestamp in
[audit-sources/MANIFEST.json](audit-sources/MANIFEST.json). Re-check with:

```bash
python scripts/fetch_audit_sources.py --verify
```

## Why

`docs/degenerate-results.md` and this repo's own history record a set of
measurement failures: single runs quoted as results, a thread scan run on the
home tool only, synthetic data standing in for real, censored data rendered as
timings. The obvious objection is that this is one project's incompetence and
generalises to nothing.

So: do the *published* benchmarks in this area have the same defects? This
audits every trainer benchmark I could find against a checklist drawn from
those failures. gigatrain's own pre-2026-08 benchmark is included as a row,
because excluding it would be exactly the kind of thing this document exists to
catch.

## The checklist

| # | Question | Why it matters |
|---|---|---|
| 1 | Peak memory reported? | The incumbent failures are OOMs (#1681, #1795, #1824), not timeouts |
| 2 | Output correctness checked? | "A speedup with different output is not a speedup" |
| 3 | Variance or repeats reported? | Between-container variance measured at ~40% here |
| 4 | Corpus size in bytes? | Row/document counts are not comparable across corpora |
| 5 | Hardware stated? | |
| 6 | Thread count controlled **for every system compared**? | HF varies 10.3x with thread count alone |
| 7 | Real (not synthetic) data? | Synthetic genomic data inverted our conclusion |
| 8 | Reproduction command public? | |

## Results

| Benchmark | 1 mem | 2 parity | 3 var | 4 bytes | 5 hw | 6 threads | 7 real | 8 repro |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YouTokenToMe `benchmark.md` | ✗ | ✗ | ✗ | ✓ | ✓ | **✗** | ✓ | ✓ |
| fast-bytelevel-bpe-go | ✗ | **✓** | ✗ | **✗** | ~ | ✗ | ✓ | ~ |
| **ffbpe** | **✓** | **✓** | ~ | ✓ | ✗ | ✗ | ✓ | ✓ |
| gigatoken | — | — | — | — | — | — | — | — |
| rustbpe | — | — | — | — | — | — | — | — |
| SentencePiece (taku910, sp#366) | ✗ | n/a | ✗ | ✓ | ✗ | ~ | ? | ✗ |
| **gigatrain, before 2026-08** | ✓ | ✓ | **✗** | ✓ | ✓ | **✗** | **~** | ✓ |

✓ yes · ✗ no · ~ partial · — no published trainer benchmark exists · ? unstated

**gigatoken and rustbpe publish no trainer benchmark at all.** gigatoken's
README benchmarks are encoding throughput (24.53 GB/s on 11.9 GB, 144 cores);
its BPE trainer is not mentioned. rustbpe's README claims only "fast training
with parallel processing (rayon)" with no numbers.

## Finding 1: the most rigorous benchmark here is the least known

An earlier draft of this audit scored ffbpe from its README and concluded that
no project reported memory and correctness together. Reading the project's
actual `BENCHMARKS.md` overturned that, and it is worth recording that the
correction ran in the generous direction.

ffbpe separates correctness from timing by policy:

> "FFBPE keeps correctness gates and timing measurements separate. Timing
> results are informational unless the input, model, configuration,
> environment, and output fingerprints all match."

and its regression suites record "token counts and SHA-256 fingerprints", "model
vocabulary and merge fingerprints", "deterministic repeats", "exact versus
bounded-memory model parity", and "timing and available RSS measurements".
That is a stronger contract than anything else in this table, including
gigatrain's.

With that correction, the tallies are: **peak memory 2 of 6** (ffbpe,
gigatrain), **output correctness 3 of 6** (ffbpe, fast-bytelevel-bpe-go,
gigatrain), **timing variance 0 of 6.**

The variance column deserves care. ffbpe checks that repeats produce *identical
output*, which none of the others do — but its published timing tables are
still explicitly single runs ("One release run used a 1 GiB FineWeb2 Chinese
inventory..."). So determinism of output is checked; variance of timing is not
reported by anyone, including this repo until 2026-08.

The memory column matters most, because the failure mode the field actually
hits *is* memory. The issues motivating every one of these projects —
tokenizers#1681, #1795, #1824, sentencepiece#1021 — are OOMs. Four of six
benchmarks do not measure the axis that is failing.

## Finding 2: the thread-count defect is in the field's most-cited benchmark

This is the one that generalises, and it is the same defect gigatrain had.

YouTokenToMe's `benchmark.md` states its hardware ("36-core Intel(R) Xeon(R)
Platinum 8124M... 256GB RAM") and states its own thread count:

> "In this benchmark, `YouTokenToMe` used 4 threads for training and
> tokenization. `SentencePiece` doesn't support multithreading for **BPE** at
> all. `fastBPE` doesn't support multithreading for training."

**HuggingFace's thread count is never stated.** With rayon's default it would
use all 36 cores. Their thread-scaling table then scans 1→16 threads for
YouTokenToMe *only*.

That is decisive given this repo's controlled sweep (one box, one binary, one
corpus, only `RAYON_NUM_THREADS` varied), which found HF's runtime is U-shaped:
15.4 s at 4 threads, 29.2 s at 32, 158.6 s at 64. HF at ~36 threads sits well
past its optimum — plausibly around 2x — so a meaningful part of YouTokenToMe's
reported "Hugging_Face_BPE ... 97.7 (x3.8)" at 1 GB English is likely measuring
HF's thread pathology rather than YouTokenToMe's speed.

I am **not** claiming their conclusion is wrong; YouTokenToMe is genuinely fast
and the ratio is large enough to survive a 2x correction. The claim is narrower
and harder to dispute: **the published number is not attributable, because a
free parameter that moves the baseline by up to 10x was left at its default and
unreported for the baseline while being carefully controlled for the home tool.**

gigatrain did exactly the same thing: `modal_benchmark.py` scanned threads for
gigatrain only, and BENCHMARKS.md carried a "19x slower with more cores" claim
built from two different machines until it was retracted and re-measured.

## Finding 3: independent corroboration from a competing maintainer

SentencePiece's maintainer prototyped HF-style parallel merging and measured it
(sentencepiece#366, 2026-06-16, on "a 19.3MB corpus (~180k unique words, vocab
32,000)"):

> "parallelizing BPE training does not provide a meaningful speedup and can
> cause slowdowns"
> · 1T baseline 6.86 s · 4T (threshold 1000) 6.30 s · 4T (threshold 10) **8.01 s**

> "**Amdahl's Law Constraint**: The parallelizable merge step only accounts for
> **~24%** of BPE training time... The remaining **76%** (priority queue
> updates) is inherently sequential, limiting the maximum theoretical speedup to
> ≈1.3x."

> "**HF-style**: Replicates HF's `Rayon` strategy of parallelizing a single
> pair's merge. It shares the same sequential PQ bottleneck."

Two things follow. First, an independent party with no stake in gigatrain
measured HF-style parallel merging as ineffective-to-harmful, which corroborates
the controlled sweep from a different direction. Second, his own measurement has
no memory figure, no repeats, no hardware statement — the same gaps as everyone
else's.

(For the record, this also corrects an earlier misreading in PRIOR_ART.md: the
phrase "compact arrays, custom hash maps, split queues" is taku910 describing
**YouTokenToMe's** architecture, not prescribing gigatrain's. The archived
comment shows the sentence in context.)

## Finding 4: disclaiming limitations is independent of being comparable

Two projects are notably careful. fast-bytelevel-bpe-go:

> "These numbers are from one local run and should be treated as a reference
> point, not a universal guarantee."
> "It should not be read as 'Go is faster than Rust'"

and it checks output correctness (`vocab: SAME / merges: SAME` at vocab
32,779). ffbpe warns that its Unicode-bigram result "changes corpus
segmentation and should be benchmarked on representative text rather than
treated as a universal speedup", notes that "peak RSS is process-level and
environment-dependent", and states plainly of its headline comparison that "it
is not a model-parity claim."

Yet neither result can be placed beside anything else. fast-bytelevel-bpe-go
describes its corpus as "the first 846,882 non-empty `text` rows" of a JSONL
file — **no size in bytes appears anywhere** — so its 739.59 s against HF's
4406.97 s is uncomparable to every other row in this table. ffbpe's headline
compares two of its *own* pipeline configurations, not two trainers, and states
so.

The lesson is that candour and comparability are orthogonal. The two most
carefully-caveated benchmarks in this area are also the two whose numbers
cannot be reused, while the least-caveated (YouTokenToMe) is the one whose
numbers are most often quoted.

## What this establishes for the case study

The failure modes documented in this repo are **not idiosyncratic**:

- *Uncontrolled comparison* (a free parameter left at default for the baseline
  and controlled for the home tool) — present in YouTokenToMe's benchmark, and
  in gigatrain's until 2026-08.
- *Missing memory* — present in four of six, in a field whose canonical failure
  is OOM.
- *No timing variance* — present in six of six, including this repo.
- *Non-comparable corpus description* — present in fast-bytelevel-bpe-go.

The remaining failures in the taxonomy (synthetic data inverting a conclusion,
censored data rendered as results, in-sample evaluation) are documented from
gigatrain's own history only, and should be presented as such rather than as
field-wide claims.

## Limitations of this audit

- **Six benchmarks is the population, not a sample.** These are all the
  published BPE-trainer benchmarks I could find; two of the six turned out to
  publish no trainer numbers at all. Small-n conclusions follow.
- **The YouTokenToMe estimate is inference, not measurement.** I did not rerun
  their benchmark on 36 cores. The claim that HF sits ~2x past optimum there is
  extrapolated from a sweep on a different machine, and is offered as "not
  attributable", not as a corrected figure.
- **Retrieved 2026-08-05.** Repositories change; the archive and manifest exist
  so the quotes stay checkable.
- **Scoring from READMEs understates projects.** ffbpe's row moved from 2/8 to
  6/8 once its `BENCHMARKS.md` was read rather than its README. Other rows may
  be similarly understated where a project documents its methodology somewhere
  I did not look — in particular, only the surfaces listed in the manifest were
  read, not entire repositories or CI configurations.
