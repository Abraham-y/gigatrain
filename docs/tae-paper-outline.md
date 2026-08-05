# TAE submission outline

**Venue.** TAE, *Can We Trust AI Evaluation?* (tai-eval.github.io), NeurIPS 2026
Sydney, Dec 11–12. **8 pages** excluding references and appendices, NeurIPS 2026
template with `\usepackage[dblblindworkshop]{neurips_2026}`, double blind,
**deadline 2026-08-29 AoE**, notification 2026-09-22, non-archival, up to three
reviews.

**Why this venue.** Four of TAE's listed topics are the paper's four sections:
*benchmark and leaderboard auditing*; *measurement and causal validity*;
*domain coverage and representation* in benchmark design; *uncertainty and
robustness* regarding evaluation stability.

**Working title.** *The Baseline Is a Free Parameter: Measurement Validity
Failures in Systems Benchmarks for ML Pipelines*

---

## Thesis

A benchmark can be internally careful — stated hardware, real corpora, public
code — and still fail to support its conclusion, because a parameter it did not
control moves the *baseline* by more than the effect it reports. We demonstrate
three such failures with original measurements, show each is present in
published work, and give a detection practice for each.

The testbed is BPE tokenizer-trainer benchmarking. It is unusually suitable:
the published population is small enough to audit exhaustively (six), and
unusually, **two systems' outputs can be compared for exact equality**, so
"did these compute the same thing" is decidable rather than approximated.

---

## Section map, with the evidence for every claim

### 1. Introduction (~0.75 p)

Systems benchmarks gate adoption. Unlike model evaluation, they are assumed
robust because timing is "objective". We show three ways that fails.

Contributions:
1. A controlled measurement showing a widely-benchmarked baseline varies **10.3x
   from one unreported parameter**.
2. A case where **synthetic data inverted the conclusion** relative to real data
   of the same nominal type.
3. A quantification showing **between-allocation variance (~40%) exceeds
   within-allocation variance (±2%) by an order of magnitude**, while 0 of 6
   published benchmarks report either.
4. An exhaustive audit of the published population, with an archived,
   checksummed source set.

### 2. Setting (~0.5 p)

BPE training; why it is benchmarked; the public failure reports that motivate
the tools (tokenizers#1681/#1795/#1824, sentencepiece#1021 — all OOMs).
Evidence: `docs/audit-sources/hf-issue-*.json`, `sp-issue-1021.json`.

### 3. Failure I — an uncontrolled free parameter in the baseline (~1.75 p)

**The measurement.** One box (64-core x86-64 Linux), one binary, one corpus
(100 MB FineWeb), vocab 32k, median of 3, varying only `RAYON_NUM_THREADS`:

| threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| HF (s) | 23.8 | 18.6 | **15.4** | 16.0 | 17.9 | 29.2 | **158.6** |
| peak RSS (MB) | 764 | 781 | 827 | 888 | 930 | 1064 | 1215 |
| gigatrain (s) | 3.18 | 3.04 | 2.86 | 2.66 | 2.98 | 2.86 | 3.04 |

U-shaped; 10.3x its own optimum at 64 threads. Source:
`modal_benchmark.py::threads`, raw log retained.

**The instance in the literature.** YouTokenToMe's `benchmark.md` states its
hardware (36-core Xeon) and its own thread count ("YouTokenToMe used 4
threads"), states that SentencePiece and fastBPE are single-threaded, and never
states HuggingFace's. Its thread-scaling table scans YouTokenToMe only.
Evidence: `docs/audit-sources/yttm-benchmark.md`.

**Claim discipline.** We do *not* claim their conclusion is wrong. We claim the
number is **not attributable**: a parameter that moves the baseline 10x was
controlled for the home system and left at default, unreported, for the
baseline. State this explicitly — it is the difference between a contribution
and a hit piece.

**The same failure in our own work.** Our repository carried "19x slower with
more cores" built from two different machines (differing in ISA, OS, allocator)
until it was retracted and replaced with the controlled 10.3x. Evidence: git
history; `BENCHMARKS.md`.

**Corroboration.** SentencePiece's maintainer prototyped HF-style parallel
merging and measured 1T 6.86 s vs 4T 8.01 s, attributing the ceiling to a
sequential priority queue (~76% of training). Evidence:
`docs/audit-sources/sp-issue-366.json`, comment 2026-06-16.

**Detection practice.** Vary one variable per experiment; report the
configuration of *every* system compared, not only the one being advocated. If
a scan is run for the home system, run it for the baseline.

### 4. Failure II — synthetic data inverting the conclusion (~1.5 p)

**The measurement.** Generated corpus: uniform-random `ACGT`, no whitespace, no
newline. Every trainer timed out; we wrote this up as an *irreducible* property
of BPE, with a mechanism (both implementations rescan the whole word per merge,
verified in source).

Real corpus of the same nominal type — human chr21 (hg38, UCSC), newlines
stripped, one 46.7 MB line — **trains in 267 s**. Real bases uppercased with `N`
removed, genuinely 4 symbols in one 40 MB line: **251 s**.

**Why.** Real genomic data has 10 distinct characters (`ACGT`, soft-masked
`acgt` marking repeats, `N/n`), and **14.18% is `N`** in multi-megabyte runs
that BPE collapses immediately. Uniform-random synthesis removes exactly the
structure that makes the real thing tractable — so the synthetic corpus is the
*hardest* case, not a representative one.

**Second cause, stated plainly.** Our timeout (180 s) was also too short. Both
choices pushed the same way. Evidence: `docs/degenerate-results.md`, which
carries the withdrawal inline.

**Generalization.** "Representative of the type" is not "representative".
Synthetic data preserves the property the designer thought mattered and
silently removes the ones they did not model.

**Detection practice.** Validate against real instances before concluding;
report timeouts as censored (`>N s`) and treat threshold-sensitive results as
provisional.

### 5. Failure III — variance that is measured in the wrong place (~1 p)

Within-container spread across 3 repeats: **±2%**. Between-container spread on
identical configurations, after a cloud preemption forced a restart: **~40%**
(`dna_real` whitespace, 20.0 s then 14.3 s).

Repeats *inside* one allocation measure scheduler jitter, not reproducibility.
0 of 6 published benchmarks report timing variance of either kind; our own
reported none until 2026-08.

**Detection practice.** Repeat across allocations, not within; report the
between-allocation figure; refuse to quote two significant figures without it.

### 6. Failure IV — censored data rendered as results (~0.5 p)

Our first cloud run of the degenerate study returned 49 uniform failures
(`rc=125`: GNU `time` rejecting a BSD-only flag). The **reporting layer** then
printed every non-completion as `>900s`, i.e. as a table of timeouts. Nothing
had been measured; the output was indistinguishable from data.

**Detection practice.** Only a timeout may render as a timeout; harness faults
must be a distinct, loud status. Uniformity across a whole table is a smell.

### 7. Audit of the published population (~1.25 p)

The checklist table from `docs/literature-audit.md` (8 criteria × 7 rows,
including our own pre-August benchmark). Headline tallies: **peak memory 2/6,
output correctness 3/6, timing variance 0/6** — in a field whose canonical
failure is OOM.

Two secondary findings worth the space:
- **The most rigorous benchmark here is the least known.** ffbpe gates timings
  on configuration match and records output SHA-256 fingerprints, model parity,
  deterministic repeats and RSS — a stronger contract than ours. We scored it
  2/8 from its README and 6/8 after reading its `BENCHMARKS.md`; the correction
  is reported, as is the lesson that README-level auditing understates projects.
- **Candour and comparability are orthogonal.** The two most carefully
  caveated benchmarks are the two whose numbers cannot be reused (one states no
  corpus size in bytes; the other compares two of its own configurations).

### 8. Threats to validity (~0.5 p) — do not compress this

- Some failures are documented from **one project**, ours. Failures I and the
  audit generalize; II, III and IV are single-instance and are labelled as such.
- **The population is six**, and two publish no trainer numbers at all.
- **The author is the subject** for the self-reported failures.
- The YouTokenToMe estimate is **inference, not measurement** — we did not rerun
  their benchmark on 36 cores.
- Sources retrieved 2026-08-05 and archived; repositories change.

### 9. A checklist (~0.25 p)

The 8 criteria, phrased as things to report. This is the reusable artifact and
should be the last thing on the page.

---

## What must be done before submitting

1. **One-session comparison** *(launched 2026-08-05)* — our own repo reports
   gigatrain's 1 GB ByteLevel time as 8.5 s, 10.22 s and 14.9 s across three
   tables, because each competitor was benchmarked in a separate session. A
   paper criticising non-comparable numbers cannot ship with that. One
   container, all trainers, both sizes, 3 repeats.
2. **Anonymize.** Double blind. The repository is identifiable; the submission
   must not name it, and the archived-source manifest needs an anonymized host
   or an appendix note.
3. **COI check** on TAE organizers before investing further (Constellation /
   MATS / Redwood overlap).
4. **OpenReview profile** — creation and updates can take two weeks; act by
   ~Aug 15.

## What is deliberately not in the paper

- The trainer's speed results. This is not a systems paper and should not read
  as one; gigatrain appears only as the object being measured and as a row in
  the audit table.
- The retracted vocabulary-scaling sweep. It is a sixth failure mode
  (in-sample evaluation, non-independent seeds, a metric window measuring the
  byte alphabet) but it is *also* the one whose headline had already been
  published by Reddy et al. Including it invites "so you also failed to read
  the related work", which is true and is a different paper.
- Any claim that the field is careless. Two of six projects are more careful
  than we were.
