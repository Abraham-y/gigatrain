# Publishing: decision and plan

**Decision (2026-08-06): no paper. Ship the tool, blog it, tweet the blog.**

## Why not a paper

Two candidates were worked up and both were dropped:

**A systems paper on the trainer.** The algorithm is standard (Zouhar et al.,
already implemented three times over), gigatoken already ships an HF-parity
trainer 3.9x behind, and no NeurIPS 2026 workshop fits a data-pipeline systems
contribution. MLSys or an efficiency-track NLP venue on a later deadline is the
right home if this is ever revisited.

**A measurement-validity case study for TAE** (*Can We Trust AI Evaluation?*,
8 pages, Aug 29). Scope fit was genuine — their topic list includes benchmark
auditing, measurement and causal validity, domain coverage in benchmark design,
and evaluation stability. It was dropped because controlled measurement shrank
its two headline findings:

- "Synthetic data inverted the conclusion" — **refuted**; it was a 180 s
  timeout. Matched pairs differ 1.2x, with no consistent sign across corpora.
- "HF is 10.3x slower from thread count" — **1.34x** at the configuration the
  literature actually uses.

What survived is genuine but modest and largely n=1 self-audit: the variance
result, the censored-data bug, the unverified-environment failure, and the
audit table. Against three better-fitting papers due the same day, it was
fourth in line with the weakest case. See docs/CORRECTIONS.md for all of it.

## What to do instead

**1. ~~File the phantom-merge bug upstream.~~ Done 2026-08-07:**
[tokenizers#2320](https://github.com/huggingface/tokenizers/issues/2320),
verified against 0.23.1. The single unambiguously novel result in the project.
Still open: comment on HF PR #2066 with independent confirmation. See
docs/upstream-issues.md.

**2. Ship it.** Flip the repo public, tag `v0.1.0` so wheels build. ~30 min.

**3. Blog post.** Most of it is already written across BENCHMARKS.md,
PARITY.md, PRIOR_ART.md and CORRECTIONS.md.

Working title: *"Byte-exact BPE training, 7x faster — and the five ways I
fooled myself measuring it."* Structure:

- **Hook.** 12.9 GB, 38 s vs 257 s, same pretokenizer, 31,790 merges identical.
- **Why training is the neglected stage.** Encoding is solved; dedup is solved
  several times over; training has #1681, #1795, #1824, sentencepiece#1021 —
  all OOMs. Cite Reddy et al. correctly: they did the scaling study, at 900 GB.
- **Parity as the hard requirement.** The tie-break rule; stale heap entries
  corrected and re-pushed; `max_token_length` filtering only newly-formed
  pairs; and the good one — HF feeds files **one line at a time**, so a
  trailing `\r\n` is terminal within its line. That is exactly how gigatoken
  matches HF on LF corpora and diverges at merge #0 on CRLF ones.
- **Where the memory actually goes.** Everyone predicts the pair index; it was
  1 MB at 1 GB. Phase 1's `HashMap<String, u64>` was 1.7 GB of 2.26 GB.
- **The negative-count phantom merge.** Reachable in 8 words.
- **Five ways I fooled myself.** The strongest section, and the reason to write
  the post at all: five headline claims, five corrections, each caught by a
  check the previous step lacked, all in public git history. Lead with the
  laptop that turned out to be running someone else's training job.

**4. X thread** pointing at the blog. Lead with the byte-identical 12.9 GB row.
**Not** an unqualified "fastest": since 2026-08-07 all seven trainers are in
one comparable table (BENCHMARKS.md, "One-session comparison") and "fastest of
the seven, on web text" is supportable — but only with its caveats attached
(only the HF rows have verified identical output; rustbpe is ~15% faster on
single-giant-pretoken corpora), and a caveated claim makes a weak lead. The
byte-identical claim needs no caveat at all.

**5. Email Marcel Rød** about gigatoken's CRLF divergence. It is a real bug
report and the courteous thing to do before publishing a comparison.

## Things to leave out

- Any *unqualified* "fastest BPE trainer" claim (the caveated seven-trainer
  version is supportable but belongs in the body, not the headline).
- The 17.2x, 22.3x and ~50x figures (different pretokenizers, or unmeasured).
- Any claim of reproducing, explaining, or first-diagnosing #1313.
- The retracted vocabulary-scaling sweep, except as a cautionary anecdote.
- The 198x naive-vs-incremental benchmark — a strawman nobody ships.
