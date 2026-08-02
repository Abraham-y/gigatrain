# Publication plan: X thread, blog post, and the paper this unblocks

Everything below is drafted against measurements recorded in this repo. Any
number not in BENCHMARKS.md, PARITY.md or PRIOR_ART.md should be cut rather
than estimated — the project has already had to retract one claim that was
asserted rather than computed, and a second retraction would cost more than
any headline is worth.

Publish order: flip repo public → tag `v0.1.0` so wheels exist → email Marcel
Rød → blog post → X thread pointing at the blog.

---

## Part 1 — The X thread

### The framing decision

**Do not lead with "fastest BPE trainer."** It invites an immediate and
correct rebuttal: gigatoken's trainer is only 1.8x behind, so "fastest" is
true but thin, and being corrected in the replies costs more than the claim
gains.

**Lead with the scaling finding.** It is surprising, reproducible, useful to
people who will never install this, and there is no counter-argument to it —
it is a measurement of their code, not a claim about mine.

### Draft thread

> **1/**
> HuggingFace's BPE trainer gets ~19x *slower* when you give it more cores.
>
> Same 100 MB corpus, same version, same machine class:
> 10 cores → 9.7 s
> 64 cores → 181 s
>
> At 1 GB: 61 s → 244 s.
>
> **2/**
> This is why tokenizers#1313 — 13 GB on 256 threads, unfinished after 10
> hours — was never explained. It was closed as stale in 2023.
>
> The cause is visible in the source: rayon-parallel pair counting reduces
> per-thread hash maps, so more threads means more merging work, not less.
>
> **3/**
> I found this while building gigatrain, a BPE trainer with byte-exact
> HuggingFace parity.
>
> 12.9 GB of FineWeb → 32k vocab:
> · gigatrain 38 s / 2.7 GB
> · HuggingFace 257 s / 29.8 GB
> · SentencePiece: segfaults
>
> Merge lists verified identical.
>
> **4/**
> The memory result surprised me more than the speed.
>
> Everyone assumes the pair index is the hazard. Profiled at 1 GB it was
> 1 MB. The actual hog was phase 1's HashMap<String, u64> — 1.7 GB of a
> 2.26 GB peak, from per-word allocations and per-worker duplication.
>
> **5/**
> Also worth saying plainly: the algorithm is not new. Incremental counting
> with an inverted index and a lazy heap is Sennrich 2015, formalized by
> Zouhar 2023, and already in tokenizers, SentencePiece and rustbpe.
>
> The contribution is memory layout and a parallel phase 1.
>
> **6/**
> And gigatoken already ships a BPE trainer with HF tie-breaking. It's
> undocumented and validated at ~120 KB, but it exists, and it's the closest
> competitor — 1.8x, not an order of magnitude.
>
> Full prior-art writeup in the repo, findings that undercut the project
> first.
>
> **7/**
> Probably the most reusable artifact isn't the trainer. It's PARITY.md —
> a specification of HuggingFace's exact training semantics: the tie-break
> rule, stale-heap handling, an i32 overflow, and which behaviours are
> nondeterministic.
>
> That document doesn't exist anywhere else, including HF's own docs.
>
> **8/**
> Then I pointed four adversarial agents at it. They found seven real bugs
> and ~fifteen overstated claims in three hours — including an invariant I
> had written a confident comment about that was simply false.
>
> All fixed. Blog post with the details: [link]

### Notes for the thread

- Tweet 3 must use the **16-core** numbers (38 s vs 257 s) because those are
  the runs where merge lists were actually diffed. The 64-core numbers are
  faster for us but HF's figure there is a whitespace run, and mixing them is
  the exact error already retracted once.
- Expect these replies and have answers ready:
  - *"What about gigatoken?"* → covered in tweet 6, link PRIOR_ART.md.
  - *"You just implemented the standard algorithm."* → yes, tweet 5 says so.
  - *"Did you use ByteLevel for yours and whitespace for theirs?"* → no, both
    rows are like-for-like; the mismatched comparison is documented as a
    corrected error.
  - *"Is HF's slowdown just your machine?"* → measured on Modal, x86-64
    Linux, reproducible with one command in the repo.
- Do **not** claim to have reproduced #1313. It used `vocab_size=512` on
  unsegmented data. The repo retracts that; the thread must match.

---

## Part 2 — The blog post

Working title: **"HuggingFace's tokenizer trainer gets slower when you add
cores"** — or, if you prefer the build story, **"What I learned making BPE
training 6x faster, including the five optimizations that didn't work."**

Target length 2,500–3,500 words. The value is the reasoning and the negative
results, not the benchmark table.

### Structure

**1. The hook (300 words)**
The anti-scaling measurement, stated plainly with the numbers. Note that
published HF figures for 1 GB rise with core count across sources — but be
careful: two of those three sources don't state their core count, so present
it as consistent-with, not as triangulation. The repo overstated this once.

**2. Why tokenizer training is the neglected stage (400 words)**
Encoding is solved (gigatoken, ~24 GB/s). Dedup is solved several times over.
Training has HF #1681 (20 GB OOM on 1.5–2 TB), #1795 (100 GB of RAM for
~1.5 GB of Chinese JSONL), sentencepiece#1021 (31.2 GB → 1.8 TB, unfinished
at 24 h). The universal workaround is sampling down.

Cite Reddy et al. **correctly**: they trained BPE, UnigramLM and WordPiece
from 1 GB to 900 GB and found diminishing returns beyond ~150 GB. They used
HF for UnigramLM and WordPiece and built their BPE trainer on minbpe. Do not
repeat the "90% overlap" claim — it is not in the paper, and the repo already
corrected it.

**3. Parity as the hard requirement (500 words)**
This is the part most readers won't expect. A faster trainer nobody can
verify is a demo. Getting byte-exact parity meant reading HF's source and
reproducing behaviours that look like bugs:
- the tie-break rule (higher count, then *smaller* ID pair)
- stale heap entries corrected and re-pushed, not discarded
- `max_token_length` filters only newly-formed pairs, never the initial count
- HF's trainer feeds files **one line at a time**, so a trailing `\r\n` is
  terminal within its line and becomes one token — undocumented, and it
  silently changes output
That last one is a good story: it's how gigatoken's trainer diverges on CRLF
corpora while matching exactly on LF ones.

**4. Where the memory actually goes (600 words)**
The best technical section. Everyone predicts the pair index; it was 1 MB at
1 GB. Phase 1's `HashMap<String, u64>` was 1.7 GB of 2.26 GB.

Then the three fixes and what each bought:
- arena-backed counter with disjoint hash shards: 1.7 GB → 422 MB
- dropping word strings once tokenized
- flat symbol arena instead of a `Vec` per word (removes 4.1M allocations)

And the design that was rejected: broadcasting chunks to every shard owner
fixes memory but replicates scanning per worker — 4.8 s vs 1.3 s, with a
fixed Amdahl floor that worsens as cores increase.

**5. Five optimizations that didn't work (600 words)**
The section most worth writing, because nobody publishes it.

| change | why it should have worked | result |
|---|---|---|
| 60/40 scanner/owner split | CPU split measured ~58/42 | no effect |
| byte-table ASCII path in piece scanning | stop decoding chars to classify | **10% slower** |
| zero-copy batching | removes a memcpy pass over the corpus | slightly worse |
| thread-pool retune at 10 cores | apparent oversubscription | inside noise |
| (the same retune at 64 cores) | — | **worked: 1.46x, 2.2x less memory** |

The ASCII path is the lesson: replacing `char_indices()` with a per-character
`class_at()` that indexes `text[i..]` re-does a UTF-8 boundary check every
character. Faster-looking code, slower code.

And the meta-lesson: the same change was correctly rejected at 10 cores and
correctly accepted at 64. The laptop wasn't a weaker version of the server;
it was the wrong instrument.

**6. The adversarial audit (600 words)**
Four agents with distinct lenses — parity attacker, unsafe-code auditor,
concurrency attacker, claims fact-checker — found seven real bugs and about
fifteen overstated claims in three hours.

The three worth describing:
- **An invariant I asserted and never verified.** A per-token-ID length table,
  documented as "always exactly HF's per-symbol length." False once
  `continuing_subword_prefix` is set, because one ID becomes reachable both
  as an initial symbol and as a merged token. Silent wrong output.
- **`process::exit(2)` in library code**, which killed the caller's Python
  interpreter on a missing file — no exception, no `finally`, no flush.
- **A claim that was never computed.** The README said the 12.9 GB run
  produced a byte-identical merge list. The harness discarded stdout. That is
  the same criticism the repo levels at gigatoken's 120 KB parity test. It has
  since been computed: identical, both modes.

Honest closing note: I wrote the code, the tests, and the comments asserting
correctness. I did not find these. A reviewer would have found the one-word
`bccaa` counterexample in an afternoon.

**7. What's actually novel, and what isn't (300 words)**
Not novel: the algorithm; being first to HF-parity training (gigatoken).
Novel: the phase-1 sharded shuffle, the parity specification, the scaling
finding, and a benchmark harness that reports wall time, peak RSS *and* merge
parity together — no existing trainer benchmark reports memory at all, and
the only one that ever existed was archived in 2024.

**8. What this unblocks (200 words)**
Lead into the paper. See Part 3.

### Things the post must include for credibility

- The reproduction command (`modal run scripts/modal_benchmark.py`)
- Hardware for every number: 10-core M-series macOS, or 64-core x86-64 Linux
- Both known divergences: HF's i32 overflow past ~2^31 pair occurrences
  (~120–150 GB of English text), and the decorated modes where HF is usually
  nondeterministic and gigatrain differs even where it isn't
- The input-dependent memory caveat: a whitespace-free corpus peaks at ~4.5x
  its size — the same failure shape criticised in HF #1681
- gigatoken, prominently, not buried

### Things to leave out

- The 17.2x and 22.3x figures (different pretokenizers)
- Any claim of reproducing #1313
- "Three unrelated sources" for the scaling triangulation
- Anything about the 198x naive-vs-incremental Python benchmark — it is a
  strawman nobody ships

---

## Part 3 — The paper this makes possible

**The tool is not the paper. The tool is the instrument.**

A systems paper on gigatrain would be rejected, and correctly: the algorithm
is Zouhar et al. 2023, already implemented three times over; the engineering
is good but not a scientific claim; and the scaling finding is a
well-characterised bug report about one library.

But the reason the tool was worth building is a real open question.

### The question

**How much does training-corpus size and composition actually matter for a
tokenizer, and where does it stop mattering?**

Nobody has answered this properly because the experiment was unaffordable.
Training twenty vocabularies on a terabyte meant weeks of compute, so the
field trains one vocabulary on a sample and moves on. Reddy et al.
([arXiv:2502.20273](https://arxiv.org/abs/2502.20273)) got closest — 1 GB to
900 GB, three algorithms — and had to write their own BPE trainer to do it.

At 12.9 GB in 38 seconds, a sweep that was a cluster job becomes a laptop
afternoon.

### What the study looks like

A grid over four axes, which is now affordable:

1. **Corpus size**: 100 MB → 1 GB → 10 GB → 100 GB → 1 TB (log steps)
2. **Vocabulary size**: 8k, 16k, 32k, 64k, 128k, 256k
3. **Composition**: English-only, multilingual, code-heavy, domain-specific
   (biomedical, legal), and deliberately degenerate (DNA, logs) — the last
   matters because that is where every incumbent trainer falls over
4. **Algorithm**: BPE vs WordPiece (both supported), ideally Unigram later

Measured against: fertility (tokens per word), compression rate, vocabulary
overlap with the largest-corpus tokenizer, per-language equity in the
multilingual arm, and — the expensive but decisive one — downstream loss for
a small model trained on a fixed budget with each tokenizer.

### The claims a paper could make

- **A saturation curve.** Where does more data stop changing the vocabulary?
  Reddy reports ~150 GB for English; nobody has the multilingual or code
  answer. A curve per domain would be genuinely new.
- **The composition-vs-size trade.** Is 10 GB of well-mixed data better than
  100 GB of English? Practitioners guess at this constantly.
- **Whether vocabulary differences survive to downstream loss.** Much of the
  tokenizer literature stops at intrinsic metrics because training models is
  expensive. Even a small-scale answer would be cited.
- **A reproducible benchmark.** No leaderboard for tokenizer *trainers*
  exists; the only suite was archived in 2024, reports no memory, and stops
  at 1 GB.

### Why it's credible now

- The trainer is byte-exact against the reference implementation at 12.9 GB,
  so results can't be dismissed as an artifact of a custom trainer — the
  precise criticism you could level at a minbpe-derived one.
- Both BPE and WordPiece are supported, so the algorithm axis is real.
- The harness already reports time, memory and parity together and runs on
  rented many-core Linux, so the sweep is a script, not an engineering
  project.

### Honest risks

- **It's an empirical study, not a method.** Venue matters: this is a
  resource/benchmark track or a workshop, not a main-conference method paper.
- **The downstream-loss arm is the expensive part** and is what separates
  "interesting" from "citable." Budget for it or scope it explicitly.
- **Reddy et al. have a head start** on the English-size axis. Differentiate
  on multilingual, code, degenerate domains, and the downstream arm.
- **Tokenizer research has a modest audience.** The likely readers are people
  building non-English or domain models — who are exactly the people who
  already train their own vocabularies.

### The cheap sweep has now been run — see docs/sweep-results.md

English FineWeb, 100 MB to 10 GB, vocab 8k/32k/128k. The result reframes the
paper:

- Fertility and compression **saturate almost immediately**: 100x more data
  moves fertility by 0.06–0.5%.
- Vocabulary identity **does not** saturate, and the gap grows with vocab
  size: at 128k, 100 MB recovers only 81% of the 10 GB vocabulary.
- So at 128k vocab, 100x less data gives **19% different tokens and 0.5%
  different fertility**.

This weakens the "you need more data" motivation for English, and it should
be stated that way in the blog post rather than omitted.

But it sharpens the paper into a better question:

**"Tokenizer vocabularies differ substantially without differing measurably.
Does the difference matter?"**

That is a stronger paper than a saturation curve, because it is a critique of
how the field evaluates tokenizers rather than another benchmark. Two
outcomes, both publishable: if downstream loss is also flat, then vocabulary
identity genuinely does not matter and everyone can stop worrying about
tokenizer training data — a useful negative result. If it is not flat, then
fertility is an inadequate proxy and the field has been optimising the wrong
metric.

### The order I'd actually do it in now

1. Ship the tool (blog + X). It stands alone.
2. Extend the sweep to **code and multilingual** corpora — English web text
   is the most homogeneous case and the least likely to show an effect. This
   is still cheap, and it is where the tail plausibly matters.
3. Only then commit to the downstream-loss arm, which is the expensive part
   and the part that decides whether the paper is interesting or merely
   tidy.
