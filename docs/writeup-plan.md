# Publication plan: X thread, blog post, and the paper this unblocks

> **Rewritten 2026-08-02 after an adversarial fact-check.** The previous
> version of this file contained, among other things: the retracted #1313
> causal claim (in the very thread whose notes forbid it), a tweet mixing
> 16-core and 64-core numbers (the exact error its own notes call out), and an
> instruction not to repeat a Reddy et al. finding that is in fact in the
> paper. It also treated the intrinsic sweep as a completed result; that sweep
> has since been **retracted in full** (docs/sweep-results.md). Treat any
> earlier copy of this file as unusable.

Everything below is drafted against measurements recorded in this repo. Any
number not in BENCHMARKS.md, PARITY.md or PRIOR_ART.md should be cut rather
than estimated. This rule has now been broken twice, so it is worth restating
as a procedure: **before publishing, grep each number in the draft against the
repo. If it appears only in the draft, it is not a measurement.**

Publish order: flip repo public → tag `v0.1.0` so wheels exist → email Marcel
Rød → blog post → X thread pointing at the blog.

---

## The one number to lead with

The single cleanest result in the repo, and the one that was buried:

**12.9 GB of FineWeb, vocab 32k, ByteLevel on both sides, 16 cores:
gigatrain 38.0 s, HuggingFace 257.1 s — and the 31,790 merges are byte
identical.** (BENCHMARKS.md, "Parity verified at 12.9 GB".)

That is 6.8x, same pretokenizer, same output, one machine, one table row. It
needs no caveat, which none of the bigger multipliers can say. Every larger
number in this repo (17.2x, 22.3x, ~50x) compares different pretokenizers,
different machines, or was never measured at all.

---

## Part 1 — The X thread

### The framing decision

**Do not lead with "fastest BPE trainer."** gigatoken's trainer is only 1.8x
behind, so "fastest" is true but thin, and being corrected in the replies costs
more than the claim gains.

**Lead with parity at scale**, not with the anti-scaling finding. The earlier
plan led with anti-scaling; that was a mistake, because the measurement behind
it is not controlled (see below) and it would be the first thing an informed
reader attacks.

### Draft thread

> **1/**
> I built a BPE tokenizer trainer that's byte-exact against HuggingFace and
> ~7x faster.
>
> 12.9 GB of FineWeb → 32k vocab, same ByteLevel pretokenizer, same 16-core
> box:
> · gigatrain 38 s
> · HuggingFace 257 s
>
> All 31,790 merges identical.
>
> **2/**
> Parity is the whole point. A faster trainer nobody can verify is a demo.
>
> Matching HF exactly meant reproducing behaviours that look like bugs — the
> tie-break rule, stale heap entries being corrected and re-pushed rather than
> dropped, and `max_token_length` filtering only newly-formed pairs.
>
> **3/**
> My favourite: HF's trainer feeds files **one line at a time**. So a trailing
> \r\n is terminal within its line and becomes a single token.
>
> Undocumented, and it silently changes your merge list. It's exactly how
> another trainer I benchmarked matches HF on LF corpora and diverges at merge
> #0 on CRLF ones.
>
> **4/**
> The memory result surprised me more than the speed.
>
> Everyone assumes the pair index is the hazard. Profiled at 1 GB it was 1 MB.
> The actual hog was phase 1's HashMap<String, u64> — 1.7 GB of a 2.26 GB
> peak, from per-word allocations and per-worker duplication.
>
> **5/**
> Worth saying plainly: the algorithm is not new. Incremental counting with an
> inverted index and a lazy heap is Sennrich 2015, formalized by Zouhar 2023,
> and already in tokenizers, SentencePiece and rustbpe.
>
> The contribution is memory layout and a parallel phase 1.
>
> **6/**
> And gigatoken already ships a BPE trainer with HF tie-breaking. Undocumented,
> validated at ~120 KB, but it exists — and it's the closest competitor at
> 1.8x, not an order of magnitude.
>
> Full prior-art writeup in the repo, findings that undercut the project first.
>
> **7/**
> Probably the most reusable artifact isn't the trainer. It's PARITY.md — a
> specification of HuggingFace's exact training semantics: the tie-break rule,
> stale-heap handling, an i32 overflow, and which behaviours are
> nondeterministic.
>
> That document doesn't exist anywhere else, including HF's own docs.
>
> **8/**
> Then I pointed adversarial agents at it. They found real bugs in the trainer
> — and then, in a second pass, six independent defects in my own *research*
> code that invalidated most of an experiment I'd already written up.
>
> Retracted in the repo. Blog post: [link]

### Notes for the thread

- **Tweet 1 must use the 16-core ByteLevel row** (38 s vs 257 s), because that
  is the one where both sides ran the same pretokenizer *and* the merge lists
  were diffed. Do not attach peak-RSS figures to it: no RSS was recorded for
  the 16-core runs. The 2.7 GB / 29.8 GB memory pair is from the **64-core**
  benchmark and belongs only in a sentence that says so.
- **Do not claim "19x slower from more cores."** The 9.7 s and 181 s figures
  come from different ISAs, OSes, allocators and machines. No HF core sweep on
  a single box exists. If the anti-scaling point is made at all, it is "HF is
  markedly slower on the bigger machine, and the mechanism is visible in its
  source" — and someone will rightly ask for the controlled experiment.
- **Do not mention #1313 as explained, reproduced, or undiagnosed.** It is a
  `vocab_size=512` run on unsegmented DNA-like data, and a maintainer
  diagnosed it in-thread as degenerate pretokenization. Claiming it "was never
  explained" is checkable in thirty seconds and false.
- Expect these replies and have answers ready:
  - *"What about gigatoken?"* → tweet 6, link PRIOR_ART.md.
  - *"You just implemented the standard algorithm."* → yes, tweet 5 says so.
  - *"Did you use ByteLevel for yours and whitespace for theirs?"* → no; tweet
    1 is ByteLevel on both sides. The mismatched comparisons are documented in
    the repo as corrected errors.

---

## Part 2 — The blog post

Working title: **"Byte-exact BPE training, 7x faster — and the six ways I
fooled myself measuring it."**

The honest version of this post is now more interesting than the original
plan, because the retraction story is the strongest material in it. Target
2,500–3,500 words.

### Structure

**1. The hook (300 words)**
Parity at 12.9 GB with the 38 s / 257 s ByteLevel row. State the hardware
once, up front.

**2. Why tokenizer training is the neglected stage (400 words)**
Encoding is solved (gigatoken, ~24 GB/s). Dedup is solved several times over.
Training has HF #1681 (20 GB OOM on 1.5–2 TB), #1795 (100 GB of RAM for
~1.5 GB of Chinese JSONL), sentencepiece#1021 (31.2 GB → 1.8 TB, unfinished at
24 h). The universal workaround is sampling down.

Cite Reddy et al. **correctly** — and note that this correction replaces an
earlier, wrong correction in this repo. They trained BPE, UnigramLM and
WordPiece from 1 GB to 900 GB (English) and to 600 GB (Russian), and found
diminishing returns beyond ~150 GB for English and ~200 GB for Russian. They
used HF for UnigramLM and WordPiece and built their BPE trainer on minbpe.
**The "90% overlap" finding is in the paper** — §1/Fig. 1 reports shared
vocabulary against the 900 GB reference rising "from approximately 58% to 97%
for BPE, from 40% to 97% for UnigramLM, and from 4% to 92% for WordPiece."

**3. Parity as the hard requirement (500 words)**
The part most readers won't expect. Reproducing behaviours that look like bugs:
- the tie-break rule (higher count, then *smaller* ID pair)
- stale heap entries corrected and re-pushed, not discarded
- `max_token_length` filters only newly-formed pairs, never the initial count
- HF's trainer feeds files **one line at a time**, so a trailing `\r\n` is
  terminal within its line and becomes one token — undocumented, and it
  silently changes output

That last one is the good story: it is how gigatoken's trainer diverges from
rank 0 on CRLF corpora while matching exactly on LF ones.

**4. Where the memory actually goes (600 words)**
The best technical section. Everyone predicts the pair index; it was 1 MB at
1 GB. Phase 1's `HashMap<String, u64>` was 1.7 GB of 2.26 GB. Then the three
fixes (arena-backed sharded counter 1.7 GB → 422 MB; dropping word strings
once tokenized; flat symbol arena removing 4.1M allocations), and the rejected
design: broadcasting chunks to every shard owner fixes memory but replicates
scanning per worker — 4.8 s vs 1.3 s, with an Amdahl floor that worsens as
cores rise.

**5. Four optimizations that didn't work, and one that did — twice (600 words)**
The section most worth writing, because nobody publishes it.

| change | why it should have worked | result |
|---|---|---|
| 60/40 scanner/owner split | CPU split measured ~58/42 | no effect |
| byte-table ASCII path in piece scanning | stop decoding chars to classify | **10% slower** |
| zero-copy batching | removes a memcpy pass over the corpus | slightly worse |
| thread-pool retune at 10 cores | apparent oversubscription | inside noise, reverted |
| the same retune at 64 cores | — | **worked: 1.46x, 2.2x less memory** |

(Four rejections; the last row is the fourth change re-measured on different
hardware, not a fifth change. The earlier draft of this post called it "five
optimizations that didn't work" over a table whose last row says it worked.)

The ASCII path is the lesson: replacing `char_indices()` with a per-character
`class_at()` that indexes `text[i..]` re-does a UTF-8 boundary check every
character. Faster-looking code, slower code.

The meta-lesson: the same change was correctly rejected at 10 cores and
correctly accepted at 64. The laptop wasn't a weaker server; it was the wrong
instrument.

**6. The adversarial audit, in two acts (900 words — now the centrepiece)**

*Act one, the trainer.* Agents with distinct lenses found real bugs. The three
worth describing:
- **An invariant I asserted and never verified.** A per-token-ID length table,
  documented as "always exactly HF's per-symbol length." False once
  `continuing_subword_prefix` is set, because one ID becomes reachable both as
  an initial symbol and as a merged token. Silent wrong output.
- **`process::exit(2)` in library code**, which killed the caller's Python
  interpreter on a missing file — no exception, no `finally`, no flush.
- **A claim that was never computed.** The README said the 12.9 GB run
  produced a byte-identical merge list. The harness discarded stdout. Same
  criticism the repo levels at gigatoken's 120 KB parity test. Since computed:
  identical, both modes.

*Act two, the research code — the part actually worth reading.* Having
hardened the trainer, I pointed the same technique at the experiment I had
just written up. It found six independent defects, each sufficient on its own
to invalidate a headline:
- the multilingual held-out set was a single language (the last shard, and the
  shards are grouped by language), so "multilingual is 70.6% worse" was
  "Japanese is"
- per-language equity was measured on text inside the training corpus
- the multilingual corpus was never language-balanced — one language held 32%
  of the characters, the worst-scoring one held 9%
- the "independent seeds" shared 91–95% of documents for English and 0% for
  multilingual, so a finding about "variance across samples" was a finding
  about shard layout
- the rank-stratified overlap metric was measuring the byte alphabet, because
  the first merge lands around token id 170 and the "top 256" window is ~97%
  alphabet

And the one that should have been caught by reading a paper rather than by an
agent: **the headline measurement was already published.** Vocabulary overlap
against a large-corpus reference as a function of training size is Reddy et
al. Figure 1, across three algorithms, at 900 GB — 90x my reference corpus.

Closing note: I wrote the code, the tests, and the comments asserting
correctness. I did not find any of this. The lesson isn't "use agents"; it's
that the confidence I had in the sweep was indistinguishable, from the inside,
from the confidence I had in the parity work — and only one of them was
earned.

**7. What's actually novel, and what isn't (300 words)**
Not novel: the algorithm; being first to HF-parity training (gigatoken); the
corpus-size/vocabulary-overlap curve (Reddy et al.).
Novel: the phase-1 sharded shuffle, the parity specification, and a benchmark
harness reporting wall time, peak RSS *and* merge parity together — no
existing trainer benchmark reports memory at all, and the only suite that ever
existed was archived in 2024.

**8. What this unblocks (200 words)** → Part 3.

### Things the post must include for credibility

- The reproduction command (`modal run scripts/modal_benchmark.py`)
- Hardware for every number: 10-core M-series macOS, or 64-core x86-64 Linux,
  or the 16-core parity runs — and never numbers from two of them in one claim
- Both known divergences: HF's i32 overflow past ~2^31 pair occurrences
  (~120–150 GB of English text), and the decorated modes, where HF is usually
  nondeterministic and gigatrain differs even in the cases where it is not
- The input-dependent memory caveat: a whitespace-free corpus peaks at ~4.5x
  its size — the same failure shape criticised in HF #1681
- gigatoken, prominently, not buried
- A link to the retraction, not just a mention of it

### Things to leave out

- The 17.2x, 22.3x and ~50x figures (different pretokenizers, or unmeasured)
- Any claim of reproducing, explaining, or being first to diagnose #1313
- "Three unrelated sources" for the scaling triangulation — they are different
  corpora, vocabularies and machines
- "~19x slower with more cores" as a controlled result
- Anything about the 198x naive-vs-incremental Python benchmark — a strawman
  nobody ships
- Every quantitative claim from the retracted sweep

---

## Part 3 — The paper this makes possible

**The tool is not the paper. The tool is the instrument.**

A systems paper on gigatrain would be rejected, and correctly: the algorithm
is Zouhar et al. 2023, implemented three times over; the engineering is good
but not a scientific claim; and the anti-scaling observation is an
uncontrolled measurement of one library.

### The question, revised

The previous version of this section proposed a saturation curve. **Reddy et
al. have already published it** — English to 900 GB, Russian to 600 GB, three
algorithms, both the fertility-saturation and the vocabulary-overlap curves.
Proposing it again would have been the paper's central weakness, and it took
an adversarial check to notice.

What is genuinely unclaimed, in descending order of confidence:

1. **Does vocabulary identity matter downstream?** Reddy et al. stop at
   intrinsic metrics, as most of the literature does. If two tokenizers differ
   in 19% of their tokens but produce equal fertility, does a model trained on
   each reach the same loss? Both outcomes are publishable: flat means
   vocabulary identity genuinely doesn't matter and the field can stop
   worrying; not flat means fertility is an inadequate proxy and the field has
   been optimising the wrong metric. **This is the paper.** It is also the
   expensive arm, and the tool does not make it cheaper — the tokenizer
   training was never the bottleneck for this question.
2. **Code and domain-specific corpora.** Reddy et al. cover English and
   Russian natural language. Code, biomedical, legal and deliberately
   degenerate corpora (DNA, logs) are uncovered, and the degenerate case is
   where every incumbent trainer falls over — which is the one place the tool
   is genuinely load-bearing.
3. **Per-language equity in a properly balanced multilingual vocabulary.**
   Plausibly novel, but the retracted sweep shows how easily it is measured
   wrong. Requires character-balanced corpora, held-out sets disjoint from
   training, and per-language evaluation on text the tokenizer has not seen.

### Why the tool still matters here

Less than the previous draft claimed. Honestly: it makes the *sweep* cheap,
and the sweep was never the expensive part of the question that matters. Where
it is load-bearing is the degenerate-corpus arm, where HF and SentencePiece
genuinely fail, and in making the results non-dismissable — byte-exactness
against the reference implementation is precisely the criticism you could
level at a minbpe-derived trainer, which is what Reddy et al. used for BPE.

### Honest risks

- **It's an empirical study, not a method.** Resource/benchmark track or
  workshop, not a main-conference method paper.
- **The downstream-loss arm is the expensive part** and is what separates
  "interesting" from "citable." Budget for it or scope it explicitly.
- **Reddy et al. have more than a head start** — they have the result the
  previous plan proposed as novel. Differentiate on code, degenerate domains,
  and the downstream arm, or don't write it.
- **Tokenizer research has a modest audience**, mostly people building
  non-English or domain models — who already train their own vocabularies.

### The order to do it in

1. Ship the tool (blog + X). It stands alone, and the parity result is real.
2. **Rebuild the intrinsic sweep correctly** before quoting any number from
   it. The fixes are known and listed in docs/sweep-results.md: balance by
   characters not documents, hold out data disjoint from training, sample
   seeds by global offset rather than per-stream, stratify overlap by merge
   rank rather than token id, and validate every cached artifact.
3. Only then decide on the downstream-loss arm, which is what determines
   whether there is a paper at all.
