# Intrinsic sweep: does more training data change the tokenizer?

Run 2026-08-02 with `modal run scripts/modal_sweep.py`. English FineWeb,
ByteLevel pretokenization, nested corpora (each smaller slice is a prefix of
the larger). Reference for each row group is the 10 GB tokenizer at the same
vocabulary size. Held-out text is 20 MB from a parquet no training slice
reaches.

| vocab | corpus | vocab overlap | leading merges | fertility | bytes/token |
|---|---|---|---|---|---|
| 8k | 100 MB | 0.955 | 2 | 1.593 | 3.749 |
| 8k | 300 MB | 0.969 | 21 | 1.592 | 3.751 |
| 8k | 1 GB | 0.979 | 21 | 1.591 | 3.751 |
| 8k | 3 GB | 0.984 | 18 | 1.592 | 3.751 |
| 8k | 10 GB | 1.000 | 7790 | 1.592 | 3.750 |
| 32k | 100 MB | 0.916 | 2 | 1.362 | 4.384 |
| 32k | 300 MB | 0.944 | 21 | 1.360 | 4.389 |
| 32k | 1 GB | 0.969 | 21 | 1.359 | 4.392 |
| 32k | 3 GB | 0.980 | 18 | 1.359 | 4.393 |
| 32k | 10 GB | 1.000 | 31790 | 1.359 | 4.394 |
| 128k | 100 MB | **0.811** | 2 | 1.275 | 4.682 |
| 128k | 300 MB | 0.874 | 21 | 1.272 | 4.693 |
| 128k | 1 GB | 0.925 | 21 | 1.270 | 4.700 |
| 128k | 3 GB | 0.957 | 18 | 1.270 | 4.702 |
| 128k | 10 GB | 1.000 | 127790 | 1.269 | 4.704 |

## Three findings

**1. Intrinsic quality saturates almost immediately.** Going from 100 MB to
10 GB — a 100x increase — changes fertility by 0.06% at 8k vocab, 0.2% at
32k, and 0.5% at 128k. Compression moves equally little. For English web
text, at these scales, **corpus size is very nearly irrelevant to how well
the tokenizer performs.**

**2. Vocabulary identity does not saturate, and the gap grows with
vocabulary size.** At 100 MB the recovered fraction of the 10 GB vocabulary
is 95.5% (8k), 91.6% (32k), 81.1% (128k). Even at 3 GB, a 128k vocabulary is
only 95.7% of the reference. Larger vocabularies have a longer tail, and the
tail needs data.

**3. These two facts together are the interesting part.** At 128k vocab,
training on 100x less data gives you a vocabulary that is **19% different
tokens and 0.5% different fertility**. The identity of the tail is close to
arbitrary; the tokens are individually different and collectively
interchangeable.

Merge order is not a useful metric: 2 to 21 leading merges agree out of tens
of thousands, at every size. Early ties resolve differently on different
corpora and the sequence diverges immediately, even where the resulting
vocabulary is 98% identical. Reporting merge-prefix agreement as a similarity
measure would be misleading.

## What this means for the project's own motivation

It weakens it, and that should be said plainly. The pitch — that people
sample corpora down and lose something — is **not supported for English web
text by intrinsic metrics**. Someone training an English BPE tokenizer on
100 MB is getting essentially the fertility they would get from 10 GB.

The tool is still what made this measurable: the whole 15-tokenizer grid,
including three 10 GB runs, took under four minutes of training. But the
justification shifts from "you need more data" to "here is the evidence about
whether you do".

## What would change the picture

- **Code and multilingual text.** English web text is the most homogeneous
  case. A long tail of identifiers, or 100 languages competing for the same
  vocabulary budget, is where the tail plausibly matters. Untested.
- **Downstream loss.** Every metric here is intrinsic. The finding that 19%
  of tokens differ with no fertility change is precisely the setup where
  intrinsic metrics may be inadequate — either the difference genuinely does
  not matter, or it matters somewhere fertility cannot see.
- **Vocabulary sizes past 128k.** The trend across 8k/32k/128k is monotone;
  256k+ may need substantially more data.

## Caveats

- English FineWeb only; one held-out set; single seed.
- Nested corpora, so larger slices contain the smaller ones. This is the
  right design for a size sweep but means the samples are not independent.
- The `train_seconds` column in the raw output is contaminated by page-cache
  warming: the vocab-8k runs came first and paid the disk cost for the 3 GB
  and 10 GB files (36 s and 107 s), while the same corpora took 8.5 s and
  23 s on later passes. Do not read those as vocabulary-size effects.
