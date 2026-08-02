# Intrinsic sweep: does more training data change the tokenizer?

Run 2026-08-02 with `modal run scripts/modal_sweep.py`. Nested corpora (each
smaller slice is a prefix of the larger), ByteLevel pretokenization. The
reference for each group is the largest-corpus tokenizer at the same
vocabulary size. Held-out text comes from a source shard no training slice
reaches.

Compositions: **english** (FineWeb), **code** (codeparrot Python),
**multilingual** (FineWeb-2: German, Russian, Arabic, Hindi, Japanese —
five languages, four scripts, documents interleaved so every prefix is
balanced).

## Vocabulary overlap with the reference tokenizer

How much of the big-corpus vocabulary a smaller corpus recovers.

| vocab | corpus | english | code | multilingual |
|---|---|---|---|---|
| 8k | 100 MB | 0.955 | 0.892 | 0.881 |
| 8k | 300 MB | 0.969 | 0.938 | 0.870 |
| 8k | 1 GB | 0.979 | 0.964 | 0.885 |
| 32k | 100 MB | 0.916 | 0.804 | 0.848 |
| 32k | 300 MB | 0.944 | 0.874 | 0.854 |
| 32k | 1 GB | 0.969 | 0.934 | 0.882 |
| 128k | 100 MB | 0.811 | **0.626** | 0.786 |
| 128k | 300 MB | 0.874 | 0.734 | 0.830 |
| 128k | 1 GB | 0.925 | 0.856 | 0.880 |

(English reference is 10 GB; code and multilingual are 3 GB.)

## Fertility — tokens per whitespace-word on held-out text

| composition | vocab | smallest corpus | largest corpus | change |
|---|---|---|---|---|
| english | 8k | 1.593 | 1.592 | 0.06% |
| english | 32k | 1.362 | 1.359 | 0.2% |
| english | 128k | 1.275 | 1.269 | 0.5% |
| code | 8k | 3.813 | 3.828 | 0.4% |
| code | 32k | 3.494 | 3.517 | 0.7% |
| code | 128k | 3.359 | 3.402 | 1.3% |
| multilingual | 8k | 20.832 | 21.108 | 1.3% |
| multilingual | 32k | 15.560 | 15.593 | 0.2% |
| multilingual | 128k | 12.503 | 12.351 | 1.2% |

## Findings

**1. Vocabulary identity depends strongly on corpus size; measured quality
barely does.** Across every composition and vocabulary size, a 100 MB corpus
gives a vocabulary that is 6–37% different from the large-corpus one, while
fertility moves by at most 1.3%. The effect is monotone in vocabulary size:
larger vocabularies have longer tails and the tail needs data.

**2. Domain changes how hard the tail is to fill.** At 128k vocab and 100 MB,
recovery is 0.811 (English), 0.786 (multilingual), 0.626 (code). Code is the
hardest — identifiers, API names and framework vocabulary form a genuinely
long tail — and it is the one domain where the curve is still climbing
steeply at 1 GB (0.856).

**3. Fertility differs enormously by domain at fixed vocabulary size.** At
32k: English 1.36, code 3.52, multilingual 15.6 tokens per whitespace-word.
The multilingual figure is inflated because Japanese and Chinese text has few
whitespace boundaries, so bytes/token is the fairer cross-domain measure:
English 4.39, code 3.39, multilingual 4.58. Code is the least compressible
per byte despite its smaller alphabet.

**4. Multilingual is the flattest arm, which was not the prediction.** Its
overlap curve is nearly level from 100 MB to 1 GB (0.881 → 0.885 at 8k), and
it is the only composition where a *smaller* corpus sometimes scores higher
(0.881 at 100 MB vs 0.870 at 300 MB). With five languages splitting one
budget, which language wins the tail appears to be closer to arbitrary than
to data-limited.

## What this means

The practical claim — that people lose something by sampling a corpus down —
**is not supported by intrinsic metrics in any of the three domains.** A
100 MB sample gets within ~1% of the fertility you would get from 10 GB. This
project's stated motivation is weaker than it appeared, and the writeup says
so.

The interesting question is now sharper, and it is the same in all three
domains: **vocabularies differ substantially (up to 37% of tokens) without
differing measurably (≤1.3% fertility). Does the difference matter
downstream?**

If it does not, everyone can stop worrying about tokenizer training data and
the field has a useful negative result. If it does, fertility is an
inadequate proxy and the field has been optimising the wrong metric. Either
answer is worth publishing, and neither can be reached with intrinsic
measures alone.

## Caveats

- One held-out set and one seed per cell. No error bars.
- Nested corpora, so samples are not independent (correct for a size sweep,
  but it means these are not 36 independent draws).
- English reference is 10 GB; code and multilingual are 3 GB, so the
  overlap columns are not directly comparable across compositions at the
  same corpus size — only the shapes are.
- Multilingual fertility is not comparable to the others because
  whitespace-word counting is meaningless for Japanese.
- codeparrot Python only; a multi-language code corpus may behave differently.

## Three measurement bugs found while running this

Recorded because each would have produced a plausible-looking wrong answer:

1. **Guessed shard indices.** Shard counts differ per config (German 181,
   Japanese 175, Python 10). A guessed index returns a JSON error that
   `curl -s` writes to disk as a `.parquet` file. Now resolved from the
   datasets-server and verified by magic bytes.
2. **Unvalidated download.** No check that a downloaded file was parquet
   until pyarrow failed on it.
3. **Unvalidated cache.** A crashed run left zero-byte corpora in the volume;
   the cache check tested only `os.path.exists`, so every size trained on the
   same empty file and produced *identical* results across all four sizes and
   all three vocabulary sizes. That was reported as a flat multilingual curve
   before the tell — `trained size=3000MB in 0.0s` — was noticed. Corpora are
   now size-validated and a short corpus raises instead of warning.
