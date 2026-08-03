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
does not measurably.** Across every composition and vocabulary size, a 100 MB
corpus gives a vocabulary that is 6–37% different from the large-corpus one.
Fertility moves by at most 1.3% — and the seed repeats below put the noise
floor at roughly ±0.7% for code and multilingual, so most of those
differences are **at or below what resampling produces**. The conclusion is
therefore stronger than "the effect is small": for fertility, no effect is
resolvable at this sample size. The vocabulary-overlap effect is real and is
monotone in vocabulary size — larger vocabularies have longer tails, and
tails need data.

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

**4. Multilingual vocabularies are far less stable across samples.** Three
independent samples per cell (vocab 32k, `modal run ...::seeds`), reported as
mean ± half-range:

| composition | overlap at 100 MB | fertility at 100 MB |
|---|---|---|
| english | 0.930 ± 0.001 | 1.361 ± 0.000 |
| code | 0.833 ± 0.002 | 3.525 ± 0.023 |
| multilingual | **0.857 ± 0.029** | 15.698 ± 0.110 |

Multilingual overlap varies **15–30x more between samples** than English or
code. With five languages splitting one budget, which language wins the tail
really is close to arbitrary — but the right way to say that is as variance,
not as a curve shape.

**Retraction.** An earlier version of this document reported that
multilingual overlap was non-monotone (0.881 at 100 MB against 0.870 at
300 MB) and read meaning into it. That difference is 0.011, well inside the
±0.029 sample spread. It was noise, and the interpretation built on it is
withdrawn.

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

- The main grid is one seed per cell. Three-seed repeats were run at vocab
  32k for 100 MB and 300 MB only (finding 4); everything else is a point
  estimate and fine structure in it should not be read.
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

---

# Follow-up analyses

Three questions the size sweep left open, answered with the same tokenizers
(`modal run scripts/modal_sweep.py::deeper`). All at a 1 GB corpus.

## 1. What does using the wrong domain's tokenizer cost?

bytes/token, higher is better compression. Rows are what the tokenizer was
trained on, columns what it was evaluated on.

**vocab 32000**

| trained on | english | code | multilingual |
|---|---|---|---|
| english | **4.401** | 1.975 | 1.342 |
| code | 3.664 | **3.390** | 2.123 |
| multilingual | 2.667 | 1.671 | **4.565** |

Cost of the mismatch, against the native tokenizer:

| text | wrong tokenizer | penalty |
|---|---|---|
| english | code | 16.7% |
| english | multilingual | 39.4% |
| code | english | 41.7% |
| code | multilingual | 50.7% |
| multilingual | english | **70.6%** |
| multilingual | code | 53.5% |

Two things stand out. **The penalty is asymmetric**: a code tokenizer on
English text costs 16.7%, but an English tokenizer on code costs 41.7%. Code
corpora contain a great deal of English — comments, identifiers, docstrings —
so a code tokenizer partly subsumes an English one, and not the reverse.

**Multilingual text is the most punished**, at 70.6% worse under an English
tokenizer. That is the concrete cost of applying an English-trained
vocabulary to non-English text, and it is far larger than any effect corpus
size produced.

## 2. Where in the vocabulary does the divergence live?

Overlap of the top-N tokens by merge rank, across domain pairs.

| pair | top 256 | top 1000 | top 4000 | top 16000 | full 32k |
|---|---|---|---|---|---|
| english \| code | 0.836 | 0.508 | 0.384 | 0.361 | 0.359 |
| english \| multilingual | 0.793 | 0.260 | 0.130 | 0.090 | 0.088 |
| code \| multilingual | 0.797 | 0.258 | 0.128 | 0.093 | 0.094 |

**The head is largely shared and the tail is almost entirely domain-specific.**
About 80% of the first 256 tokens are common to any pair of domains; by 4000
tokens that is 13% for anything involving multilingual. English and code
retain 36% agreement all the way out, again because code contains English.

This is the mechanism behind the size-sweep result. The tokens that carry
most of the traffic are forced by the data and appear everywhere; the tail is
where vocabularies differ, and the tail contributes little to fertility.

**Incidental confirmation:** the top-N rows are identical between the vocab
8000 and vocab 32000 runs. Greedy BPE is prefix-stable — training to 8k
produces the first 8k tokens of the 32k vocabulary on the same corpus — so
vocabularies are nested across target sizes.

## 3. Per-language equity in the multilingual tokenizer

Characters per token, which is the fair cross-script measure.

| language | vocab 8k | vocab 32k |
|---|---|---|
| Arabic | 2.718 | 3.434 |
| German | 2.263 | 2.976 |
| Japanese | 1.237 | 1.676 |
| Russian | 1.504 | 1.560 |
| Hindi | 1.493 | 1.544 |
| **worst/best ratio** | **2.20** | **2.22** |

A speaker of the worst-served language needs **2.2x as many tokens** for the
same amount of text as the best-served one, and quadrupling the vocabulary
does not narrow that gap at all (2.20 → 2.22). Japanese, Russian and Hindi
all sit near 1.5 characters per token while Arabic and German are 2.3–3.4.

**Methodological note, recorded because the first version of this analysis was
wrong.** It originally reported bytes per token, which made Arabic look
*best-served* (4.811) and German *worst* (2.299) — the exact opposite of the
character-based ranking. Bytes per token is confounded by UTF-8 width: Latin
text is ~1.4 bytes/char, Cyrillic and Arabic ~2.0, Devanagari and Japanese
~3.0. A byte-based equity figure flatters non-Latin scripts by roughly the
ratio of their encoding width, for reasons that have nothing to do with the
tokenizer. Any equity claim measured in bytes should be treated as suspect.

## What these add to the picture

Corpus size changes the vocabulary but not measured quality. **Domain and
language change measured quality a great deal** — 17–71% in compression, and
a 2.2x equity gap between languages that more vocabulary does not fix.

So the practically important lever is *what* you train on, not *how much*.
That is a cleaner and more useful message than the original motivation, and
it is supported by every arm of this sweep.
