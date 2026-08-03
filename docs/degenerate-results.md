# Degenerate corpora: what actually breaks a BPE trainer

Every benchmark in BENCHMARKS.md uses FineWeb — well-behaved English web text,
the case nobody fails on. But the failure reports motivating this project are
not merely *large*, they are *degenerate*:

- [tokenizers#1313](https://github.com/huggingface/tokenizers/issues/1313):
  ~13 billion characters of DNA-like data at `vocab_size=512`, **no
  pre-tokenizer**, unfinished after 10+ hours on 256 threads.
- [sentencepiece#1021](https://github.com/google/sentencepiece/issues/1021):
  31.2 GB, vocab 4096, **`Alphabet size=4`** (genomic), 1.8 TB of RAM,
  unfinished at 24 hours.
- [tokenizers#1795](https://github.com/huggingface/tokenizers/issues/1795):
  100 GB of RAM supports ~1.5 GB of Chinese JSONL.

Two of those three are low-alphabet, boundary-poor inputs. That is a different
regime from "a lot of English", and nobody benchmarks it.

## Method

`scripts/degenerate_corpora.py` generates nine shapes, deterministic given
`--seed`. `scripts/degenerate_benchmark.py` runs gigatrain and HuggingFace over
each in both pretokenization modes, reporting wall time, peak RSS, exit status
and merge-list parity, and treating timeouts and crashes as results rather than
errors.

50 MB per corpus, vocab 32,000, 10-core M-series macOS, `tokenizers` 0.22.2,
**180 s timeout**. Peak RSS via `/usr/bin/time -l`.

## Results

| corpus | mode | gigatrain | HF | parity |
|---|---|---|---|---|
| base64 | whitespace | 172.8 s / 2.2 GB | **TIMEOUT** | — |
| base64 | ByteLevel | 14.2 s / 1.2 GB | 61.8 s / 4.7 GB | identical (31,935) |
| cjk_dense | whitespace | 8.7 s / 1.1 GB | 31.9 s / 3.6 GB | identical (31,488) |
| cjk_dense | ByteLevel | 26.3 s / 1.2 GB | 61.0 s / 3.4 GB | identical (31,934) |
| cr_only | whitespace | 0.2 s / 62 MB | 9.9 s / 3.2 GB | identical (24) |
| cr_only | ByteLevel | 1.6 s / 54 MB | 21.5 s / 4.4 GB | identical (32) |
| dna_fasta | whitespace | 15.8 s / 1.5 GB | 67.0 s / 5.2 GB | identical (31,996) |
| dna_fasta | ByteLevel | 15.5 s / 1.5 GB | 76.9 s / 5.3 GB | identical (31,995) |
| dna_oneline | whitespace | **TIMEOUT** | **TIMEOUT** | — |
| dna_oneline | ByteLevel | **TIMEOUT** | **TIMEOUT** | — |
| giant_word | whitespace | 5.6 s / 1.5 GB | **TIMEOUT** | — |
| giant_word | ByteLevel | 4.7 s / 1.6 GB | **TIMEOUT** | — |
| json_oneline | whitespace | **TIMEOUT** | **TIMEOUT** | — |
| json_oneline | ByteLevel | 2.3 s / 246 MB | 30.0 s / 3.7 GB | identical (31,965) |
| logs | whitespace | 2.5 s / 282 MB | 14.4 s / 1.5 GB | identical (31,951) |
| logs | ByteLevel | 0.9 s / 212 MB | 15.7 s / 468 MB | identical (31,949) |
| minified_js | whitespace | 22.4 s / 1.1 GB | 82.0 s / 3.4 GB | identical (31,973) |
| minified_js | ByteLevel | 0.5 s / 73 MB | 15.9 s / **40 MB** | identical (10,009) |

**18 configurations. gigatrain completed 15, HuggingFace completed 12. In all
12 cases where both finished, the merge lists are byte-identical.** Speedups
range 2.3x–55.7x, median **5.8x** — the same figure as the like-for-like
whitespace comparison on 12.9 GB of FineWeb, which is a useful sanity check.

## Finding 1: the pathology is a *pretokenizer* artifact, not a data property

The single largest effect in the table is not gigatrain versus HF, it is
whitespace versus ByteLevel on the same file:

| corpus | whitespace | ByteLevel | ratio |
|---|---|---|---|
| base64 | 172.8 s | 14.2 s | **12x** |
| minified_js | 22.4 s | 0.5 s | **45x** |
| json_oneline | TIMEOUT | 2.3 s | ∞ |

Whitespace splitting turns any run without a space into one enormous "word".
base64 lines are 2000–4000 characters with no spaces; minified JS is
punctuation-dense with no spaces; the JSON dump has no whitespace at all. The
GPT-2 regex splits all of these into small pretokens, so the merge loop sees
ordinary work.

**This matters because production tokenizers use ByteLevel.** The
configuration that fails is the one nobody ships — and it is the configuration
#1313 used (`no pre-tokenizer`).

## Finding 2: the one irreducible case

`dna_oneline` is 50 MB of `ACGT` with no whitespace and no newline. Both
trainers time out in **both** modes, and that is correct behaviour rather than
a bug in either: `A`, `C`, `G` and `T` are all `\p{L}`, so the GPT-2 regex
matches the entire file as a single pretoken, exactly as whitespace splitting
does. There is no pretokenizer under which this input is anything but one
50-million-symbol token.

BPE on a single token of length *n* costs O(*n*) per merge, because both
implementations rescan the whole word:
[`word.rs:116`](../gigatrain/src/word.rs#L116) here, and HF's `Word::merge`,
which loops `i` from 0 over `self.symbols` (verified in `v0.22.2`). 32,000
merges over 50M symbols is ~10^12 operations. **This is a property of the
algorithm, not of either implementation, and no memory layout fixes it.**

The fix that would help is CLAUDE.md's position-indexed occurrence list, which
replaces the full-word rescan. It is listed in BENCHMARKS.md as a real win on
paper carrying real parity risk. This experiment is the first evidence about
when it would actually matter: **only when a single pretoken is very large**,
which is rare in text and normal in genomic data.

## Finding 3: gigatrain survives a giant token; HuggingFace does not

`giant_word` is a 45 MB single token followed by ordinary text. gigatrain
finishes in 5.6 s / 1.5 GB (whitespace) and 4.7 s / 1.6 GB (ByteLevel).
HuggingFace times out in both.

The mechanism is memory layout, which is this project's central design claim.
HF's `Word` holds a `Vec<Symbol>` where `Symbol` is `{c, prev, next, len}` at
roughly 32 bytes; a 45M-symbol word therefore costs ~1.4 GB **for one word**.
gigatrain's flat arena stores a `u32` symbol plus a `u32` length — 8 bytes, so
~360 MB for the same word. Note this is the one place where HF's
doubly-linked-list layout *should* have helped and the constant factor sinks
it.

Why `giant_word` is tractable at all while `dna_oneline` is not: the giant
token is a run of one character, so `(x,x)` merges pairwise and the word halves
each time — ~25 merges consume it. `dna_oneline` has four characters in random
order, so no merge shortens it appreciably.

## Honest caveats

- **`TIMEOUT` means "did not finish in 180 s", not "cannot finish".** The
  `dna_oneline` runs would presumably complete eventually. Only the ordering is
  established, not the failure.
- **The `cr_only` memory ratio is not a merge-phase result.** That corpus draws
  from an 8-word vocabulary, so only 24 merges are possible and both trainers
  stop early. gigatrain's 62 MB against HF's 3.2 GB is real and reproducible,
  but it measures baseline overhead on a near-empty word table. HF holding
  3.2 GB for 8 distinct words is itself worth noting.
- **HF wins one case.** `minified_js` under ByteLevel: 40 MB against
  gigatrain's 73 MB. It is the only memory loss in the table and it is not
  explained here.
- **50 MB only.** 200 MB corpora are generated by the same script but were not
  run; the buffering slope below is measured only at 50 MB.
- One machine, one OS, one allocator, one seed.

## The buffering weakness did not bite at this size

Under ByteLevel the reader cuts only after newlines, so a file with no `\n`
must be buffered whole ([`reader.rs:229`](../gigatrain/src/reader.rs#L229)).
`cr_only` and `json_oneline` have no newline anywhere.

`cr_only` isolates the cost cleanly. It peaks at 54 MB on a 50 MB file, and
`GIGATRAIN_STATS=1` reports **17 unique words** with RSS flat at 55 MB from
the end of phase 1 onward. With a word table that small, essentially all of
that 55 MB *is* the buffered file. So the buffering cost is exactly the length
of the longest boundary-free run, as predicted — no more, but no less.

`json_oneline` peaks higher at 246 MB, but that is **not** mostly buffering:
the same instrumentation reports 782,623 unique words and RSS climbing after
phase 1, so the bulk is word table and merge-phase structure. An earlier draft
of this document attributed the 4.9x to buffering; that was wrong.

### The multi-GB single line, measured — and the cost is not memory

A 2.0 GB single-line JSON file (the 200 MB corpus concatenated 10x; **zero
newline bytes**, and this JSON has no spaces either, so there is no cut point
of any kind), against a control that is the same 2.0 GB with a newline every
1000 bytes. Same trainer, same ByteLevel pretokenizer, 10-core macOS:

| | no cut points | newline every 1000 B |
|---|---|---|
| wall | **155.1 s** | **9.3 s** |
| phase 1 | 151.4 s | 5.2 s |
| peak RSS | 2.2 GB | 1.0 GB |
| effective readers | 1 | 8 |
| unique words | 3,094,044 | 3,100,657 |

**16.7x slower and 2.2x more memory, from nothing but the absence of cut
points.** The near-identical unique-word counts confirm the two files carry the
same content.

This corrects the framing in README.md, which describes the boundary-free case
purely as a memory hazard. **Memory is the smaller problem**: 2.2 GB peak on a
2.0 GB file is 1.1x, which is unremarkable. The real cost is that phase 1
collapses to a single thread — 29x slower — because `split_ranges` hands out 8
ranges, seven of them find no boundary and early-return, and range 0 buffers
the entire file into one chunk consumed by one scanner.

So the accurate statement is: **a corpus with no cut points loses phase-1
parallelism entirely and buffers the longest boundary-free run.** At 2 GB that
is 155 s instead of 9 s; the memory is fine.

## Reproducing

```bash
python scripts/degenerate_corpora.py --out-dir /tmp/degen --size-mb 50
python scripts/degenerate_benchmark.py --corpus-dir /tmp/degen \
    --vocab-size 32000 --timeout 180 --json-out results.json
```
