# Fast BPE Tokenizer Training

## What this is

A tokenizer **trainer** that runs at web scale. Not an encoder. Gigatoken
(https://github.com/marcelroed/gigatoken) already solved encoding at ~24 GB/s.
Nothing has solved training. This is the sibling project.

The pitch: train a 32k-vocab BPE tokenizer on 100 GB+ of text in minutes, with
byte-exact output parity against HuggingFace `tokenizers`.

## Why this is worth doing

HuggingFace's `BpeTrainer` is effectively unusable past a few GB:

- Issue #1313: 13 GB corpus, 256 threads, unfinished after 10+ hours.
  YouTokenToMe did the same job in under 10 minutes on 8 threads.
- Issue #1681: 20 GB corpus OOMs during the merge phase on 1.5 TB and 2 TB
  machines.

Everything published in response is hobbyist-scale: blog posts reporting 2000x
on a 114 MB Gutenberg corpus and 230x on TinyStories, plus a header-only C++
trainer that takes 1-2 hours for a 50k vocab. No production-grade tool exists.

Meanwhile the adjacent stages are all claimed. Dedup has FED (107x over CPU,
6.3x over NeMo Curator), SEDD (1.2T tokens in 3 hours on 32 V100s), and
LSHBloom. Constrained decoding has XGrammar as the default backend across vLLM,
SGLang, and TensorRT-LLM at under 40us/token. HTML extraction has moved to a
quality race (rs-trafilatura, Dripper-0.6B) rather than a speed race. Training
is the gap.

Secondary motivation: this unblocks research. Nobody studies vocabulary design
empirically at scale because you can't afford to train twenty vocabularies on a
terabyte. Make it cheap and that becomes tractable.

## The core algorithmic insight

Naive BPE training recounts every adjacent pair across the whole corpus on every
merge. That is O(pretokens * vocab_size).

The fix is incremental maintenance:

- `pair_count: HashMap<(Sym, Sym), u64>` maintained across merges, not rebuilt.
- `pair_where: HashMap<(Sym, Sym), HashSet<WordId>>`, an inverted index so a
  merge only touches words that actually contain the merged pair.
- On each merge, apply **delta updates** to the neighbouring pairs of each
  occurrence: decrement `(left, x)` and `(y, right)`, increment `(left, xy)` and
  `(xy, right)`.
- A lazy max-heap for merge selection, with stale entries filtered on pop by
  comparing against the live `pair_count`.

This is O(affected occurrences) per merge instead of O(corpus).

Measured in pure Python, before any Rust, SIMD, or parallelism:

```
corpus 4MB,  1200 merges:  naive 139.9s -> incremental  0.71s   (198x)
corpus 12MB,  800 merges:  naive 137.8s -> incremental  1.18s   (117x)
corpus 12MB,  200 merges:  naive  31.4s -> incremental  0.39s    (82x)
```

`bpe_incremental.py` in this repo is the working reference implementation of
both. Port it, do not reinvent it.

## Non-negotiable requirement: exact parity

This is the difference between a real tool and a demo, and it is the hard part.

Output must match `tokenizers.trainers.BpeTrainer` merge-for-merge, including
tie-breaking. The Python reference implementation diverges from naive around
merge 375-490 purely on ties: the heap and the linear max disagree about which
of several equal-count pairs to pick.

Before optimizing anything, determine HuggingFace's actual tie-break rule by
reading its source, then encode it in the heap comparator. Add a CI check that
trains a small vocab both ways and asserts identical merge lists. Gigatoken's
README emphasizes exact-match parity for encoding for the same reason: without
it, nobody can adopt it as a drop-in.

## Architecture

Rust core, Python bindings via PyO3, mirroring Gigatoken's layout.

Phase 1, pretokenization and word counting. Embarrassingly parallel across
documents. Map-reduce into a frequency table of unique pretokens. This is where
multithreading pays off. Read files directly from Rust; do not round-trip
through Python.

Phase 2, the merge loop. Inherently sequential, since merge N+1 depends on
merge N. Single-threaded and memory-bound. Optimization here is about memory
layout, not parallelism.

## Do NOT use a GPU

Deliberate decision, not an oversight:

- The merge loop is sequential by construction. 32k merges means 32k dependent
  steps that cannot be parallelized across.
- The hot data structures are hash maps and variable-length sets. Pointer
  chasing and irregular access, not dense arithmetic.
- Work per merge is small and irregular, so kernel launch overhead dominates.

Dedup went GPU and got claimed by three groups precisely because MinHash *is*
dense parallel hashing. This workload is the opposite shape. That mismatch is
why this gap is still open.

Target hardware: many CPU cores for phase 1, large RAM for phase 2.

## Memory is likely the real problem

Issue #1681 OOMed on 2 TB. Assume memory layout is the binding constraint, not
CPU time, and design for it from the start:

- Arena/bump allocation for word symbol sequences. No per-word `Vec`.
- Compact symbol IDs (u32) rather than string handles or byte tuples.
- Consider a flat `symbols: Vec<u32>` with per-word offset slices, and a
  doubly-linked-list representation over that arena so merges are O(1) splices
  rather than rebuilding sequences.
- Sets in `pair_where` are a memory hazard. Consider small-vec optimization
  since most pairs occur in few words, and beware the Zipfian head where a few
  pairs occur in nearly every word.
- Streaming/out-of-core phase 1 so the frequency table, not the corpus, is what
  has to fit in RAM.

## Milestones

1. Port `bpe_incremental.py` to Rust, single-threaded, correctness only.
2. Establish exact HF parity with a CI test. Do not proceed without this.
3. Benchmark harness against `tokenizers` on 100 MB, 1 GB, 13 GB. The 13 GB
   number is the headline, since it is the exact case in issue #1313.
4. Parallelize phase 1.
5. Memory-optimize phase 2. Target: 20 GB corpus trains without OOM on a normal
   machine, directly answering issue #1681.
6. PyO3 bindings with a `BpeTrainer`-compatible API so it is a drop-in.
7. Extend to WordPiece and Unigram/SentencePiece, both of which Gigatoken lists
   as unsupported.

## Benchmarking rules

- Always report against `tokenizers` as baseline, since it is the reference that
  produced real tokenizers.
- Use real corpora (OpenWebText, FineWeb samples). Synthetic Zipfian text does
  not reproduce real failure modes.
- Report wall time, peak RSS, and merge-list parity together. A speedup with
  different output is not a speedup.
- Vary both corpus size and vocab size independently. Naive scales with their
  product; the whole claim is that this does not.

## Positioning

Frame as complementary to Gigatoken, not competing. It encodes at GB/s; nothing
trains at GB/s. Worth reaching out to Marcel Rød early, since it could land as a
sibling crate or a contribution, and his repo already flags WordPiece and
SentencePiece as gaps.
