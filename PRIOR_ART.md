# Prior art

Who else exists, how this compares, and what their published benchmarks
actually report. Leads with the findings that undercut this project.

Sources quoted here are archived verbatim under
[docs/audit-sources/](docs/audit-sources/) with SHA-256 and retrieval date.
Re-check with `python scripts/fetch_audit_sources.py --verify`.

---

## 1. Things that undercut the pitch

**gigatoken already has a BPE trainer with HuggingFace parity.**
[gigatoken](https://github.com/marcelroed/gigatoken) is known as an encoder
(~24 GB/s) but `src/bpe_train.rs` exposes `train_bpe` with
`TieBreaking::HuggingFace` as the default, and
`tests/test_bpe_train_compare.py::test_merges_identical` asserts the merge list
matches `BpeTrainer` position-for-position. Shipped in 0.10.0. So "nothing
trains with HF parity" is false, and it is the closest competitor — **3.9x** in
the one-session table, narrowly ahead of YouTokenToMe.

Where this project still differs:

| | gigatoken | gigatrain |
|---|---|---|
| parity test scale | ~120 KB synthetic, vocab 500 | 12.9 GB FineWeb, vocab 32k |
| parity scope | tie-break; hardcodes the 0–255 alphabet | alphabet construction, ID reuse, `min_frequency`, `max_token_length`, `limit_alphabet`, stale re-push, duplicate-merge |
| word representation | `Vec<u32>` per word | one flat `u32` arena |

**Where gigatoken's parity breaks.** Compared rank-for-rank against HF: on LF
corpora **2744/2744 identical**; on CRLF corpora **21/2744, diverging at rank
0**. HF's trainer feeds files a line at a time, so a trailing `\r\n` is terminal
within its line and yields `čĊ` — HF's rank-0 merge. gigatoken pretokenizes
whole-text and never forms it. This project's CI trains on a CRLF corpus
precisely because of that trap.

**The "core algorithmic insight" is the status quo.** Incremental pair counts +
inverted index + lazy heap is standard. Sennrich et al. 2015 §3.2 states the
*incremental* half only ("heap" and "priority" do not appear in that paper);
the heap and the O(NM) → O(N log M) proof are Zouhar et al.
([arXiv:2306.16837](https://arxiv.org/abs/2306.16837) §4/Thm 4.2). Both are
already in `tokenizers`, SentencePiece and rustbpe — HF's `word.rs` even uses
the `Symbol {prev, next, len}` linked list this project lists as future work.
**A naive-recount baseline is a strawman and must not be quoted as a speedup.**

**The scaling question was already answered.** Reddy et al.
([arXiv:2502.20273](https://arxiv.org/abs/2502.20273)) trained BPE, UnigramLM
and WordPiece from 1 GB to 900 GB (English), reporting diminishing returns
beyond ~150 GB / ~200 GB. The Russian corpus size (600 GB) and the shared-
vocabulary figure ("from approximately 58% to 97% for BPE") are from the paper
body, which is **not** in the archive — only the abstract page is
(`reddy-scaling-abs.html`); re-verify against the paper before quoting them
onward. This repo's own sweep re-measured that question at 90x smaller scale,
badly, and retracted it.

**No major lab trains its own BPE.** OLMo 3 reuses OLMo 2's cl100k-derived
vocab; gpt-neox wraps HF; Llama 4, DeepSeek, Mistral and Qwen 3 ship no trainer
code. NVIDIA's NeMo Curator states it "doesn't handle tokenizer training". The
demand argument rests on non-English and domain-specific builders.

**GPU BPE training does not exist.** Everything branded that way (BlockBPE,
GPUTOK, RAPIDS `nvtext`) is *encoding* from a pre-trained merge table. The only
GPU trainers are hobby-scale and abandoned.

---

## 2. The field

| Tool | Lang | Trains | HF parity | Status |
|---|---|---|---|---|
| HF `tokenizers` | Rust | yes | is the reference | 0.23.1, actively maintained |
| gigatoken `train_bpe` | Rust | yes | **yes, CI-tested at 120 KB** | 0.10.0, undocumented |
| YouTokenToMe | C++ | yes | no | **archived 2024** |
| SentencePiece BPE | C++ | yes | no, by design | v0.2.2 (July 2026) |
| `karpathy/rustbpe` | Rust | yes | no claim | active |
| ffbpe | Rust | yes | partial | created Dec 2025 |
| fast-bytelevel-bpe-go | Go | yes | **yes** (`vocab: SAME / merges: SAME`) | created June 2026 |
| tiktoken, rust-gems, GPUTOK, BlockBPE | — | **no** | n/a | encoding only |

**SentencePiece got ~20x faster in v0.2.2** — against its own past, not the
field. Its maintainer separately measured HF-style parallel merging as
ineffective-to-harmful: 1T 6.86 s vs 4T 8.01 s, with the merge step only ~24%
of training and the sequential priority queue capping speedup near 1.3x
([sentencepiece#366](https://github.com/google/sentencepiece/issues/366),
2026-06-16). That independently corroborates this repo's core-count sweep.

**Two HF trainer bugs that bear on any parity claim:**
- [#2058](https://github.com/huggingface/tokenizers/issues/2058): pair counts
  are `i32` and wrap negative on large corpora. This crate uses `i64`.
- [#1794](https://github.com/huggingface/tokenizers/issues/1794): BPE/WordPiece
  training is non-deterministic, because ids assigned in `AHashMap` order feed
  the tie-break. [#2066](https://github.com/huggingface/tokenizers/pull/2066)
  is an **open PR** fixing it — not an issue. See docs/upstream-issues.md.

---

## 3. What published trainer benchmarks report

Audited 2026-08-05 against a checklist drawn from this project's own failures
(docs/CORRECTIONS.md). gigatrain's pre-August benchmark is included as a row,
because excluding it would be the kind of thing this audit exists to catch.

| Benchmark | mem | parity | variance | bytes | hw | threads | real data | repro |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YouTokenToMe | ✗ | ✗ | ✗ | ✓ | ✓ | **✗** | ✓ | ✓ |
| fast-bytelevel-bpe-go | ✗ | **✓** | ✗ | **✗** | ~ | ✗ | ✓ | ~ |
| **ffbpe** | **✓** | **✓** | ~ | ✓ | ✗ | ✗ | ✓ | ✓ |
| gigatoken | — | — | — | — | — | — | — | — |
| rustbpe | — | — | — | — | — | — | — | — |
| SentencePiece (sp#366) | ✗ | n/a | ✗ | ✓ | ✗ | ~ | ? | ✗ |
| **gigatrain, pre-Aug 2026** | ✓ | ✓ | **✗** | ✓ | ✓ | **✗** | **~** | ✓ |

— = publishes no trainer benchmark at all. gigatoken's README benchmarks are
encoding throughput; rustbpe claims only "fast training with parallel
processing" with no numbers.

**Tallies, over the five scored benchmarks** (four external plus gigatrain's
own pre-August one): **peak memory 2/5, output correctness 3/5, timing
variance 0/5** — in a field whose canonical failure (#1681, #1795, #1824,
sentencepiece#1021) is OOM. Three of the four external benchmarks do not
measure the axis that is failing, and two further tools (gigatoken, rustbpe)
publish no trainer benchmark at all.

**The thread-count defect is in the most-cited benchmark — but it cost them
little.** YouTokenToMe states its hardware (36-core Xeon) and its own thread
count ("YouTokenToMe used 4 threads"), notes SentencePiece and fastBPE are
single-threaded — and never states HuggingFace's, while scanning threads for
itself only. That is a real reporting gap, and gigatrain had the identical
defect until August 2026.

**However, their numbers reproduce.** Their benchmark reports 25.4 s for
YouTokenToMe and 97.7 s for HuggingFace on 1 GB English at 36 cores. Run here
on 16 cores: **26.3 s** and, for HF under *whitespace* — the config comparable
to their HF baseline — **101.1 s** (the same session's HF **ByteLevel** row is
46.9 s; see BENCHMARKS.md, "One-session comparison"). An earlier draft of this section implied their HF baseline was
materially inflated by the unreported thread setting; the measurement says
otherwise, consistent with the 1 GB core-count curve being shallow (1.34x at
36 threads).

So the honest claim is narrower still: the number is **not attributable from
what they published** — a reader cannot reconstruct it — but it appears to be
**substantially correct**. Unreported configuration is a reproducibility
problem here, not an accuracy one.

**The most rigorous benchmark is the least known.** ffbpe gates timings on
configuration match — *"Timing results are informational unless the input,
model, configuration, environment, and output fingerprints all match"* — and
records output SHA-256 fingerprints, model parity, deterministic repeats and
RSS. A stronger contract than this project's. It was scored 2/8 from its README
and 6/8 after reading its `BENCHMARKS.md`; scoring from READMEs understates
projects.

**Candour and comparability are orthogonal.** The two most carefully caveated
benchmarks are the two whose numbers cannot be reused: fast-bytelevel-bpe-go
states no corpus size in bytes, and ffbpe's headline compares two of its own
configurations rather than two trainers.

**Limitations.** Six is the population, not a sample. The YouTokenToMe estimate
is inference, not measurement — their benchmark was not rerun. Only the
surfaces in the manifest were read.

---

## 4. Honest positioning

> gigatrain trains a 32k BPE vocabulary on 12.9 GB of FineWeb in 38 s against
> HuggingFace's 257 s, same ByteLevel pretokenizer, with all 31,790 merges
> byte-identical; and 19.4 GB in 2.9 GB of RAM where HuggingFace needs 36.3 GB.
> The algorithm is standard; the contribution is phase-1 architecture, memory
> layout, and a parity specification that does not exist elsewhere.

**Do not claim:** novelty of the algorithm, being first to HF-parity training
(gigatoken), reproducing #1313, or an *unqualified* "fastest BPE trainer".
"Fastest of the seven, on web text" is supportable since 2026-08-07 (all seven
are in the one-session table) but only with its caveats: just the HF rows have
verified identical output, and rustbpe is ~15% faster on single-giant-pretoken
corpora.

Possibly the strongest artifact is [PARITY.md](PARITY.md) itself — HF's exact
semantics, including `limit_alphabet` nondeterminism, the `max_token_length`
asymmetry, `i32` overflow in both directions, and the duplicate-merge path, are
documented nowhere else including HuggingFace's own docs.

## 5. Open

- ~~Run ffbpe and YouTokenToMe in the one-session table~~ **done 2026-08-07.**
  All seven trainers are now in one comparable table; gigatrain is fastest at
  both sizes. YouTokenToMe is second-fastest at 1 GB but uses **6.4 GB of RAM
  for a 1 GB corpus**, 12x gigatrain's and the highest of any trainer measured
  — a cost invisible in the literature, since no published trainer benchmark
  reports memory.
- gigatoken at 12.9 GB is unmeasured.
- SentencePiece's five degenerate-corpus failures are uncharacterised beyond
  the refusal case.
