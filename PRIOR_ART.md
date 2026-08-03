# Prior art

Compiled 2026-07-30. Every claim here was checked against a primary source;
where a claim is inference rather than measurement, it says so. This file
exists to keep the project's claims honest, so it leads with the findings
that undercut them.

## Things that contradict the obvious pitch

**1. gigatoken already has a BPE trainer with HuggingFace parity.**

[gigatoken](https://github.com/marcelroed/gigatoken) is best known as an
encoder (~24 GB/s), but `src/bpe_train.rs` exposes:

```rust
pub fn train_bpe<K: AsRef<[u8]> + Eq + Hash>(
    counts: HashMap<K, usize, FxBuildHasher>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: TieBreaking,
) -> BPEResult
```

`TieBreaking::HuggingFace` is the default, with a `build_byte_to_hf_rank()`
that remaps byte IDs to HF's ByteLevel ordering, and
`tests/test_bpe_train_compare.py::test_merges_identical` asserts the merge
list matches `trainers.BpeTrainer` position-for-position. It is exported to
Python and shipped in 0.10.0 (PyPI, 2026-07-25). It is not mentioned in the
README or on PyPI.

So "nothing trains with HF parity" is false, and framing gigatrain as a
non-competing sibling to gigatoken is no longer accurate.

Where gigatrain still differs (verified by reading the source):

| | gigatoken `train_bpe` | gigatrain |
|---|---|---|
| parity test scale | ~120 KB synthetic, vocab 500 | FineWeb, vocab 32k |
| parity scope | tie-break; hardcodes the 0–255 alphabet | alphabet construction, ID reuse, `min_frequency`, `max_token_length`, `limit_alphabet`, stale re-push, duplicate-merge machinery |
| largest published training benchmark | 1 MB synthetic, vocab 2000 | 12.9 GB FineWeb |
| inverted index | `HashMap<(u32,u32), BTreeSet<u32>>`, stale entries never removed (its own comment: *"we don't remove old ones, though they will be stale"*) | arena + hash-sharded phase 1 |
| word representation | `Vec<u32>` symbols per word | one flat arena of `u32` for all words |
| known debt | `TODO(perf)`: *"a lot of contention on this map early in merging"* | — |

**Correction (2026-07-31): gigatoken's trainer was measured, and the earlier
inference here was wrong.** This file previously guessed, from its
monotonically growing `BTreeSet` index and `Vec::remove`, that it "will very
likely not survive 12.9 GB". It is in fact fast and memory-efficient — by a
wide margin the closest competitor:

| corpus | gigatoken | gigatrain (ByteLevel) |
|---|---|---|
| 100 MB | 5.26 s / 259 MB | 3.07 s / 216 MB |
| 1 GB | 18.76 s / 1166 MB | 10.22 s / 683 MB |

(10-core macOS; gigatoken's figure is its own reported train time, which
excludes process startup, so if anything it flatters gigatoken.) So
gigatrain is **~1.8x faster on ~1.7x less memory** — not the order of
magnitude that separates it from rustbpe, SentencePiece and HF.

**Where gigatoken's parity actually breaks.** Its merges were compared
against HF's `BpeTrainer` rank-for-rank, after mapping its raw bytes through
the byte-to-unicode table:

| corpus | line endings | identical at same rank |
|---|---|---|
| FineWeb sample | LF | **2744 / 2744** (its full output) |
| War and Peace | CRLF | **21 / 2744**, diverging at rank 0 |

On LF corpora it matches HF exactly. On CRLF corpora it diverges at the very
first merge: HF's trainer feeds files a line at a time, so a trailing `\r\n`
is terminal within its line and yields the token `čĊ` — HF's rank-0 merge.
gigatoken pretokenizes whole-text and never forms it (5450 unique pretokens
against HF's 5441 on the same file).

That is the same trap documented in PARITY.md and caught by this project's CI,
which trains on a CRLF corpus precisely because of it. gigatoken's own parity
test uses in-memory synthetic sentences at ~120 KB, where the case cannot
arise.

So the honest comparison with gigatoken is: **faster and leaner, but not by an
order of magnitude; the real difference is parity scope and validation
scale.**

**2. The "core algorithmic insight" is the status quo.**

Incremental pair counts + inverted index + lazy max-heap is not a new idea
and is not a differentiator. Sennrich et al. 2015
([arXiv:1508.07909](https://arxiv.org/abs/1508.07909) §3.2) states the
*incremental* half only — "we increase efficiency by indexing all pairs, and
updating data structures incrementally"; the words "heap" and "priority" do
not appear in that paper. The heap, and the O(NM) → O(N log M) proof, are
Zouhar et al., *A Formal Perspective on Byte-Pair Encoding*
([arXiv:2306.16837](https://arxiv.org/abs/2306.16837), Findings of ACL 2023,
§4/Thm 4.2). Both are already implemented in:

- HuggingFace `tokenizers` — `where_to_update: AHashMap<Pair, AHashSet<usize>>`,
  delta updates, `dary_heap::OctonaryHeap` with stale re-push, rayon-parallel
  pair counting and merge application. Its `word.rs` even uses a
  `Symbol { prev, next, len }` doubly-linked list over a `Vec` — the "future
  work" layout.
- SentencePiece — position sets and a lazy priority queue.
- `karpathy/rustbpe` — same index, same heap, same tie-break.

A naive-recount baseline is therefore a strawman and should not be quoted as
a speedup.

**3. Issue #1313 is not the workload it looks like — and both headline
issues are closed.**

#1313 was closed 2023-12-19 as stale/not-planned. #1681 was closed 2025-05-27
pointing at a `cache capacity` workaround that commenters then established
does not apply to training at all; a reopen was requested in July 2025 and
never granted. "Closed without a fix, over unanswered objections" is both more
accurate and a stronger story than "open issue". The live ones are
[#1795](https://github.com/huggingface/tokenizers/issues/1795) (100 GB of RAM
supports only ~1.5 GB of Chinese JSONL) and
[#1824](https://github.com/huggingface/tokenizers/issues/1824) (131k vocab
exceeded 750 GB).


[tokenizers #1313](https://github.com/huggingface/tokenizers/issues/1313):
~13 billion characters, but **`vocab_size=512`** on DNA-like data with no
pre-tokenizer. At vocab 512 there are only a couple of hundred merges, so
the merge loop is nearly free; the 10+ hours came from degenerate
pretokenization, not merge cost. A 32k-vocab FineWeb run does not reproduce
it. The genuinely unanswered scale issues are the memory ones: #1681
(20 GB OOM on 1.5–2 TB), #1795, #1824.

**4. SentencePiece got much faster last month — and is still slower than HF.**

v0.2.2 (2026-07-12) landed an O(log n) lazy-priority-queue BPE optimization
(commit `8db9878`, 2026-06-12), with the maintainer claiming **20x**. That
20x is against older SentencePiece, not against the field.

Measured here on FineWeb, vocab 32000, same machine (`scripts/sp_train_cli.py`,
with `--max_sentence_length` raised from its 4192-byte default so documents
are not silently truncated, `train_extremely_large_corpus=true`,
`normalization_rule_name=identity`, `character_coverage=1.0`):

| corpus | SentencePiece v0.2.2 | HF 0.22.2 | gigatrain |
|---|---|---|---|
| 100 MB | 13.7 s / 539 MB | 9.7 s / 1.0 GB | 1.7 s / 419 MB |
| 1 GB | 112.7 s / 3.0 GB | 61.2 s / 4.7 GB | 9.4 s / 1.3 GB |

So post-optimization SentencePiece is roughly **2x slower than HuggingFace**
at these sizes and ~12x slower than gigatrain, though it uses less memory
than HF. Its CPU time barely exceeds wall time (17.2 s user vs 14.0 s real at
100 MB), consistent with the maintainer's statement that BPE training is
single-threaded.

**This is a speed and memory comparison only, not a parity one.**
SentencePiece BPE deliberately produces a different tokenizer: it normalizes,
derives its alphabet from character coverage, prefixes words with U+2581, and
emits pieces rather than a merge list. There is no meaningful merge-for-merge
diff to run against it.

At 12.9 GB SentencePiece was stopped after ~8.5 minutes wall while still
normalizing the corpus — it had not begun counting pairs, and had pushed the
machine to 28 GB of swap. That is an incomplete measurement, not a timeout
result, and is reported as such.

**7. rustbpe is the closest competitor.**

[karpathy/rustbpe](https://github.com/karpathy/rustbpe) is Rust, rayon
parallel, and uses the GPT-4 split pattern with fancy-regex. Measured on the
same machine and corpora (`scripts/rustbpe_train_cli.py`, fed line by line):

| corpus | rustbpe | gigatrain (ByteLevel) |
|---|---|---|
| 100 MB | 9.9 s / 343 MB | 1.5 s / 278 MB |
| 1 GB | 78.9 s / 1.20 GB | 14.9 s / 747 MB |

Roughly **5–7x slower with comparable memory** — much closer than
SentencePiece, and the only competitor whose memory profile resembles
gigatrain's. It makes no HF-parity claim and uses a different split pattern,
so again this is speed and memory only.

Separately — and this is **taku910, SentencePiece's maintainer**, not
rustbpe's — SentencePiece's BPE merge loop was measured as ~76% sequential
priority-queue maintenance, with parallelization capping out around 1.3x by
Amdahl, and YouTokenToMe-style gains judged to need *"a complete rewrite of
the core BPE structures"*
([#366](https://github.com/google/sentencepiece/issues/366), 2026-06-16). The
"compact arrays, custom hash maps, split queues" phrasing in that thread is
him describing **YouTokenToMe's** architecture, not prescribing gigatrain's.

That is still useful: the incumbent's maintainer independently concluding that
the sequential fraction bounds what parallelism can buy, which is the same
reason CLAUDE.md rules out a GPU. But it is corroboration of the *constraint*,
not an endorsement of this design. (An earlier version of this file attributed
the analysis to rustbpe's maintainer and read it as the latter.)

**5. HuggingFace is not catastrophic at 1 GB.**

Independent numbers put HF at 59 s (arXiv:2604.05192) and 97.7 s (YTTM's
benchmark) on 1 GB, consistent with the 61.2 s measured here. The advantage
at that scale is ~6x, not orders of magnitude.

**6. Parity was scoped to whitespace pretokenization — now closed.**

Production BPE tokenizers almost universally use ByteLevel with a GPT-2-style
regex. That gap is closed: `--pretokenizer bytelevel` trains
merge-list-identical to HF, with the pretokenizer itself diffed against HF
over every non-surrogate BMP codepoint in 8 contexts plus real corpora.

Two subtleties that had to be handled, both of which silently corrupt output
if missed: Unicode class tables must match HF's regex version rather than a
Unicode database (Python 3.12 is Unicode 15, HF's regex is Unicode 16), and
HF's trainer feeds files one line at a time, so a trailing newline is
terminal within its line (`"x\r\n"` is one token, not two).

## Fresh sweep, 2026-07-31

**ffbpe** ([tokn-ai/ffbpe](https://github.com/tokn-ai/ffbpe)) is the only
other project with a bounded-memory *exact* trainer — i.e. it is attacking the
same problem. It is **not** new: the repo was created 2025-12-07 and v0.1.1
shipped 2025-12-18; v0.1.8 (2026-07-27) is simply the first release after a
rename from `unitoken`. It predates gigatrain by about seven months.

Its README's "Measured impact" headline is a 64 MiB Chinese fixture going from
26.681 s to 3.702 s. The **5.58 s** figure sometimes quoted for 1 GiB is not a
headline claim at all — it is the *before* value in a bounded-memory RSS
comparison ("training changed from 5.58 s to 5.85 s").

Measured under the same contract as everything else in BENCHMARKS.md (raw
text in, vocab 32000, English FineWeb, 10-core macOS):

| corpus | ffbpe | gigatrain (ByteLevel) |
|---|---|---|
| 100 MB | 8.70 s / 480 MB | 3.07 s / 216 MB |
| 1 GB | 65.43 s / 1513 MB | 10.22 s / 683 MB |

So **~6x slower end to end**, not faster. Their 5.58 s is a different
measurement: vocab 10,000, Chinese text, and counting begins from a
precomputed Unicode-bigram inventory, so pretokenization and word counting are
excluded. Their own docs say the bundled HF comparison "is not a pure
trainer-algorithm comparison". Nothing dishonest — it measures their
compressed-inventory contract — but it is not comparable to a raw-text number,
and it should not be cited as one in either direction. Their bounded-memory
mode is a genuine feature this project does not have.

**fast-bytelevel-bpe-go**
([yunnian/fast-bytelevel-bpe-go](https://github.com/yunnian/fast-bytelevel-bpe-go),
Go, MIT, created 2026-06-26 — the genuinely newest entrant) is the only other
project that verifies **byte-exact HF parity**: it reports `vocab: SAME /
merges: SAME` at vocab 32,779 against tokenizers 0.23.1. On the parity axis it
is the nearest peer and should be cited rather than ignored.

**Its speed cannot be compared to gigatrain's.** Its `docs/benchmarks.md`
reports 739.59 s against HF's 4406.97 s on "the first 846,882 non-empty `text`
rows" of a JSONL file, on Apple Silicon, `min_frequency=2` — **no corpus size
in bytes is stated anywhere**. So the ~6x-faster-than-HF ratio is theirs to
claim, but any cross-comparison to gigatrain is unfounded. An earlier version
of this file asserted "roughly two orders of magnitude slower than gigatrain",
which had no basis and sat badly beside gigatrain's own ~6x-over-HF figure.

**GPU BPE training still does not exist.** Everything branded "GPU BPE"
(BlockBPE, GPUTOK, RAPIDS `nvtext::byte_pair_encoding`) is *encoding* and
requires a pre-trained merge table. The only GPU trainers are hobby-scale and
abandoned: `evintunador/gpu_bpe` (vocab capped at 2^16-2, dead since
2025-05), `kuprel/minbpe-pytorch` (vocab 512, no regex pretokenization, dead
since 2024-02). CLAUDE.md's "do NOT use a GPU" is now empirically supported
rather than merely argued.

**No major lab trains its own BPE.** OLMo 3 reuses OLMo 2's cl100k-derived
vocab; gpt-neox wraps HF `trainers`; Llama 4, DeepSeek, Mistral and Qwen 3
ship no trainer code at all. NVIDIA's NeMo Curator docs state plainly that it
"doesn't handle tokenizer training". The demand argument has to rest on
non-English and domain-specific builders, not on frontier labs.

**Two HuggingFace trainer bugs that bear on any parity claim:**

- [#2058](https://github.com/huggingface/tokenizers/issues/2058): pair counts
  are `i32` and wrap negative on large corpora, silently corrupting merge
  order. PARITY.md already notes this crate uses `i64`; parity therefore holds
  only where HF itself has not overflowed.
- [#1794](https://github.com/huggingface/tokenizers/issues/1794) (open issue):
  HF BPE/WordPiece training is **non-deterministic run to run**, because token
  ids assigned in `AHashMap` order feed the tie-break comparator.
  [#2066](https://github.com/huggingface/tokenizers/pull/2066) is an **open
  pull request** (not an issue, as this file previously said) that isolates
  exactly that cause and fixes it by sorting the word counts before assigning
  ids; it is unmerged, and the bug is still present in 0.23.1. Note that its
  ordering and gigatrain's are *different* deterministic orders, so a merged
  #2066 would still not agree merge-for-merge with gigatrain's decorated
  modes. Our parity suite has been stable across many runs against 0.22.2, but
  this is a reason to pin the HF version in CI (it is) and to state which
  behaviour is matched.

**No benchmark suite for trainers exists.** The only one
([YouTokenToMe's](https://github.com/VKCOM/YouTokenToMe/blob/master/benchmark.md))
was archived read-only in 2024, reports no memory and no parity, and stops at
1 GB. Everything else called a "tokenizer benchmark" measures encoding. The
published HF figure for 1 GB varies wildly by source — 59 s (arXiv:2604.05192),
97.7 s (YTTM, 36 cores), 244.4 s (measured here, 64 cores).

**This is not triangulation and must not be presented as such.** The three
figures use different corpora, different vocabulary sizes and different
machines: the 59 s is a one-line aside on MiniPile with neither hardware nor
vocabulary stated, in a preprint under review; the 97.7 s is YTTM's Wikipedia
benchmark at vocab 30,000 on 36 cores; the 244.4 s is FineWeb at vocab 32,000
on 64 cores. They are *consistent with* HF slowing at higher core counts, and
nothing more. A harness reporting wall time, peak RSS **and** merge parity
across core counts may be a contribution in its own right.

## Landscape

| Tool | Lang | Trains | Best published training number | Largest demonstrated | HF parity | Status |
|---|---|---|---|---|---|---|
| HF `tokenizers` | Rust | yes | 1 GB in 59–98 s (61.2 s measured here) | OOM at 20–50 GB on 1–2 TB RAM; did not finish 12.9 GB in 60 min here | is the reference | 0.23.1; actively maintained (see note) |
| gigatoken `train_bpe` | Rust | yes | 1 MB synthetic | ~1 MB | **yes, CI-tested at 120 KB** | 0.10.0, undocumented |
| YouTokenToMe | C++ | yes | 1 GB in 25.4 s | 13 GB in <10 min (user report) | no | **archived 2024** |
| SentencePiece BPE | C++ | yes | **1 GB in 112.7 s / 3.0 GB (v0.2.2, measured here)** | 31.2 GB → 1.8 TB RAM, >24 h | no, by design | v0.2.2, July 2026 |
| SP `contrib/nlcodec` | C++ | yes | 24x over SP default | few hundred MB | no (99% vocab overlap) | opt-in, not in wheel |
| `karpathy/rustbpe` | Rust | yes | **1 GB in 78.9 s / 1.20 GB (measured here)** | ~2 GB | no claim | active |
| tiktoken, rust-gems `bpe`, GPUTOK, BlockBPE | — | **no** | encoding only | — | n/a | — |

**Note on HF maintenance.** An earlier version of this table said "no trainer
perf work 2024–2026". That is false, and it mattered because it framed the
baseline as abandoned. `tokenizers/src/models/bpe/trainer.rs` has had at least
two performance commits in that window: *Convert word counts to u64* (#1433,
2024-02-06) and *Consolidated optimization ahash dary compact str* (#1799,
2025-06-19) — the latter being exactly the `ahash` / `dary_heap` /
`CompactString` machinery PARITY.md describes. HF is a moving, maintained
baseline, which is the correct reason to pin a version in CI.

## Honest positioning

What is defensible:

> gigatrain trains a 32k BPE vocabulary on 12.9 GB of FineWeb in 85 s and
> 2.2 GB peak RSS on a 10-core laptop, with a merge list byte-identical to
> HuggingFace `tokenizers` under a CI parity test covering both whitespace
> and ByteLevel pretokenization, alphabet construction, ID reuse,
> `min_frequency`, `max_token_length`, and stale-heap semantics. On the same
> machine and file, HuggingFace did not finish in 60 minutes and
> SentencePiece v0.2.2 is ~12x slower at 1 GB. The algorithm is standard;
> the contribution is phase-1 architecture and memory layout, in a regime
> where the incumbents are reported to OOM.

What should not be claimed: novelty of the algorithm, being the first with HF
parity, or reproducing #1313.

Possibly the strongest artifact here is [PARITY.md](PARITY.md) itself. HF's
exact semantics — `limit_alphabet` nondeterminism, the `max_token_length`
asymmetry between initial and delta counting, `i32` count overflow, the
reachability of the duplicate-merge path — are documented nowhere else,
including HuggingFace's own documentation.

## A note on gigatrain's own numbers in this file

**The gigatrain column differs between the comparison tables above, and the
multipliers are therefore not comparable across them.** At 1 GB ByteLevel on
the 10-core laptop this repo has recorded 8.5 s (BENCHMARKS.md), 10.22 s (the
gigatoken and ffbpe tables here) and 14.9 s (the rustbpe table here); at
100 MB, 1.2 s / 3.07 s / 1.5 s. Each competitor was benchmarked in its own
paired session, under different background load — and, as BENCHMARKS.md's
"Measurement noise" section records, an occupied swap file moved identical
configurations by up to 2x.

So each table is internally valid as a same-session pairing, and none of the
ratios should be compared *to each other*. Saying "rustbpe is 5–7x slower
while gigatoken is only 1.8x" silently compares a 14.9 s baseline against a
10.22 s one. Re-running all competitors in one session on a quiet machine is
the fix, and it has not been done.

## Open questions

- **Re-run all competitors in a single session** so the gigatrain baseline is
  one number rather than three (see above). This is the most valuable
  outstanding benchmark task.
- SentencePiece at 12.9 GB is **not** a completed measurement: it was stopped
  after ~8.5 minutes wall (~12 min CPU) while still in corpus normalization,
  having driven the machine to 28 GB of swap. It never reached the merge loop.
  Worth re-running on a machine with enough RAM to finish. (It *did* complete
  on the 64-core box, where it segfaulted at 20 GB resident after 135 s.)
- gigatoken's trainer has been measured at 100 MB and 1 GB but **not** at
  12.9 GB. The earlier "will very likely not survive 12.9 GB" inference was
  wrong at the sizes tested and is retracted; the large-corpus behaviour is
  simply unknown, not suspect.
