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
and is not a differentiator. It is stated in Sennrich et al. 2015
([arXiv:1508.07909](https://arxiv.org/abs/1508.07909) §3.2), formalized and
proved O(NM) → O(N log M) by Zouhar et al., *A Formal Perspective on
Byte-Pair Encoding* ([arXiv:2306.16837](https://arxiv.org/abs/2306.16837),
Findings of ACL 2023, §4/Thm 4.2), and already implemented in:

- HuggingFace `tokenizers` — `where_to_update: AHashMap<Pair, AHashSet<usize>>`,
  delta updates, `dary_heap::OctonaryHeap` with stale re-push, rayon-parallel
  pair counting and merge application. Its `word.rs` even uses a
  `Symbol { prev, next, len }` doubly-linked list over a `Vec` — the "future
  work" layout.
- SentencePiece — position sets and a lazy priority queue.
- `karpathy/rustbpe` — same index, same heap, same tie-break.

A naive-recount baseline is therefore a strawman and should not be quoted as
a speedup.

**3. Issue #1313 is not the workload it looks like.**

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

The same maintainer measured SentencePiece's BPE merge loop as ~76%
sequential priority-queue maintenance, concluded parallelizing it caps out
around 1.3x by Amdahl, and wrote that YouTokenToMe-style gains would need
*"a complete rewrite of the core BPE structures"* toward *"compact arrays,
custom hash maps, split queues"*
([#366](https://github.com/google/sentencepiece/issues/366)). That is the
incumbent's maintainer describing this project's architecture as the correct
answer and declining to build it — useful corroboration, and also a warning
that the sequential fraction bounds what is achievable.

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

## Landscape

| Tool | Lang | Trains | Best published training number | Largest demonstrated | HF parity | Status |
|---|---|---|---|---|---|---|
| HF `tokenizers` | Rust | yes | 1 GB in 59–98 s (61.2 s measured here) | OOM at 20–50 GB on 1–2 TB RAM; did not finish 12.9 GB in 60 min here | is the reference | 0.23.1; no trainer perf work 2024–2026 |
| gigatoken `train_bpe` | Rust | yes | 1 MB synthetic | ~1 MB | **yes, CI-tested at 120 KB** | 0.10.0, undocumented |
| YouTokenToMe | C++ | yes | 1 GB in 25.4 s | 13 GB in <10 min (user report) | no | **archived 2024** |
| SentencePiece BPE | C++ | yes | **1 GB in 112.7 s / 3.0 GB (v0.2.2, measured here)** | 31.2 GB → 1.8 TB RAM, >24 h | no, by design | v0.2.2, July 2026 |
| SP `contrib/nlcodec` | C++ | yes | 24x over SP default | few hundred MB | no (99% vocab overlap) | opt-in, not in wheel |
| `karpathy/rustbpe` | Rust | yes | **1 GB in 78.9 s / 1.20 GB (measured here)** | ~2 GB | no claim | active |
| tiktoken, rust-gems `bpe`, GPUTOK, BlockBPE | — | **no** | encoding only | — | n/a | — |

## Honest positioning

What is defensible:

> gigatrain trains a 32k BPE vocabulary on 12.9 GB of FineWeb in 104 s and
> 2.4 GB peak RSS on a 10-core laptop, with a merge list byte-identical to
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

## Open questions

- Does gigatoken's trainer actually fail at 12.9 GB? (Source inference only.)
- SentencePiece at 12.9 GB is **not** a completed measurement: it was stopped
  after ~8.5 minutes wall (~12 min CPU) while still in corpus normalization,
  having driven the machine to 28 GB of swap. It never reached the merge
  loop. Worth re-running on a machine with enough RAM to finish.
- gigatoken's trainer at scale is still unmeasured (source inference only).
