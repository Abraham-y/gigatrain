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

Whether gigatoken's trainer survives a 12.9 GB corpus is **unverified** — it
is an inference from the monotonically growing index, and should be measured
before being claimed.

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

**4. SentencePiece got much faster last month.**

v0.2.2 (2026-07-12) landed an O(log n) lazy-priority-queue BPE optimization
(commit `8db9878`, 2026-06-12), with the maintainer claiming **20x**. Any
benchmark against an older SentencePiece is invalid.

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

**6. Parity is currently scoped to whitespace pretokenization.**

Production BPE tokenizers almost universally use ByteLevel with a GPT-2-style
regex, which is what gigatoken's parity test covers and this one does not.
`--words-tsv` can accept externally pretokenized counts, but that bypasses
the fast phase 1 that is the actual contribution. This is the first gap a
reviewer will find.

## Landscape

| Tool | Lang | Trains | Best published training number | Largest demonstrated | HF parity | Status |
|---|---|---|---|---|---|---|
| HF `tokenizers` | Rust | yes | 1 GB in 59–98 s | OOM at 20–50 GB on 1–2 TB RAM | is the reference | 0.23.1; no trainer perf work 2024–2026 |
| gigatoken `train_bpe` | Rust | yes | 1 MB synthetic | ~1 MB | **yes, CI-tested at 120 KB** | 0.10.0, undocumented |
| YouTokenToMe | C++ | yes | 1 GB in 25.4 s | 13 GB in <10 min (user report) | no | **archived 2024** |
| SentencePiece BPE | C++ | yes | 1 GB in 344 s (pre-v0.2.2) | 31.2 GB → 1.8 TB RAM, >24 h | no, by design | v0.2.2, July 2026 |
| SP `contrib/nlcodec` | C++ | yes | 24x over SP default | few hundred MB | no (99% vocab overlap) | opt-in, not in wheel |
| `karpathy/rustbpe` | Rust | yes | ~2 B chars, vocab 65k, ~1 min | ~2 GB | no claim | active |
| tiktoken, rust-gems `bpe`, GPUTOK, BlockBPE | — | **no** | encoding only | — | n/a | — |

## Honest positioning

What is defensible:

> gigatrain trains a 32k BPE vocabulary on 12.9 GB of FineWeb in 86 s and
> 6.4 GB peak RSS on a 10-core laptop, with a merge list byte-identical to
> HuggingFace `tokenizers` under a CI parity test that covers alphabet
> construction, ID reuse, `min_frequency`, `max_token_length`, and
> stale-heap semantics. The algorithm is standard; the contribution is
> phase-1 architecture and memory layout, in a regime where the incumbents
> are reported to OOM.

What should not be claimed: novelty of the algorithm, being the first with HF
parity, or reproducing #1313.

Possibly the strongest artifact here is [PARITY.md](PARITY.md) itself. HF's
exact semantics — `limit_alphabet` nondeterminism, the `max_token_length`
asymmetry between initial and delta counting, `i32` count overflow, the
reachability of the duplicate-merge path — are documented nowhere else,
including HuggingFace's own documentation.

## Open questions

- Does gigatoken's trainer actually fail at 12.9 GB? (Source inference only.)
- How does gigatrain compare to SentencePiece **v0.2.2** and to
  `karpathy/rustbpe`? Not yet measured. Note SentencePiece's
  `--max_sentence_length` defaults to 4192 bytes and silently drops longer
  documents, which invalidates naive comparisons.
