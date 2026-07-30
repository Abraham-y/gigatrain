# gigatrain architecture

Zero-dependency Rust. Two phases with opposite characters: phase 1 is
embarrassingly parallel and I/O bound, phase 2 is strictly sequential and
memory bound. See PARITY.md for the HF semantics both must reproduce, and
BENCHMARKS.md for measured results.

## Phase 1: corpus to word-frequency table

Three stages, connected by bounded channels:

```
reader ──chunks──> scanners ──packed batches──> shard owners
(1 thread)         (N threads)                  (N threads)
```

- **reader** streams each file in 4 MB chunks, cutting at the last ASCII
  whitespace byte and carrying the remainder forward. ASCII whitespace is
  always a real word boundary: UTF-8 continuation bytes are never ASCII, and
  every ASCII whitespace char satisfies `char::is_whitespace`. Nothing is ever
  fully resident.
- **scanners** split each chunk into words (`split.rs`), hash each word once,
  and route it to shard `hash % N`, packing into per-shard `WordBatch`
  buffers shipped at 64 KB.
- **shard owners** count their shard into a `WordCounter`. Shards are disjoint,
  so each unique word is stored exactly once machine-wide and the final
  combine is a concatenation with no lookups and no merge.

Two simpler designs were measured and rejected; the rationale, with numbers,
is in the doc comment on `count_words_parallel` and in BENCHMARKS.md. The
short version: per-worker maps are fast but store every frequent word N times
and then merge single-threaded; broadcasting chunks fixes memory but
replicates scanning and hashing per worker, leaving a fixed Amdahl floor.

`split.rs` is a byte-level whitespace splitter, tested equivalent to
`str::split_whitespace` across every BMP codepoint. It fast-paths ASCII and
only decodes UTF-8 when it sees one of the four lead bytes (0xC2, 0xE1, 0xE2,
0xE3) that can begin a whitespace char.

## Phase 2: the merge loop

Sequential by construction — merge N+1 depends on merge N — so this is about
memory layout, not parallelism.

- `WordTable` (`wordtable.rs`) holds words in one byte arena with u32 offsets.
  The trainer takes it **by value** and drops it the moment words are
  tokenized into IDs, so word strings do not stay resident through the loop.
- `WordCounter` (`counter.rs`) is that table plus a `hash -> id` index with
  intrusive collision chains, so counting costs no per-word allocation.
- `WordArena` (`word.rs`) stores all words as one flat `Vec<u32>` of token
  IDs with per-word (start, len). **A merge only ever shrinks a word**, so
  slice starts are fixed for the whole run: the arena never reallocates and
  never needs compaction.
- HF's per-symbol `len` field is gone. It always equals the char count of the
  symbol's token string, so it lives in a dense `token_chars` table indexed by
  token ID — symbols are bare u32s, halving the arena and the bytes the hot
  scan touches.
- Position lists are `Vec<u32>` (not `HashSet`), deduped on push: within one
  merge step words are processed one at a time, so duplicate inserts for a
  pair are always adjacent. Spent buffers are recycled through a pool.
- Merge selection is a lazy max-heap; stale entries are corrected and
  re-pushed on pop, matching HF exactly (PARITY.md).

## Diagnostics

`GIGATRAIN_STATS=1` prints stage-boundary RSS, structure sizes (words,
symbols, distinct pairs, position entries and capacity), and phase-2
sub-stage timings. This is what showed that phase 1, not the merge loop, held
most of the memory — worth running before optimizing anything here, since the
intuitive answer was wrong.

## Testing

- `cargo test` — unit tests including ports of HF's own trainer and word tests,
  and the splitter equivalence sweep.
- `scripts/run_parity_ci.sh` — the milestone-2 gate. Five corpus
  configurations against real `tokenizers` plus 1000 fuzz trials, all
  compared merge-for-merge. **No trainer change lands without this green.**
