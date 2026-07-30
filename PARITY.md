# HuggingFace BpeTrainer: exact semantics for parity

Determined by reading `tokenizers` source at tag `v0.22.2` (verified identical
to master for the algorithm; master only adds progress reporting). File
references are to `tokenizers/src/models/bpe/`.

## Vocabulary ID assignment

IDs are assigned in this order (`trainer.rs, do_train`):

1. Special tokens, in the order given, deduplicated (`add_special_tokens`).
2. Alphabet characters, **sorted by Unicode codepoint** (`compute_alphabet`,
   `kept.sort_unstable_by_key(|k| *k.0 as u32)`).
3. Merged tokens, in merge-creation order — except that if the merged *string*
   already exists in the vocab, its existing ID is **reused** and no new ID is
   allocated.

The alphabet is computed over **chars** (Unicode scalar values), not bytes,
weighted by word frequency. `initial_alphabet` chars get count `usize::MAX`.

`limit_alphabet`: removes the `len - limit` lowest-count chars via an
**unstable sort by count only** — ties at the cutoff are nondeterministic *in
HF itself* (hash-map iteration order). Parity CI must not use `limit_alphabet`
with count ties at the cutoff. Chars dropped from the alphabet are silently
skipped when tokenizing words.

## Merge selection (the tie-break rule)

Max-heap over `Merge { pair, count, pos }` with (`trainer.rs`, `Ord for Merge`):

- Primary: higher `count` wins.
- Tie: **smaller `pair` wins**, comparing `(u32, u32)` vocab-ID tuples
  lexicographically (`other.pair.cmp(&self.pair)` — note the reversal).

Since alphabet IDs are codepoint-ordered and merge IDs are creation-ordered,
this means: earliest-created left token, then earliest-created right token.

## The merge loop (`do_train`, step 5)

```
loop:
    stop if vocab_len >= vocab_size          # checked BEFORE popping
    top = heap.pop()  (stop if empty)
    if top.count != pair_counts[top.pair]:   # stale entry:
        top.count = live count               #   correct it and RE-PUSH
        heap.push(top); continue             #   (NOT discarded)
    stop if top.count < 1 or top.count < min_frequency   # break, not skip
    build new token string; REUSE existing ID if string already in vocab
    merges.push((pair, new_id))              # duplicates possible, see below
    for each word index in top.pos: word.merge(...) -> delta changes
    apply changes to pair_counts; where_to_update[pair] += {word} if delta > 0
    drain where_to_update: push fresh heap entry (pair, live count, new pos)
        only if live count > 0
```

Machinery that must be replicated structurally (but see reachability note):

- **No self-decrement.** `Word::merge` never emits a change for the merged
  pair itself, so `pair_counts[merged_pair]` keeps its full stale value
  forever. Only observable if the pair could re-form later.
- **Partial position sets.** A heap entry's `pos` holds only the word indices
  recorded when that entry was pushed. If a pair gained occurrences at two
  different steps, it would have two entries with partial `pos` sets, and
  merging via one would not touch the other's words — producing a
  **duplicate entry in the merges list** when the second one pops.
- **Duplicates collapse at serialization.** `model.merges` is a map keyed by
  pair; a duplicate merge overwrites the rank (keeps the LAST rank).
  `tokenizer.json` emits unique pairs sorted by that rank
  (`serialization.rs`). Parity is defined against this serialized list.

**Reachability**: we believe the duplicate/partial-pos paths are dead code in
real training. Every merge applies to its complete position set, so a pair
(x, yz) gains occurrences only at the single step where its constituent token
is created; by induction no pair ever gains count after its creation step, no
pair re-forms after being merged, and every heap entry's `pos` is complete.
(The only base-case escape would be new-token string reuse, which itself
requires a prior re-formation — circular.) Consistent with this, 2,500+
fuzz trials on run-heavy tables produced zero duplicate merges. Special
tokens colliding with merge strings (e.g. special "th" over English text)
exercise the ID-reuse lookup but still produce no duplicates; this case is
covered by a parity test. We replicate the machinery anyway so parity does
not depend on this analysis being right.

## Word::merge (`word.rs`)

Left-to-right scan, non-overlapping ("aaa" -> ["aa","a"]). For each occurrence
merged, emits (order within a word does not matter — all deltas are additive
and position-set inserts are idempotent):

- If a left neighbor exists (post-splice, i.e. possibly a symbol merged
  earlier in this same pass): `(left, c1) -= 1`, and `(left, new) += 1` **only
  if** `left.len + new.len < max_token_length` (strict `<`, lengths in chars).
- If a right neighbor exists (the symbol after the pair): `(c2, right) -= 1`,
  and `(new, right) += 1` under the same length guard.

`max_token_length` filters only these newly-formed pairs. The **initial** pair
count (`count_pairs`) applies no length filter, so 2-char tokens always form
regardless of `max_token_length`.

## Numeric types

HF keeps `pair_counts` as `i32` and casts word counts `u64 as i32` — it
overflows on huge corpora (where HF cannot finish training anyway). We use
`i64` internally; parity holds wherever HF itself doesn't overflow. Negative
live counts cast `as u64` sign-extend identically in both.

## Pretokenization reference points

- `WhitespaceSplit` = split on `char::is_whitespace`, delimiters removed.
  Rust's `str::split_whitespace` is byte-for-byte equivalent.
- `Whitespace` = regex `\w+|[^\w\s]+` (Unicode classes, `regex` crate).
- BPE pipelines have no normalizer unless configured; chars pass through
  as-is.

## Known HF-side nondeterminism (excluded from parity claims)

- `limit_alphabet` with count ties at the cutoff (unstable sort over hash
  order).
- Heap pop order between two entries with equal (count, pair) but different
  `pos` — outcome-equivalent after dedup; the serialized merge list is
  deterministic, but transient internal state may differ.
