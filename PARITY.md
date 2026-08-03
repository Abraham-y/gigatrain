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

**Reachability**: these paths are dead code for *undecorated* training, but
**live** once `continuing_subword_prefix` is set — a merge whose output id
equals one of its inputs (e.g. `('##','##') -> '###'`) has been observed
twice in one raw merge log. The serialized list dedupes, so output is
unaffected, but the machinery is exercised. The argument below applies only
to the undecorated case: Every merge applies to its complete position set, so a pair
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

HF keeps `pair_counts` as `i32` and casts word counts `u64 as i32`. Past
2^31 occurrences of a single pair the counter wraps negative, the pair is
never pushed onto the heap, and HF silently emits **fewer merges** — in the
limit, none at all. This crate uses `i64` and does not.

**This is a real, reachable divergence, not a theoretical one.** Measured: a
corpus with 2,147,850,000 occurrences of one pair makes HF emit zero merges
and a one-token vocabulary, while gigatrain emits three; both are
deterministic, 3/3 runs. HF completed that run in 10-19 seconds, so an
earlier version of this document was wrong to excuse it as happening only
"where HF cannot finish training anyway".

Extrapolating the most frequent FineWeb bigram, the threshold lands around
120-150 GB of English web text — below the scale this project targets. Above
it, gigatrain's output is arguably the correct one, but it is **not** HF's.

### The negative direction, reachable at 38 bytes

The overflow above needs 2^31 occurrences. The **opposite** sign error needs
eight words, and gigatrain must reproduce it.

Under `continuing_subword_prefix` (or `end_of_word_suffix`) combined with
`max_token_length`, a pair count can go **negative**. Merge-time id reuse is
what does it: when a merge produces a token that already has an id — e.g.
`('##', '##c') -> '##c'` — the `-1`/`+1` neighbour deltas no longer cancel,
because the length guard rejects the `+1` while the `-1` still applies. A pair
already at zero is decremented to `-1`.

HF then evaluates `pair_counts[&pair] as u64`. That turns `-1` into
`18446744073709551615`, which passes the staleness re-check, is not `< 1`, and
**wins the heap immediately** — so HF emits an extra merge for a pair that
occurs nowhere in the corpus. gigatrain's `live as u64` on an `i64` sign-extends
to the same value and reproduces this exactly. (`i32` vs `i64` is irrelevant
here: both sign-extend identically for any value in `i32` range.)

Minimal reproducing case, 8 words / 38 bytes:

```
corpus: '##c ##cac# #ab c# #### accc#a#a#b a cb'
--vocab-size 244 --continuing-subword-prefix '##' --max-token-length 4 \
  --special a --special '<unk>'
```

gigatrain emits 15 merges, with `('##c', '##a')` at index 11 — the phantom
merge. Verified against HF over 200 runs: HF produced 8 distinct outputs
(13, 14 or 15 merges, per the decorated-mode nondeterminism below), and the
**13 runs that landed on gigatrain's token ordering matched it exactly**,
phantom merge included.

Only `max_token_length` 4 triggers it on this corpus; 2, 3, 5, 6, 8, 100 and
unset do not. The behaviour is undocumented upstream and is not what any
reading of the API would predict, which is precisely why it is specified here.

## Pretokenization reference points

- `WhitespaceSplit` = split on `char::is_whitespace`, delimiters removed.
  Rust's `str::split_whitespace` is byte-for-byte equivalent.
- `Whitespace` = regex `\w+|[^\w\s]+` (Unicode classes, `regex` crate).
- BPE pipelines have no normalizer unless configured; chars pass through
  as-is.

## Known HF-side nondeterminism (excluded from parity claims)

- **`continuing_subword_prefix` / `end_of_word_suffix` make HF's trainer
  non-deterministic.** Measured directly: three runs of `BpeTrainer` over an
  identical corpus, with `continuing_subword_prefix="##"`, produced three
  different merge lists (first divergence at merge 142) and three different
  vocabularies. With `end_of_word_suffix="</w>"` likewise. With neither set,
  HF is deterministic across runs.

  The cause is that decorated tokens (`##a`, `a</w>`) are created during
  `tokenize_words` while iterating an `AHashMap`, so their ids depend on hash
  order, and those ids feed the tie-break comparator. This is
  huggingface/tokenizers#2066 / #1794, and it means **`WordPieceTrainer` is
  non-reproducible by default**, since it sets `##`.

  gigatrain registers decorated tokens in **sorted** order, before
  tokenization, so its own output is deterministic across runs and thread
  counts. Measured agreement with HF under `##` at vocab 2000: 1806 of our
  1813 merges appear in a given HF run, 1803 appear in all three, and HF
  shares only 1805 of 1813 with itself.

  **Where HF is deterministic, we still differ, and that is a deliberate
  trade.** An earlier version of this document justified the gap by saying
  there is "no single correct answer to match". That is too broad. When
  decorated-token discovery order is forced — most simply, a corpus with one
  unique word — HF's hash-map iteration cannot vary and it has exactly one
  answer, which we do not reproduce. Minimal case: the single word `bccaa`
  with `##`, where HF assigns `##c` before `##a` (first appearance) and we
  assign `##a` before `##c` (sorted), diverging at merge index 1. Fuzzing
  found HF deterministic in 60 of 550 random decorated configs, and we
  differed on 10 of those 60.

  Matching HF here would mean reproducing `AHashMap` iteration order, which
  is not a defined order at all; adopting first-appearance order instead
  would make *our* output depend on the order phase 1 happens to emit words,
  which varies with thread scheduling. Reproducibility was judged worth more
  than matching an arbitrary order, so **byte-exact parity is claimed only
  for the undecorated modes**.

- `limit_alphabet` with count ties at the cutoff (unstable sort over hash
  order).
- Heap pop order between two entries with equal (count, pair) but different
  `pos` — outcome-equivalent after dedup; the serialized merge list is
  deterministic, but transient internal state may differ.
