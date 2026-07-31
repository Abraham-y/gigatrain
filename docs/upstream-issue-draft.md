# Draft: issue for huggingface/tokenizers

Not filed. This is a draft for review — filing it is a decision for the repo
owner, and it should probably be checked against the latest `tokenizers`
release first, since the measurements below are on 0.22.2.

Related existing issues: #2066, #1794 (both report nondeterminism without
isolating the trigger).

---

**Title:** `BpeTrainer` is non-deterministic when `continuing_subword_prefix`
or `end_of_word_suffix` is set (so `WordPieceTrainer` is non-reproducible by
default)

**Body:**

Training the same corpus twice with `continuing_subword_prefix` set produces
different merge lists and different vocabularies. Without it, training is
deterministic. Since `WordPieceTrainer` sets `continuing_subword_prefix("##")`
in its default builder, **WordPiece training is not reproducible by default**.

This looks like the underlying cause of #2066 and #1794, which report
nondeterminism but do not isolate the trigger.

### Reproduction

```python
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def train(**kw):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train(["corpus.txt"],
              trainers.BpeTrainer(vocab_size=2000, show_progress=False, **kw))
    return [tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]

for label, kw in [("plain", {}),
                  ("prefix", {"continuing_subword_prefix": "##"}),
                  ("suffix", {"end_of_word_suffix": "</w>"})]:
    runs = [train(**kw) for _ in range(3)]
    print(label, "deterministic" if runs[0] == runs[1] == runs[2] else "NOT deterministic")
```

Observed on `tokenizers` 0.22.2, corpus = Project Gutenberg *War and Peace*
(any corpus with enough word-internal characters will do):

```
plain   deterministic
prefix  NOT deterministic     # merge lists first differ at index 142
suffix  NOT deterministic     # merge lists first differ at index 1266
```

Across three runs with `##`, the runs shared only 1805 of 1813 merges, and the
vocabularies also differed.

### Cause

In `BpeTrainer::tokenize_words` (`models/bpe/trainer.rs`), decorated tokens
are created while iterating the word-count map:

```rust
for (word, count) in wc {           // wc: AHashMap<CompactString, u64>
    for (is_first, is_last, c) in word.chars().with_first_and_last() {
        let mut s = c.to_string();
        if !is_first { /* prepend continuing_subword_prefix */ }
        if is_last   { /* append end_of_word_suffix */ }
        if !w2id.contains_key(&s) {
            id2w.push(s.clone());   // <-- id depends on iteration order
            w2id.insert(s, (id2w.len() - 1) as u32);
        }
        ...
    }
}
```

`wc` is an `AHashMap`, so iteration order varies between processes. Those ids
then feed the tie-break comparator:

```rust
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count { self.count.cmp(&other.count) }
        else { other.pair.cmp(&self.pair) }   // <-- compares vocabulary ids
    }
}
```

so equal-count pairs are ordered by ids that are themselves order-dependent,
and the output varies run to run.

Without decoration this does not arise: every token entered here is a single
character already registered by `compute_alphabet`, which sorts by codepoint,
so no new ids are allocated in map order.

### Suggested fix

Register the decorated tokens in a deterministic order before tokenizing —
collect the set that will be needed and insert it sorted. That is O(alphabet)
extra work and makes ids independent of map iteration order.

I implemented exactly this in a separate trainer and it produces byte-identical
output across runs and thread counts, so the approach works in practice.

### Impact

Anyone who needs to rebuild a WordPiece vocabulary reproducibly — for
provenance, for a paper, or to verify a released tokenizer was trained from
the data claimed — currently cannot, and there is no warning that this is the
case.
