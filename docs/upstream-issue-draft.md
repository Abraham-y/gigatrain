# Upstream: do NOT file this as a new issue

**Status: withdrawn as an issue draft.** An earlier version of this file was a
ready-to-file bug report against `huggingface/tokenizers` describing
`BpeTrainer` nondeterminism under `continuing_subword_prefix` /
`end_of_word_suffix`. Filing it would have been wrong on two counts.

## What was wrong with the draft

**1. It described #2066 as an issue that "reports nondeterminism without
isolating the trigger". #2066 is an open pull request that isolates it
exactly.**

[huggingface/tokenizers#2066](https://github.com/huggingface/tokenizers/pull/2066),
*"Make BPE/WordPiece training deterministic"*, opened 2026-05-22 by
`ATOM00blue`, still open, +99/−1 in `tokenizers/src/models/bpe/trainer.rs`. Its
body names the same root cause the draft did — `tokenize_words` assigning ids
while iterating an `AHashMap`, those ids feeding `Merge::cmp`'s tie-break — and
notes the same contrast with `compute_alphabet`, which already sorts. It
proposes the same class of fix and adds `test_train_is_deterministic`.

The draft's "Cause" and "Suggested fix" sections, and its closing "I implemented
exactly this in a separate trainer", would have restated an existing
contributor's open PR as new work. That is the kind of mistake that is hard to
walk back publicly.

**2. The diagnosis is still correct — only the framing was wrong.** Checked
against `tokenizers` v0.23.1: `tokenize_words` still iterates `wc` unsorted and
`pair_counts` is still `i32`. So the reproducer below holds against current
releases; the bug is real and unfixed *in a release*, because the fix is sitting
in an unmerged PR.

## What is actually worth doing

Not a new issue. Options, in rough order of value:

1. **Comment on #2066 with independent confirmation.** An outside reproduction
   on a different corpus, plus the impact framing the PR does not state — that
   `WordPieceTrainer` sets `continuing_subword_prefix("##")` in its default
   builder, so *WordPiece training is non-reproducible by default*, and anyone
   trying to verify a released tokenizer was trained from the data claimed
   currently cannot. Unmerged PRs are often waiting on evidence that the bug
   matters to someone other than the author.

2. **Note the ordering choice.** #2066 sorts the word counts before assigning
   ids. gigatrain instead registers the decorated token strings themselves in
   sorted order. Both are deterministic; they are not the *same* determinism, so
   a merged #2066 would still not agree merge-for-merge with gigatrain's
   decorated modes. If upstream determinism is going to become the reference,
   it is worth asking which order they intend to freeze before matching it.

3. **The `i32` overflow is a separate, still-unfiled matter** —
   [#2058](https://github.com/huggingface/tokenizers/issues/2058) covers it and
   is genuinely an issue rather than a PR.

## Reproducer (kept — it is independently useful)

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

Observed on `tokenizers` 0.22.2, corpus = Project Gutenberg *War and Peace*:

```
plain   deterministic
prefix  NOT deterministic     # merge lists first differ at index 142
suffix  NOT deterministic     # merge lists first differ at index 1266
```

Across three runs with `##`, the runs shared only 1805 of 1813 merges, and the
vocabularies also differed.

## Related

- [#2066](https://github.com/huggingface/tokenizers/pull/2066) — **open PR**,
  fixes this, unmerged as of 2026-08-02.
- [#1794](https://github.com/huggingface/tokenizers/issues/1794) — open issue,
  `WordPieceTrainer.train_from_iterator` is not deterministic. #2066 says
  "Fixes #1794".
- [#2058](https://github.com/huggingface/tokenizers/issues/2058) — `i32` pair
  count overflow. Separate bug, separate fix.
