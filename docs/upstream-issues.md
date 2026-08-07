# Upstream: what to file, and what not to

## 1. FILE THIS — negative pair counts produce a phantom merge

**Not reported anywhere.** The single novel finding in this project, verified
against HuggingFace over 200 runs.

PARITY.md documents the *positive* `i32` overflow (>2^31 occurrences of one
pair ⇒ HF emits fewer merges). The opposite sign is reachable at **8 words /
38 bytes** and does the opposite.

Under `continuing_subword_prefix` (or `end_of_word_suffix`) combined with
`max_token_length`, merge-time id reuse drives a pair count **negative**: when
a merge produces a token that already has an id — e.g. `('##','##c') -> '##c'`
— the `-1`/`+1` neighbour deltas no longer cancel, because the length guard
rejects the `+1` while the `-1` still applies. A pair already at zero goes
to `-1`.

HF then evaluates `pair_counts[&pair] as u64`, turning `-1` into
`18446744073709551615`. That passes the staleness re-check, is not `< 1`, and
**wins the heap immediately** — so HF emits a merge for a pair that occurs
nowhere in the corpus.

**Reproducer:**

```
corpus: '##c ##cac# #ab c# #### accc#a#a#b a cb'
--vocab-size 244 --continuing-subword-prefix '##' --max-token-length 4 \
  --special a --special '<unk>'
```

15 merges, with the phantom `('##c','##a')` at index 11. Verified against HF
over 200 runs: HF produced 8 distinct outputs (decorated-mode nondeterminism),
and the **13 runs that landed on our token ordering matched exactly**, phantom
included. Only `max_token_length` 4 triggers it on this corpus; 2, 3, 5, 6, 8,
100 and unset do not.

gigatrain reproduces this deliberately — `live as u64` on an `i64` sign-extends
identically — because parity means matching the bugs too.

## 2. DO NOT FILE — comment on the existing PR instead

An earlier draft here was a bug report for `BpeTrainer` nondeterminism under
`continuing_subword_prefix`. **Do not file it.**
[#2066](https://github.com/huggingface/tokenizers/pull/2066) is an **open pull
request** (ATOM00blue, 2026-05-22, still open) that already isolates the same
trigger — `tokenize_words` assigning ids while iterating an `AHashMap`, those
ids feeding `Merge::cmp`'s tie-break — proposes the same class of fix, and adds
a regression test. Filing the draft would have restated another contributor's
open work as new.

The diagnosis is still correct against v0.23.1: `tokenize_words` still iterates
`wc` unsorted. So **comment on #2066** with:

- Independent reproduction on a different corpus.
- The impact framing the PR omits: `WordPieceTrainer` sets
  `continuing_subword_prefix("##")` in its default builder, so **WordPiece
  training is non-reproducible by default**, and nobody can verify a released
  tokenizer was trained from the data claimed.
- A note that #2066 sorts *word counts* while gigatrain sorts the *decorated
  token strings* — both deterministic, but not the same determinism, so a
  merged #2066 still would not agree merge-for-merge with gigatrain's decorated
  modes. Worth asking which order they intend to freeze.

**Reproducer for the comment:**

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

On 0.22.2 with War and Peace: `plain` deterministic; `prefix` differs at merge
142; `suffix` differs at merge 1266. Across three `##` runs the outputs shared
only 1805 of 1813 merges.

## 3. Also worth reporting

- **gigatoken CRLF divergence** — matches HF 2744/2744 on LF corpora, 21/2744
  on CRLF, diverging at rank 0, because it pretokenizes whole-text rather than
  line-at-a-time. Email Marcel Rød rather than filing; it is a small fix.
- **SentencePiece degenerate-corpus failures** — 5 of 7 in this repo's runs,
  including a SIGABRT on minified JS and refusals on low-alphabet corpora
  ("Vocabulary size too high (2000). Please set it to a value <= 28"). Related
  to sentencepiece#862 and #1021. Needs characterising before filing.
