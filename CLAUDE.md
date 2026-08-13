# Fast BPE Tokenizer Training

## What this is

A tokenizer **trainer** that runs at web scale. Not an encoder — Gigatoken
(https://github.com/marcelroed/gigatoken) already solved encoding at ~24 GB/s.

The pitch: train a 32k-vocab BPE tokenizer on 10 GB+ of text in under a minute,
with byte-exact output parity against HuggingFace `tokenizers`.

## Status, 2026-08-06

Milestones 1–6 done, 7 half done (WordPiece; no Unigram). Byte-exact HF parity
verified at 12.9 GB in both pretokenization modes. Milestone 5 met: 19.4 GB
trains in 47 s / 2.9 GB where HF needs 36.3 GB.

**Read [docs/CORRECTIONS.md](docs/CORRECTIONS.md) before quoting any number.**
This project has retracted an entire experiment and revised five headline
figures, several more than once. The standing rules at the bottom of that file
exist because each was broken at least twice.

Decision (2026-08-06): **no paper.** Ship the tool, blog it, file the upstream
bug. See [docs/publishing.md](docs/publishing.md) for why both candidate papers
were dropped.

## Architecture

Rust core, Python bindings via PyO3.

**Phase 1** — pretokenization and word counting. Embarrassingly parallel:
reader → scanners (split + hash) → disjoint hash-sharded owners (count). Read
files directly from Rust; do not round-trip through Python.

**Phase 2** — the merge loop. Inherently sequential (merge N+1 depends on
merge N), memory-bound. Optimization here is memory layout, not parallelism.

Details in [ARCHITECTURE.md](ARCHITECTURE.md).

### Do NOT use a GPU

Deliberate. The merge loop is sequential by construction; the hot structures
are hash maps and variable-length sets (pointer chasing, irregular access, not
dense arithmetic); work per merge is small and irregular so kernel launch
overhead dominates. Dedup went GPU because MinHash *is* dense parallel hashing.
This workload is the opposite shape. Empirically supported: every "GPU BPE"
project is an *encoder* with a pre-trained merge table.

Target hardware: many cores for phase 1, large RAM for phase 2.

### Memory is the binding constraint

The incumbent failures are OOMs, not timeouts. Design for it: arena/bump
allocation for symbol sequences, compact `u32` ids, a flat `symbols: Vec<u32>`
with per-word offset slices, streaming phase 1 so the frequency table rather
than the corpus must fit in RAM.

Profiling overturned the obvious guess: `pair_where` was 1 MB at 1 GB, while
phase 1's `HashMap<String, u64>` was 1.7 GB of a 2.26 GB peak.

## Non-negotiable: exact parity

Output must match `tokenizers.trainers.BpeTrainer` merge-for-merge including
tie-breaking. Without it nobody can adopt this as a drop-in.

`scripts/run_parity_ci.sh` gates every commit. Never land a trainer change
without it passing. [PARITY.md](PARITY.md) is the specification — HF's tie-break
rule, stale-heap handling, `max_token_length` asymmetry, `i32` overflow in both
directions, and the line-at-a-time file feed that silently changes output.

## Benchmarking rules

These are not style preferences; each one exists because it was violated.

1. **Grep every number against its source before publishing.** If it appears
   only in the write-up, it is not a measurement.
2. **One variable per experiment.** If a thread scan is run for gigabpe, run
   it for the baseline. State the configuration of *every* system compared.
3. **Verify the machine is quiet and say so.** Every laptop number in this
   repo's history was taken while an unrelated training job was running.
4. **Repeat across allocations, not within.** ±2% within a container; 20–28%
   between them.
5. **Only a timeout may print as a timeout.** A harness fault once rendered 49
   failures as a table of `>900s` timeouts.
6. **Validate against real data before concluding.** Synthetic corpora differ
   from real ones of the same nominal type by up to 24x, in both directions.
7. **One session per comparison table**, or the ratios are not comparable.
8. Always report wall time, peak RSS, and merge-list parity together. A speedup
   with different output is not a speedup.

## What is and is not defensible

**Defensible:** byte-exact parity at 12.9 GB with a 6.8x speedup on the same
pretokenizer; 19.4 GB in 2.9 GB of RAM against HF's 36.3 GB; completing 20/20
degenerate configurations where HF does 15/20; PARITY.md as an artifact.

**Defensible with caveats:** "fastest of the seven trainers that exist, on web
text" — all seven are now in one comparable table. Attach that only the HF rows
have verified identical output, and that rustbpe is ~15% faster on
single-giant-pretoken corpora.

**Not defensible:** novelty of the algorithm (Zouhar et al., implemented three
times over); being first to HF-parity training (gigatoken); an unqualified
"fastest BPE trainer"; reproducing #1313.

## Outstanding

- Interior cut rule for boundary-free input (2.0x, parity-critical, designed).
- `train_from_iterator` — the Python API takes file paths only, which is the
  biggest adoption gap.
- No regression test for the scanner-panic deadlock (needs a 4 GiB pretoken).
- ~~File the phantom-merge bug~~ filed as
  [tokenizers#2320](https://github.com/huggingface/tokenizers/issues/2320).
  Still to do: comment on HF PR #2066 with independent confirmation.
- ~~Ship it~~ published 2026-08-13: `pip install gigabpe`
  ([PyPI](https://pypi.org/project/gigabpe/)), v0.1.0, 30 wheels (CPython
  3.9–3.14 × Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64) plus
  sdist, via Trusted Publishing. Verified from real PyPI in a clean venv:
  byte-identical to HF in both pretokenizer modes.
- The name is **gigabpe**, not gigatrain. PyPI's similarity check squashes
  separators, so `gigatrain` collided with GigaAI's existing `giga-train`
  training framework. Renamed 2026-08-13 (commit 4e8afa5) before publishing.
