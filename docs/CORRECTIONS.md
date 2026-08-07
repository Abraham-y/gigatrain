# Corrections

Every claim this project made and then withdrew, with the cause. Kept in one
place because the failures are more instructive than the findings, and because
several were made twice.

Read the **Standing rules** at the bottom before adding any number anywhere.

---

## A. Claims retracted outright

**The intrinsic vocabulary-scaling sweep (2026-08-02).** An entire experiment,
withdrawn. Six independent defects, each sufficient alone:

1. The multilingual held-out set was **one language**. `_resolve_urls` grouped
   shards by language, so `local[-1]` was the last Japanese shard. The headline
   "multilingual text costs 70.6% of compression" was "Japanese text does".
2. Per-language equity was measured **in-sample** — the evaluation text was the
   first ~2 MB of each language's first shard, all inside training.
3. The multilingual corpus was **never balanced**. Round-robin took one
   *document* per stream, but mean document size varies 2.7x; Russian held 32%
   of characters, Japanese 9%.
4. The English 10 GB reference was **unreproducible and contaminated** — the
   current code cannot rebuild it, and the run that produced it trained on its
   own held-out shard.
5. The "independent seeds" were not independent. Seeds skipped N documents *per
   stream*; English had 1 stream (91–95% document overlap between "samples"),
   multilingual had 39 (0% overlap). A finding about variance was a finding
   about shard layout.
6. Rank-stratified overlap **measured the byte alphabet**. Token ids run
   specials → alphabet → merges, and the first merge lands around id 170, so
   the "top 256" window was ~97% alphabet.

And, independently of all six: **the headline had already been published.**
Vocabulary overlap against a large-corpus reference as a function of corpus
size is Reddy et al. (arXiv:2502.20273) §1/Fig. 1, at 900 GB across three
algorithms. This repo twice asserted in writing that it was not in that paper.

**"Reproduces tokenizers#1313" (2026-07-31).** #1313 is `vocab_size=512` on
unsegmented DNA-like data, and a maintainer diagnosed it in-thread as
degenerate pretokenization. A 32k-vocab FineWeb run does not reproduce it. This
claim was retracted and then **re-asserted 100 lines later in the same file**;
the second instance survived another week.

**"Synthetic data inverted the conclusion" (2026-08-05).** Synthetic
`dna_oneline` timed out every trainer and was written up as an irreducible
property of BPE. Controlled rerun — matched synthetic/real pairs, one
container, one 1800 s timeout — shows the pair differs by **1.20x**, inside the
between-allocation noise. The original result was the **180 s timeout**, not
the data. Across all five matched pairs the synthetic/real ratio ranges
0.33x–24x *with no consistent sign*; real is slower in three of them.

**"gigatoken's trainer likely won't survive 12.9 GB."** Source inference from
its `BTreeSet` index. Measured: it is fast and memory-efficient, the closest
competitor.

---

## B. Numbers that moved once measured properly

| claim | published | corrected | cause |
|---|---|---|---|
| HF slowdown from cores | **19x** | **10.3x** @100 MB, **1.34x** at YTTM's 1 GB/36-thread config | cross-machine (ISA, OS, allocator) vs controlled sweep |
| boundary-free penalty | **16.7x** | **2.0x** | two laptop runs minutes apart, on a corpus that was 10x-repeated (inflating the effect) |
| between-allocation variance | **40%** | **20–28%** | n=2 accidental preemption vs n=8 verified allocations |
| "~50x at 100 MB ByteLevel" | quoted | **deleted** | HF's 1 GB *whitespace* time pasted into a 100 MB *ByteLevel* cell; never measured |
| 12.9 GB laptop run | 104 s / 2.4 GB | 85 s / 2.2 GB | superseded, old figure left in a second file |
| gigatrain @ 1 GB ByteLevel | 8.5 s / 10.22 s / 14.9 s | **6.7 s** (one-session) | three separate sessions; ratios built on them were never mutually comparable |
| README figures | 20.3 s, 14.5 s | **deleted** | appeared in no measurement anywhere |

**"Fastest BPE trainer"** was unsupported while two of the seven trainers had
never been run. Resolved 2026-08-07: all seven are now in one comparable table
and gigatrain is fastest at 100 MB and 1 GB. The claim is supportable *with
caveats* — only the HuggingFace rows have verified identical output, and
rustbpe is ~15% faster on single-giant-pretoken corpora (`dna_real_oneline`
235.7 s vs 266.8 s).

Adding the two missing trainers also moved gigatoken's ratio from **3.5x to
3.9x**, because gigatrain's own baseline shifted 7.1 s → 6.7 s between two
single-allocation runs — well inside the 20–28% between-allocation spread. A
reminder that one-container ratios are sound but one-container *absolutes* are
not.

---

## C. Attribution and citation errors

- **huggingface/tokenizers#2066 is an open pull request, not an issue.** It
  already isolates the nondeterminism trigger and proposes the same fix. A
  drafted bug report would have re-filed another contributor's work as new.
- The SentencePiece merge-loop analysis (76% sequential, ~1.3x Amdahl ceiling)
  is **taku910's**, not rustbpe's maintainer's. "Compact arrays, custom hash
  maps, split queues" is him describing **YouTokenToMe**, not prescribing this
  project's architecture.
- The **lazy heap is Zouhar et al., not Sennrich et al.** Sennrich §3.2 states
  the incremental half only; "heap" and "priority" do not appear in that paper.
- **HF has had trainer performance work in 2024–2025** (#1433 u64 word counts,
  #1799 ahash/dary_heap/CompactString). "No trainer perf work 2024–2026" was
  false and framed a maintained baseline as abandoned.
- **ffbpe dates to December 2025**, not July 2026; v0.1.8 is its first release
  after a rename. Its README headline is a 64 MiB Chinese fixture, not the
  5.58 s figure, which is a *before* value in an RSS comparison.
- **fast-bytelevel-bpe-go's speed is not comparable to anything** — its corpus
  is "846,882 non-empty rows" with no size in bytes. "Two orders of magnitude
  slower than gigatrain" had no basis.
- Scoring a project from its README understates it: **ffbpe moved 2/8 → 6/8**
  after reading its `BENCHMARKS.md`.

---

## D. Process failures that produced the above

- **The measurement environment was never verified.** All laptop numbers were
  taken while an unrelated training job ran (load average 4.4 on 10 cores),
  unnoticed until someone asked. None of the six audited benchmarks state their
  machine was quiet either.
- **A harness fault was rendered as data.** `/usr/bin/time -l` is BSD-only;
  GNU time rejects it and exits 125 without running anything. 49 cells came
  back as failures, and the reporting layer printed every one as `>900s` — a
  full table of fabricated timeouts.
- **A broken identifier nearly supported a wrong attribution.** The first
  between-allocation experiment reported `distinct_hosts=1` because every Modal
  sandbox is named `modal`. The number was defensible; the word
  "between-container" was not.
- **A parity result was asserted before it was computed.** The README claimed
  byte-identical merges at 12.9 GB while the harness sent trainer stdout to
  `/dev/null`.
- **An orphaned process corrupted later measurements.** `subprocess.run`
  timeouts kill `/usr/bin/time` but leave the trainer at PPID 1; one ran 12+
  minutes alongside subsequent runs.
- **Cached artifacts were trusted.** `curl -s` without `-f` wrote HTTP error
  bodies to `*.parquet` and exited 0; a zero-byte cached corpus once produced a
  perfectly flat curve across all sizes.

---

## E. Standing rules

1. **Grep every number against its source before publishing.** If it appears
   only in the write-up, it is not a measurement.
2. **One variable per experiment.** If a scan is run for the home tool, run it
   for the baseline. State the configuration of *every* system compared.
3. **Verify the environment is quiet, and say so.** Record load average.
4. **Repeat across allocations, not within**, and verify the allocations
   actually differ. Within-run spread (±2%) understates reality by ~10x.
5. **Only a timeout may print as a timeout.** Harness faults get a loud,
   distinct status. Uniformity across a table is a smell.
6. **Validate against real data before concluding.** Synthetic corpora differ
   from real ones of the same nominal type by up to 24x, in both directions.
7. **Read the related work before running the experiment**, not after.
8. **One session per comparison table**, or the ratios are not comparable.
9. **Never delete a retraction.** Correct in place and keep the original text.
