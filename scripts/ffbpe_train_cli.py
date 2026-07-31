#!/usr/bin/env python3
"""Train tokn-ai/ffbpe, for comparison.

ffbpe (released 2026-07-27) is the newest fast BPE trainer and the only other
project with a bounded-memory exact mode. Its published headline (1 GiB in
5.58 s) is not directly comparable to the numbers in BENCHMARKS.md: it uses
vocab 10,000 on Chinese text, counts from a precomputed Unicode-bigram
inventory rather than raw text, and states no hardware. Its own docs say the
HF comparison in their repo "is not a pure trainer-algorithm comparison".

This runs it end to end from raw text, which is the same contract the other
trainers are measured under.

Usage:
  python3 scripts/ffbpe_train_cli.py --vocab-size 32000 corpus.txt
"""
import argparse
import sys
import time

import ffbpe


def records(path):
    """Feed one record per line, matching how HF's trainer consumes files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("input")
    args = p.parse_args()

    t0 = time.perf_counter()
    model = ffbpe.train_bpe(
        records(args.input),
        vocab_size=args.vocab_size,
        special_tokens=args.special,
    )
    elapsed = time.perf_counter() - t0

    print(f"train_seconds: {elapsed:.3f}", file=sys.stderr)
    for attr in ("vocab_size", "merges"):
        val = getattr(model, attr, None)
        if val is None:
            continue
        try:
            print(f"{attr}: {val if isinstance(val, int) else len(val)}",
                  file=sys.stderr)
        except TypeError:
            pass


if __name__ == "__main__":
    main()
