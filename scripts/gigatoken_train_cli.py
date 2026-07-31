#!/usr/bin/env python3
"""Train gigatoken's BPE trainer, for comparison.

gigatoken (github.com/marcelroed/gigatoken) is best known as an encoder, but
ships an undocumented `train_bpe` with a HuggingFace tie-breaking mode. Its
own parity test runs at ~120 KB / vocab 500, so this measures what it does at
corpus scale.

`train_bpe` takes a FileSource and does its own pretokenization, so this is a
speed and memory comparison; merge lists are only comparable where both sides
use the same pretokenizer and alphabet, which is checked separately.

Usage:
  python3 scripts/gigatoken_train_cli.py --vocab-size 32000 corpus.txt
"""
import argparse
import sys
import time

import gigatoken


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("--tie-breaking", default="huggingface")
    p.add_argument("input", nargs="+")
    args = p.parse_args()

    source = gigatoken.TextFileSource(args.input)

    t0 = time.perf_counter()
    result = gigatoken.train_bpe(
        source, args.vocab_size, args.special, args.tie_breaking
    )
    elapsed = time.perf_counter() - t0

    print(f"train_seconds: {elapsed:.3f}", file=sys.stderr)
    # The result shape is not documented; report what we can see.
    for attr in ("merges", "vocab"):
        val = getattr(result, attr, None)
        if val is not None:
            try:
                print(f"{attr}: {len(val)}", file=sys.stderr)
            except TypeError:
                pass


if __name__ == "__main__":
    main()
