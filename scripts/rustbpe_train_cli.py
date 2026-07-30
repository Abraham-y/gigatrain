#!/usr/bin/env python3
"""Train karpathy/rustbpe on a text file, for speed/memory comparison only.

rustbpe is not merge-list comparable with HuggingFace: it uses the GPT-4
split pattern (not GPT-2), and exports tiktoken mergeable ranks rather than a
HF merge list. Only wall time and peak RSS are comparable.

It takes a Python iterator of strings, so the file is fed line by line, which
matches how HF's trainer consumes files.

Usage:
  python3 scripts/rustbpe_train_cli.py --vocab-size 32000 corpus.txt
"""
import argparse
import sys
import time

import rustbpe


def lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--pattern", default=None,
                   help="override the split pattern (default: GPT-4)")
    p.add_argument("input")
    args = p.parse_args()

    tok = rustbpe.Tokenizer()
    t0 = time.perf_counter()
    if args.pattern:
        tok.train_from_iterator(lines(args.input), args.vocab_size,
                                pattern=args.pattern)
    else:
        tok.train_from_iterator(lines(args.input), args.vocab_size)
    elapsed = time.perf_counter() - t0

    print(f"train_seconds: {elapsed:.3f}", file=sys.stderr)
    print(f"vocab_size: {tok.vocab_size}", file=sys.stderr)


if __name__ == "__main__":
    main()
