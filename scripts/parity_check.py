#!/usr/bin/env python3
"""Train HF tokenizers' BpeTrainer and gigatrain on identical input and
compare serialized merge lists merge-for-merge.

Usage:
  python scripts/parity_check.py --files corpus.txt --vocab-size 2000
  python scripts/parity_check.py --files a.txt b.txt --vocab-size 32000 \
      --special "<|endoftext|>" --max-token-length 16
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
GIGATRAIN = ROOT / "gigatrain" / "target" / "release" / "gigatrain"


def hf_train(files, args):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    kwargs = dict(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special,
        show_progress=False,
    )
    if args.max_token_length is not None:
        kwargs["max_token_length"] = args.max_token_length
    if args.limit_alphabet is not None:
        kwargs["limit_alphabet"] = args.limit_alphabet
    trainer = trainers.BpeTrainer(**kwargs)
    t0 = time.perf_counter()
    tok.train(files, trainer)
    elapsed = time.perf_counter() - t0
    model = json.loads(tok.to_str())["model"]
    merges = [tuple(m) for m in model["merges"]]
    return merges, elapsed


def gigatrain_train(files, args):
    cmd = [
        str(GIGATRAIN),
        "--vocab-size", str(args.vocab_size),
        "--min-frequency", str(args.min_frequency),
    ]
    for s in args.special:
        cmd += ["--special", s]
    if args.max_token_length is not None:
        cmd += ["--max-token-length", str(args.max_token_length)]
    if args.limit_alphabet is not None:
        cmd += ["--limit-alphabet", str(args.limit_alphabet)]
    cmd += files
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - t0
    print(f"  gigatrain stats: {proc.stderr.strip()}", file=sys.stderr)
    merges = []
    for line in proc.stdout.splitlines():
        a, sep, b = line.partition(" ")
        assert sep, f"bad merge line: {line!r}"
        merges.append((a, b))
    return merges, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--min-frequency", type=int, default=0)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("--max-token-length", type=int, default=None)
    p.add_argument("--limit-alphabet", type=int, default=None)
    args = p.parse_args()

    print(f"HF tokenizers training (vocab={args.vocab_size})...", file=sys.stderr)
    hf_merges, hf_time = hf_train(args.files, args)
    print(f"  {len(hf_merges)} merges in {hf_time:.2f}s", file=sys.stderr)

    print("gigatrain training...", file=sys.stderr)
    our_merges, our_time = gigatrain_train(args.files, args)
    print(f"  {len(our_merges)} merges in {our_time:.2f}s (incl. subprocess)", file=sys.stderr)

    n = min(len(hf_merges), len(our_merges))
    for i in range(n):
        if hf_merges[i] != our_merges[i]:
            print(f"DIVERGENCE at merge {i}:")
            lo = max(0, i - 3)
            for j in range(lo, min(n, i + 4)):
                marker = " <-- HERE" if j == i else ""
                print(f"  [{j}] hf={hf_merges[j]!r} ours={our_merges[j]!r}{marker}")
            sys.exit(1)
    if len(hf_merges) != len(our_merges):
        print(
            f"LENGTH MISMATCH: hf={len(hf_merges)} ours={len(our_merges)} "
            f"(first {n} identical)"
        )
        sys.exit(1)

    speedup = hf_time / our_time if our_time > 0 else float("inf")
    print(
        f"PARITY OK: {len(hf_merges)} merges identical. "
        f"hf={hf_time:.2f}s ours={our_time:.2f}s ({speedup:.1f}x)"
    )


if __name__ == "__main__":
    main()
