#!/usr/bin/env python3
"""Train HF tokenizers' BpeTrainer and gigabpe on identical input and
compare serialized merge lists merge-for-merge.

Usage:
  python scripts/parity_check.py --files corpus.txt --vocab-size 2000
  python scripts/parity_check.py --files a.txt b.txt --vocab-size 32000 \
      --special "<|endoftext|>" --max-token-length 16
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
GIGABPE = ROOT / "gigabpe" / "target" / "release" / "gigabpe"


def hf_train(files, args):
    tok = Tokenizer(models.BPE())
    if args.pretokenizer == "bytelevel":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                     use_regex=True)
    else:
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
    if args.continuing_subword_prefix is not None:
        kwargs["continuing_subword_prefix"] = args.continuing_subword_prefix
    if args.end_of_word_suffix is not None:
        kwargs["end_of_word_suffix"] = args.end_of_word_suffix
    trainer = trainers.BpeTrainer(**kwargs)
    t0 = time.perf_counter()
    tok.train(files, trainer)
    elapsed = time.perf_counter() - t0
    model = json.loads(tok.to_str())["model"]
    merges = [tuple(m) for m in model["merges"]]
    # vocab is {token: id}; invert to id order so it compares positionally.
    vocab = [t for t, _ in sorted(model["vocab"].items(), key=lambda kv: kv[1])]
    return merges, vocab, elapsed


def gigabpe_train(files, args):
    cmd = [
        str(GIGABPE),
        "--vocab-size", str(args.vocab_size),
        "--min-frequency", str(args.min_frequency),
    ]
    for s in args.special:
        cmd += ["--special", s]
    if args.max_token_length is not None:
        cmd += ["--max-token-length", str(args.max_token_length)]
    if args.limit_alphabet is not None:
        cmd += ["--limit-alphabet", str(args.limit_alphabet)]
    if args.pretokenizer == "bytelevel":
        cmd += ["--pretokenizer", "bytelevel"]
    if args.continuing_subword_prefix is not None:
        cmd += ["--continuing-subword-prefix", args.continuing_subword_prefix]
    if args.end_of_word_suffix is not None:
        cmd += ["--end-of-word-suffix", args.end_of_word_suffix]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        vocab_path = f.name
    cmd += ["--vocab-out", vocab_path]
    cmd += files
    # The temp file must be removed even when the trainer exits non-zero, which
    # is exactly what the fuzzers do thousands of times per run.
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - t0
        print(f"  gigabpe stats: {proc.stderr.strip()}", file=sys.stderr)
        merges = []
        for line in proc.stdout.splitlines():
            a, sep, b = line.partition(" ")
            assert sep, f"bad merge line: {line!r}"
            merges.append((a, b))
        with open(vocab_path, encoding="utf-8") as fh:
            vocab = json.load(fh)
    finally:
        try:
            os.unlink(vocab_path)
        except FileNotFoundError:
            pass
    return merges, vocab, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--min-frequency", type=int, default=0)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("--max-token-length", type=int, default=None)
    p.add_argument("--limit-alphabet", type=int, default=None)
    p.add_argument("--pretokenizer", choices=["whitespace", "bytelevel"],
                   default="whitespace")
    p.add_argument("--continuing-subword-prefix", default=None)
    p.add_argument("--end-of-word-suffix", default=None)
    args = p.parse_args()

    print(f"HF tokenizers training (vocab={args.vocab_size})...", file=sys.stderr)
    hf_merges, hf_vocab, hf_time = hf_train(args.files, args)
    print(f"  {len(hf_merges)} merges in {hf_time:.2f}s", file=sys.stderr)

    print("gigabpe training...", file=sys.stderr)
    our_merges, our_vocab, our_time = gigabpe_train(args.files, args)
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

    # Identical merges do not imply identical vocabularies: ids, special
    # tokens and the alphabet can drift while every merge stays the same.
    nv = min(len(hf_vocab), len(our_vocab))
    for i in range(nv):
        if hf_vocab[i] != our_vocab[i]:
            print(f"VOCAB DIVERGENCE at id {i}:")
            lo = max(0, i - 3)
            for j in range(lo, min(nv, i + 4)):
                marker = " <-- HERE" if j == i else ""
                print(f"  [{j}] hf={hf_vocab[j]!r} ours={our_vocab[j]!r}{marker}")
            sys.exit(1)
    if len(hf_vocab) != len(our_vocab):
        print(
            f"VOCAB LENGTH MISMATCH: hf={len(hf_vocab)} ours={len(our_vocab)} "
            f"(first {nv} identical)"
        )
        sys.exit(1)

    speedup = hf_time / our_time if our_time > 0 else float("inf")
    print(
        f"PARITY OK: {len(hf_merges)} merges and {len(hf_vocab)} vocab entries "
        f"identical. hf={hf_time:.2f}s ours={our_time:.2f}s ({speedup:.1f}x)"
    )


if __name__ == "__main__":
    main()
