#!/usr/bin/env python3
"""Fuzz merge-loop parity on small random word-count tables.

Small alphabets and short repetitive words maximize count ties, overlapping
pairs (aaa...), and token-string reuse ("abc" formed as both "ab"+"c" and
"a"+"bc") — the pathological paths documented in PARITY.md that natural
corpora rarely exercise.

HF side feeds each word `count` times through train_from_iterator, which
produces the identical word-count table. Our side gets the table directly
via --words-tsv.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
GIGABPE = ROOT / "gigabpe" / "target" / "release" / "gigabpe"


def random_table(rng):
    # Tiny alphabets (down to a single char) with long words maximize
    # same-char runs, which is what triggers token-string reuse and the
    # duplicate-merge / partial-pos paths (PARITY.md).
    size = rng.choice([1, 1, 2, 2, 3, 4])
    alphabet = rng.sample("abcde", size)
    n_words = rng.randint(3, 12)
    table = {}
    for _ in range(n_words):
        length = rng.randint(1, 16)
        word = "".join(rng.choice(alphabet) for _ in range(length))
        table[word] = table.get(word, 0) + rng.randint(1, 30)
    return table


def hf_train(table, vocab_size, max_token_length):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    kwargs = dict(vocab_size=vocab_size, min_frequency=0, show_progress=False)
    if max_token_length is not None:
        kwargs["max_token_length"] = max_token_length
    trainer = trainers.BpeTrainer(**kwargs)

    def gen():
        for word, count in table.items():
            for _ in range(count):
                yield word

    tok.train_from_iterator(gen(), trainer)
    model = json.loads(tok.to_str())["model"]
    return [tuple(m) for m in model["merges"]]


def ours_train(table, vocab_size, max_token_length):
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        for word, count in table.items():
            f.write(f"{word}\t{count}\n")
        path = f.name
    cmd = [str(GIGABPE), "--vocab-size", str(vocab_size), "--words-tsv", path]
    if max_token_length is not None:
        cmd += ["--max-token-length", str(max_token_length)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        Path(path).unlink()
    merges = []
    for line in proc.stdout.splitlines():
        a, _, b = line.partition(" ")
        merges.append((a, b))
    return merges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    for trial in range(args.trials):
        table = random_table(rng)
        vocab_size = rng.randint(5, 60)
        max_token_length = rng.choice([None, None, 2, 3, 5])
        hf = hf_train(table, vocab_size, max_token_length)
        ours = ours_train(table, vocab_size, max_token_length)
        if hf != ours:
            print(f"FAIL trial {trial} (seed {args.seed})")
            print(f"  table: {table}")
            print(f"  vocab_size={vocab_size} max_token_length={max_token_length}")
            print(f"  hf   ({len(hf)}): {hf}")
            print(f"  ours ({len(ours)}): {ours}")
            sys.exit(1)
        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1}/{args.trials} ok", file=sys.stderr)
    print(f"FUZZ OK: {args.trials} trials, zero divergence")


if __name__ == "__main__":
    main()
