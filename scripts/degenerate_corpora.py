#!/usr/bin/env python3
"""Generate corpora shaped like the inputs that break real BPE trainers.

The failure reports motivating this project are not merely large; they are
*degenerate*. tokenizers#1313 is DNA-like data at vocab_size=512 with no
pre-tokenizer. sentencepiece#1021 logs `Alphabet size=4` on genomic data and
took 1.8 TB of RAM. tokenizers#1795 is Chinese JSONL. Web text is the case
everyone benchmarks and the case nobody fails on.

Each generator is deterministic given --seed so runs are comparable.

Usage:
  python scripts/degenerate_corpora.py --out-dir data/degenerate --size-mb 200
  python scripts/degenerate_corpora.py --list
"""
import argparse
import os
import random
import string
import sys

MB = 1 << 20


def _fill(write, size, make_line):
    """Call make_line() until `size` bytes have been written."""
    written = 0
    while written < size:
        chunk = make_line()
        write(chunk)
        written += len(chunk)
    return written


# Each generator writes bytes and returns nothing. They are given a raw binary
# writer and a byte budget.

def gen_dna_fasta(w, size, rng):
    """4-letter alphabet, 70-char lines. The #1313 / #1021 shape, but with
    line breaks, so the reader has cut points under both rules."""
    letters = b"ACGT"
    def line():
        return bytes(rng.choice(letters) for _ in range(70)) + b"\n"
    _fill(w, size, line)


def gen_dna_oneline(w, size, rng):
    """The same alphabet with NO newline and NO whitespace anywhere.

    Worst case for both cut rules: the entire file is a single pretoken under
    whitespace splitting, and a single line under ByteLevel."""
    letters = b"ACGT"
    block = 1 << 20
    written = 0
    while written < size:
        n = min(block, size - written)
        w(bytes(rng.choice(letters) for _ in range(n)))
        written += n


def gen_json_oneline(w, size, rng):
    """A single-line JSON array of records — a realistic database dump.

    No newline in the file at all, so under ByteLevel the reader buffers the
    whole thing; but the *pretokens* inside are short, so the memory cost is
    pure buffering overhead rather than genuine token size."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    w(b'[')
    written = 1
    i = 0
    while written < size:
        rec = ('{"id":%d,"name":"%s","tags":["%s","%s"],"score":%0.3f},' % (
            i, rng.choice(words), rng.choice(words), rng.choice(words),
            rng.random())).encode()
        w(rec)
        written += len(rec)
        i += 1
    w(b']')


def gen_cr_only(w, size, rng):
    """Classic-Mac line endings: \\r with no \\n anywhere.

    Under ByteLevel the cut rule is newline-only, so this file has no cut
    point at all despite looking like ordinary line-structured text."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    def line():
        n = rng.randint(6, 14)
        return (" ".join(rng.choice(words) for _ in range(n))).encode() + b"\r"
    _fill(w, size, line)


def gen_no_whitespace_cjk(w, size, rng):
    """Dense CJK with no spaces — the #1795 shape.

    Under whitespace splitting each line is one enormous word; under ByteLevel
    the GPT-2 regex splits every CJK char into its own pretoken, so the two
    modes should diverge sharply in cost."""
    # A small fixed set keeps the alphabet bounded and the Zipf head sharp.
    chars = [chr(0x4E00 + i) for i in range(512)]
    def line():
        n = rng.randint(200, 400)
        return ("".join(rng.choice(chars) for _ in range(n))).encode() + b"\n"
    _fill(w, size, line)


def gen_minified_js(w, size, rng):
    """Punctuation-dense single-line-ish code: long lines, tiny pretokens."""
    names = ["a", "b", "fn", "x1", "_t", "obj", "res"]
    def line():
        parts = []
        for _ in range(rng.randint(40, 80)):
            parts.append("%s=%s(%s,%d);" % (
                rng.choice(names), rng.choice(names), rng.choice(names),
                rng.randint(0, 9999)))
        return ("".join(parts)).encode() + b"\n"
    _fill(w, size, line)


def gen_base64(w, size, rng):
    """High-entropy alphanumeric with long lines: almost no repeated tokens,
    so the word table stays huge and the Zipf head is flat."""
    alpha = (string.ascii_letters + string.digits + "+/").encode()
    def line():
        n = rng.randint(2000, 4000)
        return bytes(rng.choice(alpha) for _ in range(n)) + b"\n"
    _fill(w, size, line)


def gen_logs(w, size, rng):
    """Highly repetitive structured lines — the opposite extreme: a tiny
    unique-word set over a huge corpus."""
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    mods = ["auth", "db", "http", "cache", "worker"]
    def line():
        return ("2026-08-03T12:%02d:%02d %s [%s] request_id=%d latency_ms=%d\n" % (
            rng.randint(0, 59), rng.randint(0, 59), rng.choice(levels),
            rng.choice(mods), rng.randint(0, 10**6), rng.randint(1, 5000))
        ).encode()
    _fill(w, size, line)


def gen_one_giant_word(w, size, rng):
    """A single token of the requested size, then ordinary text.

    Directly targets the u32 batch-offset limit and the reader's unbounded
    accumulation. At >=4 GiB this must abort cleanly, not hang or corrupt."""
    block = 1 << 20
    written = 0
    while written < size - min(size // 10, 8 * MB):
        n = min(block, size - written)
        w(b"x" * n)
        written += n
    w(b"\n")
    words = ["the", "quick", "brown", "fox"]
    def line():
        return (" ".join(rng.choice(words) for _ in range(10))).encode() + b"\n"
    _fill(w, size - written - 1, line)


GENERATORS = {
    "dna_fasta": gen_dna_fasta,
    "dna_oneline": gen_dna_oneline,
    "json_oneline": gen_json_oneline,
    "cr_only": gen_cr_only,
    "cjk_dense": gen_no_whitespace_cjk,
    "minified_js": gen_minified_js,
    "base64": gen_base64,
    "logs": gen_logs,
    "giant_word": gen_one_giant_word,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/degenerate")
    p.add_argument("--size-mb", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--only", action="append", default=[],
                   help="generate only these (repeatable)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, fn in GENERATORS.items():
            print(f"{name:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return

    names = args.only or list(GENERATORS)
    os.makedirs(args.out_dir, exist_ok=True)
    size = args.size_mb * MB
    for name in names:
        if name not in GENERATORS:
            sys.exit(f"unknown corpus {name!r}; try --list")
        path = os.path.join(args.out_dir, f"{name}_{args.size_mb}mb.txt")
        if os.path.exists(path) and os.path.getsize(path) >= size * 0.95:
            print(f"  {name:14s} cached ({os.path.getsize(path)/MB:.0f} MB)")
            continue
        rng = random.Random(args.seed)
        with open(path, "wb", buffering=1 << 20) as f:
            GENERATORS[name](f.write, size, rng)
        got = os.path.getsize(path)
        print(f"  {name:14s} {got/MB:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
