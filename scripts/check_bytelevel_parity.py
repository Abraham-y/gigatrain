#!/usr/bin/env python3
"""Differential test: our ByteLevel pretokenizer vs HuggingFace's.

The GPT-2 pattern is reimplemented by hand (the `regex` crate cannot express
its `(?!\\S)` lookahead), so it is verified by diffing against HF's own output
rather than by reading it. Checks:

1. handcrafted edge cases around spaces, contractions and boundaries
2. every non-surrogate BMP codepoint in several contexts, to catch category
   misclassification (\\p{L} vs Alphabetic, Nl, Other_Alphabetic, ...)
3. real corpus text, line by line

Cases are batched through `pretok --lines` so the whole sweep is one process.
Newlines cannot appear inside a batched case, so any case containing one is
checked individually.

Usage: python3 scripts/check_bytelevel_parity.py [corpus.txt ...]
"""
import subprocess
import sys
from pathlib import Path

from tokenizers import pre_tokenizers

ROOT = Path(__file__).resolve().parent.parent
PRETOK = ROOT / "gigatrain" / "target" / "release" / "pretok"

HF = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)


def ours_batch(cases):
    """Pretokenize many newline-free cases in one process."""
    payload = "\n".join(cases)
    proc = subprocess.run(
        [str(PRETOK), "--bytelevel", "--lines"],
        input=payload.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    lines = proc.stdout.decode("utf-8").split("\n")
    # `split` on the trailing newline leaves one extra empty entry.
    lines = lines[: len(cases)]
    return [ln.split("\t") if ln else [] for ln in lines]


def ours_single(text):
    proc = subprocess.run(
        [str(PRETOK), "--bytelevel"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    out = proc.stdout.decode("utf-8")
    return out.split("\n")[:-1] if out else []


def theirs(text):
    return [s for s, _ in HF.pre_tokenize_str(text)]


def report(text, a, b, label, shown):
    if shown < 5:
        print(f"FAIL [{label}] {text!r}\n  ours={a!r}\n  hf  ={b!r}")
    return 1


def run_batch(cases, label):
    """Returns number of mismatches."""
    got = ours_batch(cases)
    bad = 0
    for text, a in zip(cases, got):
        b = theirs(text)
        if a != b:
            bad = bad + report(text, a, b, label, bad)
    return bad


def main():
    failures = 0

    print("edge cases (individual, may contain newlines)...", file=sys.stderr)
    edge = [
        "", " ", "  ", "\n", "\t", "a\nb", "a\n\nb", "  \n  ", "\r\n",
        "a", "a b", "a  b", "a   b", " a", "  a", "a ", "a  ",
        "don't", "DON'T", "'s", "'S", "''", "can't stop", "y'all've",
        "123", "a1", "1a", "1.5", "-3", "e=mc²", "x_y", "a-b", "!!!", "?!",
        "　x", " x", "🙂", "a🙂b", "🙂 🙂",
        "Ⅷ", "Ⅷa", "Ⅷ123", "aͅb", "café", "中文", "中文 text",
        "½", "½a", "ᵃ", "ﬀ", "ʰ", "〇", "一二三",
    ]
    for t in edge:
        a, b = ours_single(t), theirs(t)
        if a != b:
            failures += report(t, a, b, "edge", failures)
    print(f"  {len(edge)} cases, {failures} mismatches", file=sys.stderr)

    print("BMP codepoint sweep...", file=sys.stderr)
    codepoints = [
        chr(cp)
        for cp in range(0x20, 0x10000)
        if not (0xD800 <= cp <= 0xDFFF) and chr(cp) not in "\r\n"
    ]
    for ctx in ["{}", "a{}b", " {}", "{} ", "1{}2", "{}{}", "{}a", "a{}"]:
        cases = [ctx.replace("{}", c) for c in codepoints]
        bad = run_batch(cases, f"sweep {ctx!r}")
        failures += bad
        print(f"  context {ctx!r}: {len(cases)} cases, {bad} mismatches",
              file=sys.stderr)

    for path in sys.argv[1:]:
        print(f"corpus {path}...", file=sys.stderr)
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = [ln for ln in text.split("\n")[:200000] if ln]
        bad = run_batch(lines, path)
        failures += bad
        print(f"  {len(lines)} lines, {bad} mismatches", file=sys.stderr)

    if failures:
        print(f"\nBYTELEVEL PARITY FAILED: {failures} mismatches")
        sys.exit(1)
    print("\nBYTELEVEL PARITY OK")


if __name__ == "__main__":
    main()
