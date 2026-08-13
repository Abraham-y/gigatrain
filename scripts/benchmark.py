#!/usr/bin/env python3
"""Benchmark gigabpe vs HF tokenizers' BpeTrainer on the same files.

Per CLAUDE.md benchmarking rules, reports wall time, peak RSS, and merge-list
parity together — a speedup with different output is not a speedup. Each
trainer runs as a subprocess under /usr/bin/time -l (macOS) so peak RSS is
the whole process, fairly measured.

Usage:
  python scripts/benchmark.py --files data/fineweb_100mb.txt --vocab-size 32000
  python scripts/benchmark.py --files a.txt --vocab-size 32000 --skip-hf   # ours only
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIGABPE = ROOT / "gigabpe" / "target" / "release" / "gigabpe"
HF_CLI = ROOT / "scripts" / "hf_train_cli.py"


def run_timed(cmd, merges_path):
    """Run cmd under /usr/bin/time -l; stdout -> merges_path.
    Returns (wall_seconds, peak_rss_bytes, stderr_text)."""
    t0 = time.perf_counter()
    with open(merges_path, "w") as out:
        proc = subprocess.run(
            ["/usr/bin/time", "-l"] + cmd,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
        )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {cmd}")
    m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    rss = int(m.group(1)) if m else -1
    return wall, rss, proc.stderr


def fmt_bytes(n):
    if n < 0:
        return "?"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--min-frequency", type=int, default=0)
    p.add_argument("--special", action="append", default=[])
    p.add_argument("--max-token-length", type=int, default=None)
    p.add_argument("--skip-hf", action="store_true",
                   help="only run gigabpe (for sizes where HF is impractical)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    corpus_bytes = sum(os.path.getsize(f) for f in args.files)
    out_dir = Path(args.out_dir) if args.out_dir else Path("/tmp")
    tag = f"v{args.vocab_size}_{corpus_bytes // 1_000_000}mb"

    common = ["--vocab-size", str(args.vocab_size), "--min-frequency", str(args.min_frequency)]
    for s in args.special:
        common += ["--special", s]
    if args.max_token_length is not None:
        common += ["--max-token-length", str(args.max_token_length)]

    print(f"corpus: {corpus_bytes / 1e6:.0f} MB across {len(args.files)} file(s), "
          f"vocab_size={args.vocab_size}", file=sys.stderr)

    ours_merges = out_dir / f"ours_{tag}.merges"
    print("running gigabpe...", file=sys.stderr)
    ours_wall, ours_rss, ours_err = run_timed(
        [str(GIGABPE)] + common + args.files, ours_merges
    )
    stats = [l for l in ours_err.splitlines() if "phase1" in l]
    if stats:
        print(f"  {stats[0].strip()}", file=sys.stderr)

    hf_wall = hf_rss = None
    if not args.skip_hf:
        hf_merges = out_dir / f"hf_{tag}.merges"
        print("running HF tokenizers (all cores)...", file=sys.stderr)
        hf_wall, hf_rss, _ = run_timed(
            [sys.executable, str(HF_CLI)] + common + args.files, hf_merges
        )
        parity = ours_merges.read_bytes() == hf_merges.read_bytes()
    else:
        parity = None

    print()
    print(f"{'':>12} {'wall':>10} {'peak RSS':>10}")
    print(f"{'gigabpe':>12} {ours_wall:>9.1f}s {fmt_bytes(ours_rss):>10}")
    if hf_wall is not None:
        print(f"{'HF':>12} {hf_wall:>9.1f}s {fmt_bytes(hf_rss):>10}")
        print(f"\nspeedup: {hf_wall / ours_wall:.1f}x   parity: "
              f"{'IDENTICAL' if parity else 'MISMATCH -- NOT A VALID RESULT'}")
        if not parity:
            sys.exit(1)


if __name__ == "__main__":
    main()
