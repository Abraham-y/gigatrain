#!/usr/bin/env python3
"""Run gigatrain and HuggingFace over degenerate corpora and report survival.

Unlike scripts/benchmark.py this tolerates failure: the whole point is to find
which inputs kill which trainer, so timeouts, OOM-kills, non-zero exits and
crashes are results rather than errors. Reports wall time, peak RSS, exit
status and (when both finish) merge-list parity.

Usage:
  python scripts/degenerate_benchmark.py --corpus-dir /tmp/degen \
      --vocab-size 32000 --timeout 600
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIGATRAIN = ROOT / "gigatrain" / "target" / "release" / "gigatrain"
HF_CLI = ROOT / "scripts" / "hf_train_cli.py"


def run_timed(cmd, out_path, timeout):
    """Run cmd under /usr/bin/time -l. Never raises on trainer failure.

    The child gets its own process group and the whole group is killed on
    timeout. Without that, killing `/usr/bin/time` leaves the actual trainer
    orphaned (PPID 1) and still burning CPU and RAM, which silently corrupts
    every subsequent measurement — observed on the first run of this script.

    Returns dict with wall, rss, rc, status, stderr."""
    t0 = time.perf_counter()
    err = ""
    with open(out_path, "w") as out:
        proc = subprocess.Popen(
            ["/usr/bin/time", "-l"] + cmd,
            stdout=out, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            _, err = proc.communicate(timeout=timeout)
            wall = time.perf_counter() - t0
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                _, err = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                err = ""
            return {"wall": wall, "rss": -1, "rc": None,
                    "status": "TIMEOUT", "stderr": err or ""}

    m = re.search(r"(\d+)\s+maximum resident set size", err)
    rss = int(m.group(1)) if m else -1
    if rc == 0:
        status = "ok"
    elif rc == 101:
        status = "PANIC"
    elif rc == 2:
        status = "rejected"
    elif rc and rc < 0:
        status = f"SIG{-rc}"
    elif rc == 137:
        status = "OOM-KILL"
    else:
        status = f"rc={rc}"
    return {"wall": wall, "rss": rss, "rc": rc, "status": status, "stderr": err}


def fmt_bytes(n):
    if n is None or n < 0:
        return "—"
    v = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if v < 1024 or unit == "GB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}GB"


def first_diff(a_path, b_path):
    """Compare two merge files. Returns (identical, n, detail)."""
    def load(p):
        out = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                l, sep, r = line.partition(" ")
                out.append((l, r) if sep else (line, ""))
        return out
    try:
        a, b = load(a_path), load(b_path)
    except OSError:
        return None, 0, "unreadable"
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return False, i, f"ours={a[i]!r} hf={b[i]!r}"
    if len(a) != len(b):
        return False, n, f"length {len(a)} vs {len(b)}"
    return True, len(a), ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default="/tmp/degen")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--json-out", default=None)
    p.add_argument("--work", default="/tmp/degen_out")
    args = p.parse_args()

    os.makedirs(args.work, exist_ok=True)
    corpora = sorted(Path(args.corpus_dir).glob("*.txt"))
    if not corpora:
        sys.exit(f"no corpora in {args.corpus_dir}")

    modes = [("whitespace", []), ("bytelevel", ["--pretokenizer", "bytelevel"])]
    results = []

    hdr = f"{'corpus':<22} {'mode':<11} {'trainer':<10} {'wall':>9} {'peak RSS':>10} {'status':>10}  parity"
    print(hdr)
    print("-" * len(hdr))

    for corpus in corpora:
        size_mb = corpus.stat().st_size / (1 << 20)
        for mode, mode_flags in modes:
            tag = f"{corpus.stem}.{mode}"
            ours_path = os.path.join(args.work, f"{tag}.ours")
            cmd = [str(GIGATRAIN), "--vocab-size", str(args.vocab_size)] + mode_flags
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            cmd.append(str(corpus))
            ours = run_timed(cmd, ours_path, args.timeout)

            hf = None
            if not args.skip_hf:
                hf_path = os.path.join(args.work, f"{tag}.hf")
                hf_cmd = [args.python, str(HF_CLI), "--vocab-size",
                          str(args.vocab_size), "--pretokenizer", mode,
                          str(corpus)]
                hf = run_timed(hf_cmd, hf_path, args.timeout)

            parity = "—"
            if hf and ours["status"] == "ok" and hf["status"] == "ok":
                same, n, detail = first_diff(ours_path, hf_path)
                parity = f"IDENTICAL ({n})" if same else f"DIFFER @{n} {detail}"

            print(f"{corpus.stem:<22} {mode:<11} {'gigatrain':<10} "
                  f"{ours['wall']:>8.1f}s {fmt_bytes(ours['rss']):>10} "
                  f"{ours['status']:>10}  {parity}")
            if hf:
                print(f"{'':<22} {'':<11} {'HF':<10} "
                      f"{hf['wall']:>8.1f}s {fmt_bytes(hf['rss']):>10} "
                      f"{hf['status']:>10}")
            sys.stdout.flush()

            results.append({
                "corpus": corpus.stem, "size_mb": round(size_mb, 1),
                "mode": mode, "vocab_size": args.vocab_size,
                "gigatrain": {k: ours[k] for k in ("wall", "rss", "rc", "status")},
                "hf": ({k: hf[k] for k in ("wall", "rss", "rc", "status")} if hf else None),
                "parity": parity,
            })

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
