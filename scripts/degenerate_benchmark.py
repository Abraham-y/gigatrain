#!/usr/bin/env python3
"""Run every available BPE trainer over degenerate corpora and report survival.

Unlike scripts/benchmark.py this tolerates failure: the point is to find which
inputs kill which trainer, so timeouts, OOM-kills, non-zero exits and crashes
are results rather than errors.

Differences from the first version of this script, all of which were objections
a reviewer would raise:

  * `--repeats N` runs each cell N times and reports median plus spread. This
    repo has recorded up to 2x run-to-run variation under memory pressure, so
    single measurements were never defensible.
  * All five trainers, not two. gigatoken in particular is the closest
    competitor and omitting it looked like cherry-picking.
  * Timeouts are reported as censored data (">Ns"), never as a finishing time,
    and the timeout is a parameter you are expected to set generously.
  * The child runs in its own process group and the whole group is killed on
    timeout. Killing `/usr/bin/time` alone leaves the trainer orphaned at
    PPID 1, still burning a core — observed corrupting later measurements.

Usage:
  python scripts/degenerate_benchmark.py --corpus-dir data/real \\
      --vocab-size 32000 --timeout 1200 --repeats 3
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIGATRAIN = ROOT / "gigatrain" / "target" / "release" / "gigatrain"
SCRIPTS = ROOT / "scripts"

# `/usr/bin/time -l` is BSD/macOS and reports peak RSS in BYTES; GNU time on
# Linux wants `-v` and reports KILOBYTES. Getting this wrong is not a silent
# mis-parse — GNU time rejects `-l` and exits 125 without running anything, so
# an entire results table comes back as uniform failure. That happened.
_IS_MAC = sys.platform == "darwin"
TIME_FLAG = "-l" if _IS_MAC else "-v"
_RSS_RE = (re.compile(r"(\d+)\s+maximum resident set size") if _IS_MAC
           else re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)"))
_RSS_SCALE = 1 if _IS_MAC else 1024


def run_timed(cmd, out_path, timeout):
    """Run cmd under /usr/bin/time. Never raises on trainer failure."""
    t0 = time.perf_counter()
    err = ""
    with open(out_path, "w") as out:
        proc = subprocess.Popen(
            ["/usr/bin/time", TIME_FLAG] + cmd,
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

    m = _RSS_RE.search(err)
    rss = int(m.group(1)) * _RSS_SCALE if m else -1
    if rc == 0:
        status = "ok"
    # 125 means `time` itself refused to run the command (bad flag, missing
    # binary). It is a harness fault, not a trainer result, and must never be
    # reported as one.
    elif rc == 125:
        status = "HARNESS-ERR"
    elif rc is not None and rc > 128:
        status = f"SIG{rc - 128}"          # /usr/bin/time relays 128+signal
    elif rc is not None and rc < 0:
        status = f"SIG{-rc}"
    elif rc == 101:
        status = "PANIC"
    # A refusal is not a crash and not a timeout: SentencePiece declines
    # outright when the corpus alphabet cannot support the requested vocab,
    # which is exactly the low-alphabet regime of sentencepiece#1021. Give it
    # its own outcome so it is not scored as a generic error.
    elif "Vocabulary size too high" in err or "vocab_size" in err and "set it to" in err:
        status = "REFUSED"
    elif rc == 2:
        status = "rejected"
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


def merges_from_file(path):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                l, sep, r = line.partition(" ")
                out.append((l, r) if sep else (line, ""))
    except OSError:
        return None
    return out


def compare(a_path, b_path):
    a, b = merges_from_file(a_path), merges_from_file(b_path)
    if a is None or b is None:
        return "—"
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return f"DIFFER@{i}"
    if len(a) != len(b):
        return f"LEN {len(a)}v{len(b)}"
    return f"IDENTICAL ({len(a)})"


def build_jobs(corpus, mode, args, py):
    """(name, cmd, produces_merges) for every trainer that can run this mode."""
    gt = [str(GIGATRAIN), "--vocab-size", str(args.vocab_size)]
    if mode == "bytelevel":
        gt += ["--pretokenizer", "bytelevel"]
    jobs = [("gigatrain", gt + [str(corpus)], True),
            ("HF", [py, str(SCRIPTS / "hf_train_cli.py"),
                     "--vocab-size", str(args.vocab_size),
                     "--pretokenizer", mode, str(corpus)], True)]
    # The rest use their own pretokenizers, so they appear once (under the
    # bytelevel pass) and are time/memory comparisons only.
    if mode == "bytelevel":
        jobs += [
            ("gigatoken", [py, str(SCRIPTS / "gigatoken_train_cli.py"),
                           "--vocab-size", str(args.vocab_size), str(corpus)], False),
            ("rustbpe", [py, str(SCRIPTS / "rustbpe_train_cli.py"),
                         "--vocab-size", str(args.vocab_size), str(corpus)], False),
            ("sentencepiece", [py, str(SCRIPTS / "sp_train_cli.py"),
                               "--vocab-size", str(args.vocab_size),
                               "--model-prefix", "/tmp/sp_degen", str(corpus)], False),
        ]
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default="data/real")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--trainers", default=None,
                   help="comma-separated subset, e.g. gigatrain,HF")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--json-out", default=None)
    p.add_argument("--work", default="/tmp/degen_out")
    args = p.parse_args()

    os.makedirs(args.work, exist_ok=True)
    corpora = sorted(Path(args.corpus_dir).glob("*.txt"))
    if args.only:
        corpora = [c for c in corpora if any(o in c.stem for o in args.only)]
    if not corpora:
        sys.exit(f"no corpora in {args.corpus_dir}")
    want = set(args.trainers.split(",")) if args.trainers else None

    results = []
    hdr = (f"{'corpus':<24} {'mode':<10} {'trainer':<14} {'median':>9} "
           f"{'spread':>9} {'peak RSS':>10} {'status':>9}  parity")
    print(f"# vocab={args.vocab_size} repeats={args.repeats} "
          f"timeout={args.timeout}s")
    print(hdr)
    print("-" * len(hdr))

    for corpus in corpora:
        for mode in ("whitespace", "bytelevel"):
            ref_path = None
            for name, cmd, makes_merges in build_jobs(corpus, mode, args, args.python):
                if want and name not in want:
                    continue
                if args.threads and name == "gigatrain":
                    cmd = cmd + ["--threads", str(args.threads)]
                walls, rsss, statuses, last_out = [], [], [], None
                for r in range(args.repeats):
                    out_path = os.path.join(
                        args.work, f"{corpus.stem}.{mode}.{name}.{r}")
                    res = run_timed(cmd, out_path, args.timeout)
                    statuses.append(res["status"])
                    if res["status"] == "ok":
                        walls.append(res["wall"])
                        if res["rss"] > 0:
                            rsss.append(res["rss"])
                        last_out = out_path
                    else:
                        # A failure is decisive; no point repeating it.
                        break
                ok = all(s == "ok" for s in statuses) and walls
                status = statuses[-1]
                if ok:
                    med = statistics.median(walls)
                    spread = (max(walls) - min(walls)) / med if med else 0.0
                    med_s, spr_s = f"{med:.1f}s", f"±{100*spread/2:.0f}%"
                    rss = statistics.median(rsss) if rsss else -1
                else:
                    med = None
                    med_s = (f">{args.timeout}s" if status == "TIMEOUT"
                             else f"{walls[0]:.1f}s" if walls else "—")
                    spr_s, rss = "—", -1

                parity = "—"
                if makes_merges and ok:
                    if name == "gigatrain":
                        ref_path = last_out
                    elif ref_path:
                        parity = compare(ref_path, last_out)

                print(f"{corpus.stem:<24} {mode:<10} {name:<14} {med_s:>9} "
                      f"{spr_s:>9} {fmt_bytes(rss):>10} {status:>9}  {parity}")
                sys.stdout.flush()
                results.append({
                    "corpus": corpus.stem,
                    "size_mb": round(corpus.stat().st_size / (1 << 20), 1),
                    "mode": mode, "trainer": name, "vocab_size": args.vocab_size,
                    "repeats": args.repeats, "timeout_s": args.timeout,
                    "walls": walls, "median_wall": med, "rss": rss,
                    "status": status, "parity": parity,
                })

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
