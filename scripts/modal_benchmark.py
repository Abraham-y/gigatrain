#!/usr/bin/env python3
"""Run the trainer benchmark on Modal, on a many-core Linux box.

Everything in BENCHMARKS.md came from one 10-core macOS laptop. This answers
the two questions that machine could not:

1. Do the performance claims hold on x86-64 Linux with glibc, or are they an
   Apple Silicon / libmalloc artifact?
2. Where does phase 1 stop scaling? At 10 cores the oversubscription effect
   was buried in noise; at 64 it should be obvious.

It also gives HuggingFace and SentencePiece enough RAM to actually finish
12.9 GB, which the laptop could not — so the 13 GB comparison becomes a real
timing result instead of "it thrashed".

Usage:
  modal run scripts/modal_benchmark.py                  # 100MB + 1GB, 64 cores
  modal run scripts/modal_benchmark.py --sizes 100,1000,13000 --cpu 64
  modal run scripts/modal_benchmark.py --thread-scan-only
"""
import modal

REPO = "https://github.com/HuggingFaceFW/fineweb"  # data source, for reference

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "git", "time")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable"
    )
    .pip_install(
        "tokenizers==0.22.2",
        "sentencepiece",
        "rustbpe",
        # The closest competitor (~1.8x). Omitting it from a comparison table
        # reads as cherry-picking whether or not it is.
        "gigatoken",
        "pyarrow",
        "maturin",
    )
    # Source is added last so code edits do not invalidate the heavy layers.
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=["data", "target", ".git", "**/target", "**/__pycache__"],
    )
)

app = modal.App("gigatrain-benchmark", image=image)

# FineWeb slices are reused across runs; downloading 25 GB every time is slow.
volume = modal.Volume.from_name("gigatrain-data", create_if_missing=True)
DATA = "/data"


def _sh(cmd, **kw):
    import subprocess

    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, text=True, **kw)


def _is_parquet(path):
    """Parquet files begin and end with the magic bytes `PAR1`."""
    import os

    try:
        if os.path.getsize(path) < 8:
            return False
        with open(path, "rb") as f:
            if f.read(4) != b"PAR1":
                return False
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _prepare_corpora(sizes):
    """Download FineWeb parquets and slice to the requested sizes."""
    import os

    os.makedirs(f"{DATA}/parquet", exist_ok=True)
    largest = max(sizes)
    # Measured, not estimated: 6 parquets yielded 19,370 MB, i.e. ~3.2 GB of
    # text each, not the 4-5 GB previously assumed. At 20 GB that shortfall
    # silently produced a 19.4 GB corpus and a warning rather than the
    # requested size, which matters because 20 GB is the exact figure in
    # tokenizers#1681 and in milestone 5. Budget 3 GB per parquet and add one
    # for rounding; extra downloads are cached and cost only disk.
    n_parquet = (largest // 3000) + 2
    for i in range(n_parquet):
        path = f"{DATA}/parquet/{i:03d}.parquet"
        # A cached file is only trusted if it is actually a parquet. `curl -s`
        # without -f writes an HTTP error body to the output path and exits 0,
        # so a 404 previously produced a JSON blob named *.parquet that was
        # then cached forever. The same bug silently corrupted the intrinsic
        # sweep (see docs/sweep-results.md).
        if os.path.exists(path) and _is_parquet(path):
            continue
        if os.path.exists(path):
            print(f"  discarding corrupt cached {path}", flush=True)
            os.remove(path)
        url = (
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb/"
            f"resolve/main/sample/10BT/{i:03d}_00000.parquet"
        )
        _sh(f"curl -fsSL -o {path} '{url}'", check=True)
        if not _is_parquet(path):
            raise RuntimeError(f"{url} did not return a parquet file")

    missing = [mb for mb in sizes if not os.path.exists(f"{DATA}/fineweb_{mb}mb.txt")]
    if missing:
        _sh(
            f"python3 /repo/scripts/slice_fineweb.py {DATA}/parquet/*.parquet "
            f"--sizes-mb {' '.join(str(m) for m in missing)} --out-dir {DATA}",
            check=True,
        )
    volume.commit()


def _measure(tool, size, cmd):
    """Run cmd under /usr/bin/time -v; return (seconds, peak_mb)."""
    import re
    import subprocess
    import time

    print(f"\n--- {tool} @ {size} MB", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        f"/usr/bin/time -v {cmd}",
        shell=True,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    wall = time.perf_counter() - t0
    peak_kb = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    if m:
        peak_kb = int(m.group(1))
    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    print(f"    {wall:.1f}s  {peak_kb/1024:.0f} MB  {status}", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-1500:], flush=True)
    return wall, peak_kb / 1024, proc.returncode


@app.function(
    volumes={DATA: volume},
    timeout=24 * 3600,
    # cpu/memory are overridden per-call from main().
)
def benchmark(sizes, cpu, run_baselines=True, thread_scan=True):
    import json
    import os

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"

    print("=== environment", flush=True)
    _sh("uname -a")
    _sh("nproc")
    _sh("free -g | head -2")
    _sh("rustc --version")
    _sh("cat /proc/cpuinfo | grep 'model name' | head -1")

    print("\n=== building gigatrain", flush=True)
    _sh("cargo build --release --manifest-path /repo/gigatrain/Cargo.toml", check=True)
    gt = "/repo/gigatrain/target/release/gigatrain"

    print("\n=== preparing corpora", flush=True)
    _prepare_corpora(sizes)

    results = []
    for mb in sizes:
        corpus = f"{DATA}/fineweb_{mb}mb.txt"
        if not os.path.exists(corpus):
            print(f"missing {corpus}, skipping", flush=True)
            continue
        # Warm the page cache so the first tool is not penalised.
        _sh(f"cat {corpus} > /dev/null")

        jobs = [
            ("gigatrain-bytelevel",
             f"{gt} --vocab-size 32000 --pretokenizer bytelevel "
             f'--special "<|endoftext|>" {corpus}'),
            ("gigatrain-whitespace",
             f'{gt} --vocab-size 32000 --special "<|endoftext|>" {corpus}'),
        ]
        if run_baselines:
            jobs += [
                ("hf",
                 f"python3 /repo/scripts/hf_train_cli.py --vocab-size 32000 "
                 f'--special "<|endoftext|>" {corpus}'),
                ("rustbpe",
                 f"python3 /repo/scripts/rustbpe_train_cli.py "
                 f"--vocab-size 32000 {corpus}"),
                ("sentencepiece",
                 f"python3 /repo/scripts/sp_train_cli.py --vocab-size 32000 "
                 f"--model-prefix /tmp/sp_{mb} {corpus}"),
            ]

        for tool, cmd in jobs:
            secs, peak_mb, rc = _measure(tool, mb, cmd)
            results.append(
                {"tool": tool, "size_mb": mb, "seconds": round(secs, 2),
                 "peak_mb": round(peak_mb), "rc": rc, "cpu": cpu}
            )

    scan = []
    if thread_scan:
        scan_corpus = f"{DATA}/fineweb_1000mb.txt"
        if os.path.exists(scan_corpus):
            print("\n=== gigatrain thread scaling (1 GB, ByteLevel)", flush=True)
            for t in [1, 2, 4, 8, 16, 32, 48, 64, 96]:
                if t > cpu * 2:
                    break
                secs, peak_mb, rc = _measure(
                    f"threads={t}", 1000,
                    f"{gt} --vocab-size 32000 --threads {t} "
                    f"--pretokenizer bytelevel {scan_corpus}",
                )
                scan.append({"threads": t, "seconds": round(secs, 2),
                             "peak_mb": round(peak_mb)})

    print("\n=== RESULTS JSON", flush=True)
    print(json.dumps({"results": results, "thread_scan": scan, "cpu": cpu},
                     indent=2), flush=True)
    return {"results": results, "thread_scan": scan, "cpu": cpu}


@app.function(volumes={DATA: volume}, timeout=24 * 3600)
def parity_at_scale(size_mb: int = 13000, pretokenizer: str = "whitespace",
                    vocab_size: int = 32000):
    """Diff gigatrain's merge list against HF's at full corpus scale.

    The benchmark functions discard stdout, so nothing above 1 GB had ever
    been compared merge-for-merge — the headline number asserted a parity
    result that was never computed. This computes it.
    """
    import os

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    _sh("cargo build --release --manifest-path /repo/gigatrain/Cargo.toml", check=True)
    gt = "/repo/gigatrain/target/release/gigatrain"
    corpus = f"{DATA}/fineweb_{size_mb}mb.txt"
    if not os.path.exists(corpus):
        _prepare_corpora([size_mb])

    special = '--special "<|endoftext|>"'
    pt_gt = f"--pretokenizer {pretokenizer}"
    pt_hf = f"--pretokenizer {pretokenizer}"

    print(f"=== gigatrain ({pretokenizer}, vocab {vocab_size})", flush=True)
    ours, ours_s, _ = _measure(
        "gigatrain", size_mb,
        f"{gt} --vocab-size {vocab_size} {special} {pt_gt} {corpus} > /tmp/ours.merges",
    )
    print(f"=== HuggingFace ({pretokenizer}) — this is the slow one", flush=True)
    theirs, theirs_s, _ = _measure(
        "hf", size_mb,
        f"python3 /repo/scripts/hf_train_cli.py --vocab-size {vocab_size} "
        f"{special} {pt_hf} {corpus} > /tmp/hf.merges",
    )

    a = open("/tmp/ours.merges").read().splitlines()
    b = open("/tmp/hf.merges").read().splitlines()
    identical = a == b
    first_diff = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            first_diff = (i, x, y)
            break
    result = {
        "size_mb": size_mb, "pretokenizer": pretokenizer,
        "vocab_size": vocab_size,
        "ours_merges": len(a), "hf_merges": len(b),
        "identical": identical, "first_diff": first_diff,
        "ours_seconds": round(ours, 1), "hf_seconds": round(theirs, 1),
    }
    print(f"\n=== PARITY AT {size_mb} MB ({pretokenizer}): "
          f"{'IDENTICAL' if identical else 'DIVERGED'}", flush=True)
    print(result, flush=True)
    return result


@app.function(volumes={DATA: volume}, timeout=3600)
def parity_check():
    """Run the full parity gate on Linux, as a cross-check of CI."""
    import os

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    r = _sh("cd /repo && bash scripts/run_parity_ci.sh")
    return r.returncode


@app.local_entrypoint()
def parity(size_mb: int = 13000, pretokenizer: str = "whitespace",
           vocab_size: int = 32000, cpu: int = 16, memory: int = 192):
    """Verify merge-list parity at full scale: `modal run ... ::parity`."""
    out = parity_at_scale.with_options(cpu=cpu, memory=memory * 1024).remote(
        size_mb, pretokenizer, vocab_size
    )
    print("\n================ PARITY AT SCALE ================")
    for k, v in out.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def main(sizes: str = "100,1000", cpu: int = 64, memory: int = 192,
         baselines: bool = True, thread_scan: bool = True):
    """Kick off the benchmark with the requested shape."""
    size_list = [int(s) for s in sizes.split(",")]
    print(f"requesting {cpu} CPU / {memory} GiB for sizes {size_list}")
    out = benchmark.with_options(cpu=cpu, memory=memory * 1024).remote(
        size_list, cpu, baselines, thread_scan
    )

    print("\n================ SUMMARY ================")
    by_size = {}
    for r in out["results"]:
        by_size.setdefault(r["size_mb"], []).append(r)
    for mb, rows in sorted(by_size.items()):
        print(f"\n{mb} MB:")
        base = next((r["seconds"] for r in rows
                     if r["tool"] == "gigatrain-bytelevel"), None)
        for r in sorted(rows, key=lambda x: x["seconds"]):
            rel = f"{r['seconds']/base:6.1f}x" if base and base > 0 else "     -"
            flag = "" if r["rc"] == 0 else "  (FAILED)"
            print(f"  {r['tool']:<22} {r['seconds']:8.1f}s "
                  f"{r['peak_mb']:7d} MB  {rel}{flag}")
    if out["thread_scan"]:
        print("\nthread scaling (1 GB ByteLevel, total wall):")
        for row in out["thread_scan"]:
            print(f"  threads={row['threads']:<3} {row['seconds']:7.2f}s "
                  f"{row['peak_mb']:6d} MB")


# --------------------------------------------------------------------------
# Degenerate-corpus study on real data.
#
# The laptop run used synthetic corpora and two trainers. This runs the real
# corpora (human chr21, npm packuments, cdnjs bundles) across every installed
# trainer with repeats, on a machine that is not the author's laptop.
# --------------------------------------------------------------------------


@app.function(volumes={DATA: volume}, timeout=24 * 3600)
def degenerate(size_mb: int, vocab_size: int, timeout: int, repeats: int,
               only: str = ""):
    import os

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    print("=== environment", flush=True)
    _sh("uname -a"); _sh("nproc")

    print("\n=== building gigatrain", flush=True)
    _sh("cargo build --release --manifest-path /repo/gigatrain/Cargo.toml", check=True)

    corpus_dir = f"{DATA}/real_{size_mb}mb"
    print("\n=== acquiring real corpora", flush=True)
    _sh(f"python3 /repo/scripts/real_corpora.py --out-dir {corpus_dir} "
        f"--cache-dir {DATA}/real_cache --size-mb {size_mb}", check=True)
    volume.commit()
    _sh(f"ls -la {corpus_dir}")

    sel = " ".join(f"--only {o}" for o in only.split(",") if o)
    out_json = "/tmp/degen_real.json"
    print("\n=== benchmark", flush=True)
    _sh(f"python3 /repo/scripts/degenerate_benchmark.py "
        f"--corpus-dir {corpus_dir} --vocab-size {vocab_size} "
        f"--timeout {timeout} --repeats {repeats} {sel} "
        f"--json-out {out_json}")

    import json as _json
    try:
        with open(out_json) as f:
            return _json.load(f)
    except OSError:
        return []


@app.local_entrypoint()
def degen(size_mb: int = 45, vocab_size: int = 32000, timeout: int = 900,
          repeats: int = 3, cpu: int = 16, memory: int = 64, only: str = ""):
    """Degenerate-corpus study on REAL data:
    `modal run scripts/modal_benchmark.py::degen`"""
    rows = degenerate.with_options(cpu=cpu, memory=memory * 1024).remote(
        size_mb, vocab_size, timeout, repeats, only
    )
    print("\n================ DEGENERATE (REAL CORPORA) ================")
    for r in rows:
        # Only a TIMEOUT status may be rendered as ">Ns". Rendering every
        # non-completion that way once turned a harness failure (rc=125, GNU
        # time rejecting a BSD flag) into a table of fabricated timeouts.
        if r.get("median_wall"):
            med = f"{r['median_wall']:.1f}s"
        elif r.get("status") == "TIMEOUT":
            med = f">{r['timeout_s']}s"
        else:
            med = "—"
        rss = f"{r['rss']/(1<<20):.0f}MB" if r.get("rss", -1) > 0 else "—"
        print(f"  {r['corpus']:<24} {r['mode']:<10} {r['trainer']:<14} "
              f"{med:>9} {rss:>9} {r['status']:>9}  {r['parity']}")


# --------------------------------------------------------------------------
# Controlled core-count sweep.
#
# BENCHMARKS.md has long said HuggingFace "gets slower with more cores", based
# on 9.7 s on a 10-core macOS laptop against 181 s on a 64-core Linux box. That
# comparison changes ISA, OS, allocator and machine as well as core count, so
# it was retracted as an uncontrolled measurement. This is the experiment that
# actually tests it: ONE box, ONE binary, ONE corpus, varying only the number
# of threads rayon is allowed to use.
# --------------------------------------------------------------------------


@app.function(volumes={DATA: volume}, timeout=12 * 3600)
def thread_sweep(size_mb: int, vocab_size: int, threads: str, repeats: int):
    import json
    import os
    import statistics

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    _sh("uname -a"); _sh("nproc")
    _sh("cargo build --release --manifest-path /repo/gigatrain/Cargo.toml", check=True)
    gt = "/repo/gigatrain/target/release/gigatrain"

    _prepare_corpora([size_mb])
    corpus = f"{DATA}/fineweb_{size_mb}mb.txt"
    _sh(f"cat {corpus} > /dev/null")

    tlist = [int(t) for t in threads.split(",")]
    rows = []
    for t in tlist:
        for tool in ("hf", "gigatrain"):
            if tool == "hf":
                # tokenizers parallelises with rayon, which reads this.
                cmd = (f"RAYON_NUM_THREADS={t} python3 /repo/scripts/hf_train_cli.py "
                       f"--vocab-size {vocab_size} {corpus}")
            else:
                cmd = (f"{gt} --vocab-size {vocab_size} --threads {t} {corpus}")
            walls, peaks, rc_last = [], [], 0
            for _ in range(repeats):
                secs, peak_mb, rc = _measure(f"{tool} t={t}", size_mb, cmd)
                rc_last = rc
                if rc == 0:
                    walls.append(secs); peaks.append(peak_mb)
            row = {
                "tool": tool, "threads": t, "size_mb": size_mb,
                "vocab_size": vocab_size, "repeats": repeats,
                "walls": [round(w, 2) for w in walls],
                "median_s": round(statistics.median(walls), 2) if walls else None,
                "peak_mb": round(statistics.median(peaks)) if peaks else None,
                "rc": rc_last,
            }
            rows.append(row)
            print(f"  => {tool} threads={t}: {row['median_s']}s "
                  f"{row['peak_mb']}MB", flush=True)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


@app.local_entrypoint()
def threads(size_mb: int = 100, vocab_size: int = 32000,
            threads: str = "1,2,4,8,16,32,64", repeats: int = 3,
            cpu: int = 64, memory: int = 128):
    """Controlled core-count sweep on one box:
    `modal run scripts/modal_benchmark.py::threads`"""
    rows = thread_sweep.with_options(cpu=cpu, memory=memory * 1024).remote(
        size_mb, vocab_size, threads, repeats
    )
    print("\n============ CONTROLLED CORE-COUNT SWEEP (one box) ============")
    print(f"{'threads':>8}  {'HF':>12}  {'gigatrain':>12}")
    by = {}
    for r in rows:
        by.setdefault(r["threads"], {})[r["tool"]] = r
    base_hf = by[min(by)]["hf"]["median_s"] if by else None
    for t in sorted(by):
        hf = by[t].get("hf", {}).get("median_s")
        gt = by[t].get("gigatrain", {}).get("median_s")
        rel = f"  ({hf/base_hf:.2f}x vs 1 thread)" if hf and base_hf else ""
        print(f"{t:>8}  {hf if hf else '—':>12}  {gt if gt else '—':>12}{rel}")
