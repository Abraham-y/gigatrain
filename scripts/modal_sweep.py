#!/usr/bin/env python3
"""Intrinsic sweep: does more training data change the tokenizer, and where
does it stop?

The question this settles, cheaply, before committing to anything larger:
vocabularies are trained on nested corpora of increasing size, and each is
compared against the one trained on the most data. If the curves saturate at
the same point regardless of vocabulary size, there is nothing to write about.
If saturation depends on vocabulary size or composition, there is.

Metrics per (corpus size, vocab size):
  - vocab_overlap:  |V_s ∩ V_max| / |V_max|, how much of the reference
                    vocabulary is recovered from a smaller corpus
  - merge_prefix:   leading merges identical to the reference, a much
                    stricter measure than set overlap since BPE merges are
                    order-dependent
  - fertility:      tokens per whitespace-word on held-out text
  - bytes_per_token: compression on the same held-out text

Held-out text is drawn from a parquet not used for training, so it is disjoint
from every training slice.

Usage:
  modal run scripts/modal_sweep.py                       # default grid
  modal run scripts/modal_sweep.py --sizes 100,1000,10000 --vocabs 32000
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable"
    )
    .pip_install("tokenizers==0.22.2", "pyarrow", "maturin")
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=["data", "target", ".git", "**/target", "**/__pycache__"],
    )
)

app = modal.App("gigatrain-sweep", image=image)
volume = modal.Volume.from_name("gigatrain-data", create_if_missing=True)
DATA = "/data"


def _sh(cmd, **kw):
    import subprocess

    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, text=True, **kw)


@app.function(volumes={DATA: volume}, timeout=24 * 3600, cpu=32, memory=131072)
def sweep(sizes, vocabs, heldout_mb=20):
    import json
    import os
    import time

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    _sh("cd /repo && maturin build --release --features python "
        "--manifest-path gigatrain/Cargo.toml", check=True)
    _sh("pip install --force-reinstall --find-links /repo/gigatrain/target/wheels gigatrain",
        check=True)

    import gigatrain
    from tokenizers import Tokenizer

    # --- corpora -------------------------------------------------------
    largest = max(sizes)
    n_parquet = (largest // 4000) + 2  # +1 spare for held-out
    os.makedirs(f"{DATA}/parquet", exist_ok=True)
    for i in range(n_parquet):
        path = f"{DATA}/parquet/{i:03d}.parquet"
        if not os.path.exists(path):
            url = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb/"
                   f"resolve/main/sample/10BT/{i:03d}_00000.parquet")
            _sh(f"curl -sL -o {path} '{url}'", check=True)

    missing = [s for s in sizes if not os.path.exists(f"{DATA}/fineweb_{s}mb.txt")]
    if missing:
        _sh(f"python3 /repo/scripts/slice_fineweb.py {DATA}/parquet/*.parquet "
            f"--sizes-mb {' '.join(map(str, missing))} --out-dir {DATA}", check=True)

    # Held-out from the LAST parquet, which no training slice reaches.
    heldout = f"{DATA}/heldout_{heldout_mb}mb.txt"
    if not os.path.exists(heldout):
        import pyarrow.parquet as pq

        written, parts = 0, []
        pf = pq.ParquetFile(f"{DATA}/parquet/{n_parquet - 1:03d}.parquet")
        for batch in pf.iter_batches(columns=["text"], batch_size=512):
            for t in batch.column("text").to_pylist():
                parts.append(t)
                written += len(t.encode())
                if written >= heldout_mb * 1_000_000:
                    break
            if written >= heldout_mb * 1_000_000:
                break
        open(heldout, "w").write("\n\n".join(parts))
    held_text = open(heldout, encoding="utf-8", errors="ignore").read()
    held_words = len(held_text.split())
    held_bytes = len(held_text.encode())
    print(f"held-out: {held_bytes/1e6:.1f} MB, {held_words} words", flush=True)
    volume.commit()

    # --- train the grid ------------------------------------------------
    models = {}
    timings = {}
    for v in vocabs:
        for s in sizes:
            out = f"/tmp/tok_{s}_{v}.json"
            t0 = time.perf_counter()
            gigatrain.train_tokenizer(
                [f"{DATA}/fineweb_{s}mb.txt"], v, out,
                pretokenizer="bytelevel", special_tokens=["<|endoftext|>"],
            )
            secs = time.perf_counter() - t0
            timings[(s, v)] = secs
            models[(s, v)] = out
            print(f"  trained size={s}MB vocab={v} in {secs:.1f}s", flush=True)

    # --- measure -------------------------------------------------------
    rows = []
    for v in vocabs:
        ref_path = models[(max(sizes), v)]
        ref = json.load(open(ref_path))["model"]
        ref_vocab, ref_merges = set(ref["vocab"]), [tuple(m) for m in ref["merges"]]

        for s in sizes:
            d = json.load(open(models[(s, v)]))["model"]
            vocab, merges = set(d["vocab"]), [tuple(m) for m in d["merges"]]

            overlap = len(vocab & ref_vocab) / len(ref_vocab)
            prefix = 0
            for a, b in zip(merges, ref_merges):
                if a != b:
                    break
                prefix += 1

            tok = Tokenizer.from_file(models[(s, v)])
            ids = tok.encode(held_text).ids
            rows.append({
                "size_mb": s, "vocab": v,
                "vocab_overlap": round(overlap, 4),
                "merge_prefix": prefix,
                "merge_prefix_frac": round(prefix / max(len(ref_merges), 1), 4),
                "fertility": round(len(ids) / held_words, 4),
                "bytes_per_token": round(held_bytes / len(ids), 3),
                "train_seconds": round(timings[(s, v)], 1),
            })
            print(f"  {s}MB v{v}: overlap={overlap:.3f} "
                  f"prefix={prefix} fertility={rows[-1]['fertility']:.3f}",
                  flush=True)

    print("\n=== SWEEP JSON")
    print(json.dumps(rows, indent=2), flush=True)
    return rows


@app.local_entrypoint()
def main(sizes: str = "100,300,1000,3000,10000",
         vocabs: str = "8000,32000,128000"):
    size_list = [int(x) for x in sizes.split(",")]
    vocab_list = [int(x) for x in vocabs.split(",")]
    rows = sweep.remote(size_list, vocab_list)

    print("\n================ INTRINSIC SWEEP ================")
    for v in vocab_list:
        print(f"\nvocab {v} (reference = {max(size_list)} MB):")
        print(f"  {'corpus':>9} {'overlap':>9} {'merge-prefix':>14} "
              f"{'fertility':>10} {'bytes/tok':>10} {'train':>8}")
        for r in [x for x in rows if x["vocab"] == v]:
            print(f"  {r['size_mb']:>7} MB {r['vocab_overlap']:>9.3f} "
                  f"{r['merge_prefix']:>8} ({r['merge_prefix_frac']:>4.2f}) "
                  f"{r['fertility']:>10.3f} {r['bytes_per_token']:>10.3f} "
                  f"{r['train_seconds']:>7.1f}s")
