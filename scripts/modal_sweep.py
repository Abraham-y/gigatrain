#!/usr/bin/env python3
"""Intrinsic sweep: does more training data change the tokenizer, and where
does it stop?

Three compositions, because English web text is the most homogeneous case and
the least likely to show an effect:
  english      FineWeb (English web text)
  code         codeparrot/github-code-clean, Python
  multilingual FineWeb-2, five languages across four scripts, interleaved

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


# Composition -> (dataset, list of configs). URLs are resolved from the
# datasets-server at run time rather than guessed: shard counts differ per
# config, and a guessed index returns a JSON error that `curl -s` will happily
# write to disk as a "parquet" file.
LANGS = ["deu_Latn", "rus_Cyrl", "arb_Arab", "hin_Deva", "jpn_Jpan"]
SOURCES = {
    "english": ("HuggingFaceFW/fineweb", [None]),
    "code": ("codeparrot/github-code-clean", ["Python-all"]),
    # Five languages, four scripts.
    "multilingual": ("HuggingFaceFW/fineweb-2", LANGS),
}


def _resolve_urls(composition, per_config=8):
    """Ask the datasets-server which parquet shards actually exist."""
    import json
    import urllib.request

    if composition == "english":
        # FineWeb's 10BT sample is served directly and is not in the
        # datasets-server parquet listing.
        return [
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/"
            f"main/sample/10BT/{i:03d}_00000.parquet"
            for i in range(4)
        ]

    ds, configs = SOURCES[composition]
    out = []
    for cfg in configs:
        url = f"https://huggingface.co/api/datasets/{ds}/parquet"
        if cfg:
            url += f"/{cfg}"
        with urllib.request.urlopen(url) as r:
            d = json.load(r)
        splits = d if cfg else next(iter(d.values()))
        files = splits.get("train") or next(iter(splits.values()))
        out.extend(files[:per_config])
    return out


def _download_parquet(url, dest):
    """Download and verify. A silent curl will write an error page happily."""
    import os

    if os.path.exists(dest) and _is_parquet(dest):
        return True
    _sh(f"curl -sL -o {dest} '{url}'")
    if _is_parquet(dest):
        return True
    print(f"  SKIP: {url} did not return a parquet file", flush=True)
    if os.path.exists(dest):
        os.remove(dest)
    return False


def _is_parquet(path):
    import os

    try:
        if os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            if f.read(4) != b"PAR1":
                return False
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _build_corpora(composition, sizes, heldout_mb):
    """Download and slice, returning (corpus paths by size, held-out path).

    Slices are nested: the 100 MB corpus is a prefix of the 1 GB one, so size
    is the only variable. Held-out text comes from the last source file, which
    no training slice reaches.
    """
    import os

    import pyarrow.parquet as pq

    urls = _resolve_urls(composition)
    root = f"{DATA}/{composition}"
    os.makedirs(f"{root}/parquet", exist_ok=True)

    targets = sorted(sizes)
    largest = targets[-1]
    paths = {s: f"{root}/corpus_{s}mb.txt" for s in targets}
    heldout = f"{root}/heldout_{heldout_mb}mb.txt"

    def usable(path, want_bytes):
        # Existence is not enough: a crashed earlier run leaves zero-byte
        # corpora behind, and reusing those silently trains every size on an
        # empty file — which produces identical tokenizers and looks like a
        # real (flat) result rather than a failure.
        try:
            return os.path.getsize(path) >= 0.9 * want_bytes
        except OSError:
            return False

    if all(usable(paths[s], s * 1_000_000) for s in targets) and usable(
        heldout, heldout_mb * 1_000_000 * 0.5
    ):
        return paths, heldout
    for p_ in list(paths.values()):
        if os.path.exists(p_):
            os.remove(p_)

    # Pull only as many source files as the largest slice needs, plus one
    # spare reserved for held-out text. For multilingual every language file
    # is fetched regardless: stopping early would silently make the corpus
    # two languages instead of five, and language balance is the whole point
    # of that arm.
    need_bytes = largest * 1_000_000
    fetch_all = composition == "multilingual"
    local = []
    for i, url in enumerate(urls):
        f = f"{root}/parquet/{i:03d}.parquet"
        if not _download_parquet(url, f):
            continue
        local.append(f)
        got = sum(os.path.getsize(x) for x in local)
        # Parquet is compressed ~2-3x, so stop once there is comfortably
        # enough, keeping one file back for held-out.
        if not fetch_all and got * 2 > need_bytes and len(local) >= 2:
            break

    text_col = "content" if composition == "code" else "text"

    def docs_of(path):
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        col = text_col if text_col in cols else ("text" if "text" in cols else cols[0])
        for batch in pf.iter_batches(columns=[col], batch_size=512):
            for t in batch.column(col).to_pylist():
                if t:
                    yield t

    # Round-robin across source files so every prefix is balanced across
    # languages (and across repos, for code) rather than being the first file
    # followed by the second.
    if len(local) < 2:
        raise RuntimeError(
            f"{composition}: only {len(local)} usable parquet files; "
            "cannot build a corpus and hold one back for evaluation"
        )
    streams = [docs_of(f) for f in local[:-1]]
    handles = {s: open(paths[s], "w") for s in targets}
    done = {s: False for s in targets}
    written = 0
    while streams and not all(done.values()):
        for stream in list(streams):
            try:
                t = next(stream)
            except StopIteration:
                streams.remove(stream)
                continue
            doc = t + "\n\n"
            written += len(doc.encode())
            for s in targets:
                if not done[s]:
                    handles[s].write(doc)
                    if written >= s * 1_000_000:
                        done[s] = True
                        handles[s].close()
            if all(done.values()):
                break
    short = []
    for s in targets:
        if not done[s]:
            handles[s].close()
            short.append(s)
    if short:
        raise RuntimeError(
            f"{composition}: sources yielded only {written/1e6:.0f} MB, "
            f"short of {short}. Raise per_config in _resolve_urls or drop "
            f"those sizes; a truncated corpus would silently look like a "
            f"flat result."
        )

    if not os.path.exists(heldout):
        parts, hb = [], 0
        pf = pq.ParquetFile(local[-1])
        cols = pf.schema_arrow.names
        col = text_col if text_col in cols else ("text" if "text" in cols else cols[0])
        for batch in pf.iter_batches(columns=[col], batch_size=512):
            for t in batch.column(col).to_pylist():
                if not t:
                    continue
                parts.append(t)
                hb += len(t.encode())
            if hb >= heldout_mb * 1_000_000:
                break
        open(heldout, "w").write("\n\n".join(parts))
    return paths, heldout


@app.function(volumes={DATA: volume}, timeout=24 * 3600, cpu=32, memory=131072)
def analyse(vocabs, size_mb=1000, heldout_mb=20):
    """Three questions the size sweep leaves on the table.

    1. Cross-domain cost: what does using the wrong domain's tokenizer
       actually cost? Everyone assumes it is bad; few numbers exist.
    2. Rank-stratified overlap: the size sweep shows vocabularies differ
       while performing identically. If the head is stable and only the tail
       moves, that explains it — and is testable by comparing the top-N
       tokens by merge rank rather than the whole set.
    3. Per-language fertility inside the multilingual tokenizer, and whether
       more data makes the split between languages more or less even. This is
       the token-equity question, and it has a real cost attached.
    """
    import json
    import os
    import statistics

    os.environ["PATH"] = f"/root/.cargo/bin:{os.environ['PATH']}"
    _sh("cd /repo && maturin build --release --features python "
        "--manifest-path gigatrain/Cargo.toml", check=True)
    _sh("pip install --force-reinstall --find-links "
        "/repo/gigatrain/target/wheels gigatrain", check=True)

    import gigatrain
    from tokenizers import Tokenizer

    comps = ["english", "code", "multilingual"]
    corpora, heldouts = {}, {}
    for c in comps:
        paths, held = _build_corpora(c, [size_mb], heldout_mb)
        corpora[c], heldouts[c] = paths[size_mb], held

    # Train one tokenizer per (composition, vocab) at a fixed corpus size.
    models = {}
    for c in comps:
        for v in vocabs:
            out = f"{DATA}/models/{c}_{size_mb}_{v}.json"
            os.makedirs(f"{DATA}/models", exist_ok=True)
            if not os.path.exists(out):
                gigatrain.train_tokenizer(
                    [corpora[c]], v, out, pretokenizer="bytelevel",
                    special_tokens=["<|endoftext|>"],
                )
            models[(c, v)] = out
    volume.commit()

    held_text = {c: open(heldouts[c], encoding="utf-8", errors="ignore").read()
                 for c in comps}

    # --- 1. cross-domain -------------------------------------------------
    cross = []
    for v in vocabs:
        for train_c in comps:
            tok = Tokenizer.from_file(models[(train_c, v)])
            for eval_c in comps:
                text = held_text[eval_c]
                ids = tok.encode(text).ids
                cross.append({
                    "vocab": v, "trained_on": train_c, "evaluated_on": eval_c,
                    "bytes_per_token": round(len(text.encode()) / len(ids), 3),
                })

    # --- 2. rank-stratified overlap --------------------------------------
    # Compare each composition's vocabulary against every other, restricted
    # to the first N tokens by id (ids are assigned in merge order, so this
    # is "the N most-important tokens").
    strata = []
    for v in vocabs:
        vocabs_by_c = {}
        for c in comps:
            d = json.load(open(models[(c, v)]))["model"]["vocab"]
            # id -> token, so we can take prefixes by rank
            by_id = sorted(d.items(), key=lambda kv: kv[1])
            vocabs_by_c[c] = [t for t, _ in by_id]
        for n in [256, 1000, 4000, 16000, 64000, v]:
            if n > v:
                continue
            for i, a in enumerate(comps):
                for b in comps[i + 1:]:
                    sa, sb = set(vocabs_by_c[a][:n]), set(vocabs_by_c[b][:n])
                    strata.append({
                        "vocab": v, "top_n": n, "pair": f"{a}|{b}",
                        "overlap": round(len(sa & sb) / n, 4),
                    })

    # --- 3. per-language fertility ---------------------------------------
    # Re-derive the per-language held-out sets from the multilingual sources.
    per_lang = []
    lang_files = {}
    root = f"{DATA}/multilingual/parquet"
    import pyarrow.parquet as pq

    urls = _resolve_urls("multilingual")
    n_per = max(1, len(urls) // len(LANGS))
    for li, lang in enumerate(LANGS):
        # _resolve_urls emits shards grouped per language, in LANGS order.
        idx = li * n_per
        f = f"{root}/{idx:03d}.parquet"
        if not os.path.exists(f):
            continue
        parts, nb = [], 0
        for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=256):
            for t in batch.column("text").to_pylist():
                if t:
                    parts.append(t)
                    nb += len(t.encode())
            if nb >= 2_000_000:
                break
        lang_files[lang] = "\n\n".join(parts)

    for v in vocabs:
        tok = Tokenizer.from_file(models[("multilingual", v)])
        chars, byts = {}, {}
        for lang, text in lang_files.items():
            n = len(tok.encode(text).ids)
            # Characters per token is the fair cross-script measure. Bytes per
            # token is confounded by UTF-8 width: Latin is ~1.4 bytes/char
            # while Devanagari and Japanese are ~3.0, so a byte-based figure
            # flatters non-Latin scripts by a factor of two for reasons that
            # have nothing to do with the tokenizer.
            chars[lang] = len(text) / n
            byts[lang] = len(text.encode()) / n
        if chars:
            per_lang.append({
                "vocab": v,
                "chars_per_token": {k: round(x, 3) for k, x in chars.items()},
                "bytes_per_token": {k: round(x, 3) for k, x in byts.items()},
                "chars_worst_over_best": round(max(chars.values()) / min(chars.values()), 3),
                "chars_stdev": round(statistics.pstdev(chars.values()), 3),
            })

    out = {"cross_domain": cross, "rank_strata": strata, "per_language": per_lang}
    print("\n=== ANALYSIS JSON")
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def deeper(vocabs: str = "8000,32000", size_mb: int = 1000):
    """Cross-domain, rank-stratified overlap, and per-language equity."""
    import json as _json

    vs = [int(x) for x in vocabs.split(",")]
    out = analyse.remote(vs, size_mb)
    print("RAW:", _json.dumps(out))

    print("\n=== 1. CROSS-DOMAIN (bytes/token; higher = better compression)")
    for v in vs:
        print(f"\nvocab {v}:")
        print(f"  {'trained on':>14} | " + " ".join(f"{c:>13}" for c in
              ["english", "code", "multilingual"]))
        for tc in ["english", "code", "multilingual"]:
            row = [next(r["bytes_per_token"] for r in out["cross_domain"]
                        if r["vocab"] == v and r["trained_on"] == tc
                        and r["evaluated_on"] == ec)
                   for ec in ["english", "code", "multilingual"]]
            print(f"  {tc:>14} | " + " ".join(f"{x:>13.3f}" for x in row))

    print("\n=== 2. RANK-STRATIFIED VOCABULARY OVERLAP")
    for v in vs:
        print(f"\nvocab {v}:")
        pairs = sorted({r["pair"] for r in out["rank_strata"]})
        ns = sorted({r["top_n"] for r in out["rank_strata"] if r["vocab"] == v})
        print(f"  {'pair':>26} | " + " ".join(f"{n:>7}" for n in ns))
        for p in pairs:
            row = []
            for n in ns:
                m = [r["overlap"] for r in out["rank_strata"]
                     if r["vocab"] == v and r["pair"] == p and r["top_n"] == n]
                row.append(m[0] if m else float("nan"))
            print(f"  {p:>26} | " + " ".join(f"{x:>7.3f}" for x in row))

    print("\n=== 3. PER-LANGUAGE EQUITY (multilingual tokenizer)")
    print("    chars/token is the fair measure; bytes/token shown for contrast")
    for r in out["per_language"]:
        print(f"\nvocab {r['vocab']}  worst/best (chars) = "
              f"{r['chars_worst_over_best']}")
        for lang, cpt in sorted(r["chars_per_token"].items(), key=lambda kv: -kv[1]):
            print(f"  {lang:>10} {cpt:>7.3f} chars/token "
                  f"({r['bytes_per_token'][lang]:>6.3f} bytes/token)")


@app.function(volumes={DATA: volume}, timeout=24 * 3600, cpu=32, memory=131072)
def sweep(sizes, vocabs, heldout_mb=20, composition="english"):
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
    print(f"=== composition: {composition}", flush=True)
    corpora, heldout = _build_corpora(composition, sizes, heldout_mb)
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
                [corpora[s]], v, out,
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
                "composition": composition, "size_mb": s, "vocab": v,
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
def main(sizes: str = "100,300,1000,3000",
         vocabs: str = "8000,32000,128000",
         compositions: str = "english,code,multilingual"):
    size_list = [int(x) for x in sizes.split(",")]
    vocab_list = [int(x) for x in vocabs.split(",")]

    rows = []
    for comp in compositions.split(","):
        rows.extend(sweep.remote(size_list, vocab_list, 20, comp))

    print("\n================ INTRINSIC SWEEP ================")
    import json as _json
    print("RAW:", _json.dumps(rows))
    for comp in compositions.split(","):
        for v in vocab_list:
            sub = [x for x in rows if x["vocab"] == v and x["composition"] == comp]
            if not sub:
                continue
            print(f"\n{comp} / vocab {v} (reference = {max(size_list)} MB):")
            print(f"  {'corpus':>9} {'overlap':>9} {'fertility':>10} "
                  f"{'bytes/tok':>10}")
            for r in sub:
                print(f"  {r['size_mb']:>7} MB {r['vocab_overlap']:>9.3f} "
                      f"{r['fertility']:>10.3f} {r['bytes_per_token']:>10.3f}")
