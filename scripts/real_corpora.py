#!/usr/bin/env python3
"""Acquire REAL degenerate corpora, as opposed to the generated ones in
scripts/degenerate_corpora.py.

Why this exists: the synthetic generators produce uniform-random content over a
fixed alphabet, which is the *hardest* possible input for BPE — there is no
structure to exploit. Real degenerate data is messier and usually easier. Human
chr21, for example, is not 4 symbols but 10 (ACGT + soft-masked acgt + N/n),
and 14% of it is multi-megabyte runs of `N`, which BPE collapses immediately.
Conclusions drawn from synthetic data alone are therefore suspect, and this
script exists so they can be checked against the real thing.

Every corpus is labelled with its provenance:
  REAL     — bytes as published, only truncated to the size budget
  DERIVED  — a real corpus mechanically transformed (newlines stripped, etc.)

Transformations are named so nothing is silently synthetic.

Usage:
  python scripts/real_corpora.py --out-dir data/real --size-mb 45
  python scripts/real_corpora.py --list
"""
import argparse
import gzip
import json
import os
import sys
import urllib.error
import urllib.request

MB = 1 << 20
UA = {"User-Agent": "gigatrain-benchmark/0.1 (+https://github.com/Abraham-y/gigatrain)"}


def _get(url, timeout=600):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _cached(path, fn):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    data = fn()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------- genomic

UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{}.fa.gz"


def fetch_chr(cache, chrom="chr21"):
    """Human genome assembly hg38, one chromosome, as published by UCSC."""
    raw = _cached(os.path.join(cache, f"{chrom}.fa.gz"),
                  lambda: _get(UCSC.format(chrom)))
    with gzip.open(raw, "rb") as f:
        return f.read()


def dna_real(cache, size):
    """REAL. Human chr21 FASTA exactly as UCSC publishes it: 50-char lines,
    soft-masked repeats in lowercase, N runs at telomeres/centromeres."""
    return fetch_chr(cache)[:size]


def dna_real_oneline(cache, size):
    """DERIVED from dna_real: FASTA header and all newlines removed, so the
    whole file is one boundary-free run of real genomic sequence."""
    data = fetch_chr(cache)
    body = data.split(b"\n", 1)[1]  # drop '>chr21'
    return body.replace(b"\n", b"")[:size]


def dna_real_acgt_only(cache, size):
    """DERIVED from dna_real: uppercased and with N runs removed, i.e. the
    4-symbol alphabet the synthetic generator assumed. Included so the gap
    between real and synthetic genomic data can be attributed."""
    data = fetch_chr(cache)
    body = data.split(b"\n", 1)[1].replace(b"\n", b"").upper()
    return body.replace(b"N", b"")[:size]


# ------------------------------------------------------------- minified JS

# Real, widely-deployed minified bundles. Chosen for size so a useful corpus
# can be built without fetching hundreds of files.
CDN_LIBS = [
    "https://cdnjs.cloudflare.com/ajax/libs/typescript/5.4.5/typescript.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.24.4/babel.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.47.0/min/vs/editor/editor.main.js",
    "https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.9/ace.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.2/chart.umd.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.21/lodash.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.30.1/moment.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/vue/3.4.21/vue.global.prod.js",
    "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.2.67/pdf.min.mjs",
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js",
]


def _minjs_parts(cache):
    parts, seen = [], 0
    for url in CDN_LIBS:
        name = url.rsplit("/", 1)[-1]
        try:
            p = _cached(os.path.join(cache, "js_" + name), lambda u=url: _get(u))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"    skip {name}: {e}", file=sys.stderr)
            continue
        b = open(p, "rb").read()
        parts.append(b)
        seen += len(b)
    if not parts:
        raise RuntimeError("no CDN libraries could be fetched")
    print(f"    {len(parts)} libraries, {seen/MB:.1f} MB distinct", file=sys.stderr)
    return parts


def minjs_real(cache, size):
    """REAL (repeated if short). Production minified JS bundles from cdnjs.
    Long lines, punctuation-dense, few spaces."""
    parts = _minjs_parts(cache)
    out, i = bytearray(), 0
    while len(out) < size:
        out += parts[i % len(parts)]
        out += b"\n"
        i += 1
    if i > len(parts):
        print(f"    NOTE: distinct content cycled {i/len(parts):.1f}x to reach "
              f"{size/MB:.0f} MB — redundancy is inflated", file=sys.stderr)
    return bytes(out[:size])


# ------------------------------------------------------------------- JSON

# npm registry packument documents are real, large, single-line JSON.
NPM_PKGS = ["react", "typescript", "lodash", "express", "webpack", "vue",
            "eslint", "axios", "moment", "rxjs", "@angular/core", "next",
            "jest", "rollup", "svelte", "three"]


def _npm_parts(cache):
    parts, seen = [], 0
    for pkg in NPM_PKGS:
        safe = pkg.replace("/", "_")
        url = "https://registry.npmjs.org/" + pkg.replace("/", "%2F")
        try:
            p = _cached(os.path.join(cache, f"npm_{safe}.json"),
                        lambda u=url: _get(u))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"    skip {pkg}: {e}", file=sys.stderr)
            continue
        b = open(p, "rb").read()
        # Registry documents are already single-line; be certain.
        b = json.dumps(json.loads(b), separators=(",", ":")).encode()
        parts.append(b)
        seen += len(b)
    if not parts:
        raise RuntimeError("no npm documents could be fetched")
    print(f"    {len(parts)} packuments, {seen/MB:.1f} MB distinct", file=sys.stderr)
    return parts


def json_real_oneline(cache, size):
    """REAL (repeated if short). npm registry packument documents, which are
    published as single-line minified JSON with no spaces and no newlines —
    a genuine boundary-free file, not a constructed one."""
    parts = _npm_parts(cache)
    out, i = bytearray(), 0
    while len(out) < size:
        out += parts[i % len(parts)]
        i += 1
    if i > len(parts):
        print(f"    NOTE: distinct content cycled {i/len(parts):.1f}x to reach "
              f"{size/MB:.0f} MB — redundancy is inflated", file=sys.stderr)
    return bytes(out[:size])


# ------------------------------------------------------------------- text

def text_real_cjk(cache, size):
    """REAL (repeated if short). Project Gutenberg 红楼梦 — dense CJK with no
    inter-word spaces."""
    p = _cached(os.path.join(cache, "hongloumeng.txt"),
                lambda: _get("https://www.gutenberg.org/cache/epub/24264/pg24264.txt"))
    b = open(p, "rb").read()
    out = bytearray()
    n = 0
    while len(out) < size:
        out += b
        n += 1
    if n > 1:
        print(f"    NOTE: repeated {n}x to reach {size/MB:.0f} MB", file=sys.stderr)
    return bytes(out[:size])


def text_real_cr_only(cache, size):
    """DERIVED from real English text: Project Gutenberg War and Peace with
    every newline replaced by a carriage return, i.e. classic-Mac endings.
    Under ByteLevel the newline-only cut rule then finds nothing."""
    p = _cached(os.path.join(cache, "war_and_peace.txt"),
                lambda: _get("https://www.gutenberg.org/files/2600/2600-0.txt"))
    b = open(p, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r")
    out = bytearray()
    while len(out) < size:
        out += b
    return bytes(out[:size])


GENERATORS = {
    "dna_real":           (dna_real,           "REAL"),
    "dna_real_oneline":   (dna_real_oneline,   "DERIVED"),
    "dna_real_acgt_only": (dna_real_acgt_only, "DERIVED"),
    "minjs_real":         (minjs_real,         "REAL"),
    "json_real_oneline":  (json_real_oneline,  "REAL"),
    "text_real_cjk":      (text_real_cjk,      "REAL"),
    "text_real_cr_only":  (text_real_cr_only,  "DERIVED"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/real")
    p.add_argument("--cache-dir", default="data/real/.cache")
    p.add_argument("--size-mb", type=int, default=45)
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, (fn, prov) in GENERATORS.items():
            first = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"{name:20s} [{prov:7s}] {first}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    size = args.size_mb * MB
    for name in (args.only or list(GENERATORS)):
        if name not in GENERATORS:
            sys.exit(f"unknown corpus {name!r}; try --list")
        fn, prov = GENERATORS[name]
        out = os.path.join(args.out_dir, f"{name}_{args.size_mb}mb.txt")
        if os.path.exists(out) and os.path.getsize(out) >= size * 0.95:
            print(f"  {name:20s} cached ({os.path.getsize(out)/MB:.1f} MB)")
            continue
        print(f"  {name:20s} [{prov}] building...", flush=True)
        try:
            data = fn(args.cache_dir, size)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"    FAILED: {e}", file=sys.stderr)
            continue
        with open(out, "wb") as f:
            f.write(data)
        print(f"  {name:20s} {len(data)/MB:8.1f} MB  {out}")


if __name__ == "__main__":
    main()
