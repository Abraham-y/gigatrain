#!/usr/bin/env python3
"""Archive every source the literature audit quotes, with checksums.

An audit of other people's benchmarks should not itself rest on claims the
reader cannot check. Web pages change and repositories move, so every source
quoted in PRIOR_ART.md is fetched here, stored verbatim under
docs/audit-sources/, and recorded in a manifest with its URL, retrieval
timestamp and SHA-256.

Anything that cannot be fetched is recorded as a failure in the manifest rather
than silently omitted, so the audit's coverage is auditable too.

Usage:
  python scripts/fetch_audit_sources.py
  python scripts/fetch_audit_sources.py --verify   # re-check stored checksums
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "audit-sources")
MANIFEST = os.path.join(OUT, "MANIFEST.json")
UA = {"User-Agent": "gigatrain-literature-audit/0.1 (research; contact via repo)"}

# (slug, kind, locator). kind: "url" | "gh"
SOURCES = [
    # --- competing trainers: their own published benchmark claims -----------
    ("yttm-benchmark", "url",
     "https://raw.githubusercontent.com/VKCOM/YouTokenToMe/master/benchmark.md"),
    ("yttm-readme", "url",
     "https://raw.githubusercontent.com/VKCOM/YouTokenToMe/master/README.md"),
    ("rustbpe-readme", "url",
     "https://raw.githubusercontent.com/karpathy/rustbpe/master/README.md"),
    ("gigatoken-readme", "url",
     "https://raw.githubusercontent.com/marcelroed/gigatoken/main/README.md"),
    ("ffbpe-readme", "url",
     "https://raw.githubusercontent.com/tokn-ai/ffbpe/master/README.md"),
    ("ffbpe-benchmarks", "url",
     "https://raw.githubusercontent.com/tokn-ai/ffbpe/master/BENCHMARKS.md"),
    ("fastbytelevelgo-benchmarks", "url",
     "https://raw.githubusercontent.com/yunnian/fast-bytelevel-bpe-go/main/docs/benchmarks.md"),
    ("fastbytelevelgo-readme", "url",
     "https://raw.githubusercontent.com/yunnian/fast-bytelevel-bpe-go/main/README.md"),
    ("sentencepiece-readme", "url",
     "https://raw.githubusercontent.com/google/sentencepiece/master/README.md"),
    ("hf-tokenizers-readme", "url",
     "https://raw.githubusercontent.com/huggingface/tokenizers/main/README.md"),
    ("hf-bpe-trainer-src", "url",
     "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/models/bpe/trainer.rs"),
    ("hf-bpe-word-src", "url",
     "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/models/bpe/word.rs"),

    # --- papers -------------------------------------------------------------
    ("reddy-scaling-abs", "url", "https://arxiv.org/abs/2502.20273"),
    ("zouhar-formal-bpe-abs", "url", "https://arxiv.org/abs/2306.16837"),
    ("sennrich-bpe-abs", "url", "https://arxiv.org/abs/1508.07909"),

    # --- issue threads the project's motivation rests on --------------------
    ("hf-issue-1313", "gh", "huggingface/tokenizers/issues/1313"),
    ("hf-issue-1681", "gh", "huggingface/tokenizers/issues/1681"),
    ("hf-issue-1795", "gh", "huggingface/tokenizers/issues/1795"),
    ("hf-issue-1824", "gh", "huggingface/tokenizers/issues/1824"),
    ("hf-issue-1794", "gh", "huggingface/tokenizers/issues/1794"),
    ("hf-issue-2058", "gh", "huggingface/tokenizers/issues/2058"),
    ("hf-pr-2066", "gh", "huggingface/tokenizers/pulls/2066"),
    ("sp-issue-366", "gh", "google/sentencepiece/issues/366"),
    ("sp-issue-862", "gh", "google/sentencepiece/issues/862"),
    ("sp-issue-1021", "gh", "google/sentencepiece/issues/1021"),
]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def fetch_url(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_gh(locator):
    """Fetch an issue/PR plus its comments through the authenticated gh CLI.

    Stored as JSON rather than HTML: the rendered page is unstable, the API
    payload is not, and it carries state/dates that the audit cites.
    """
    owner_repo, kind, number = locator.rsplit("/", 2)[0], *locator.rsplit("/", 2)[1:]
    api = f"repos/{owner_repo}/{kind}/{number}"
    body = None
    for attempt in range(3):
        r = subprocess.run(["gh", "api", api], capture_output=True, text=True)
        if r.returncode == 0:
            body = r.stdout
            break
        time.sleep(2 * (attempt + 1))
    if body is None:
        raise subprocess.CalledProcessError(1, ["gh", "api", api], stderr=r.stderr)
    comments_api = f"repos/{owner_repo}/issues/{number}/comments?per_page=100"
    try:
        comments = subprocess.run(["gh", "api", comments_api],
                                  capture_output=True, text=True,
                                  check=True).stdout
    except subprocess.CalledProcessError:
        comments = "[]"
    return json.dumps(
        {"item": json.loads(body), "comments": json.loads(comments)},
        indent=2, sort_keys=True,
    ).encode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="re-hash stored files against the manifest")
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.verify:
        with open(MANIFEST) as f:
            man = json.load(f)
        bad = 0
        for e in man["sources"]:
            if e.get("error"):
                continue
            path = os.path.join(OUT, e["file"])
            if not os.path.exists(path):
                print(f"MISSING {e['file']}"); bad += 1; continue
            got = sha256(open(path, "rb").read())
            if got != e["sha256"]:
                print(f"CHANGED {e['file']}\n  manifest {e['sha256']}\n  actual   {got}")
                bad += 1
        print("all sources match manifest" if not bad else f"{bad} problems")
        return 0 if not bad else 1

    entries = []
    for slug, kind, locator in SOURCES:
        ext = ".json" if kind == "gh" else (
            ".md" if locator.endswith(".md") else
            ".rs" if locator.endswith(".rs") else ".html")
        fname = slug + ext
        try:
            data = fetch_gh(locator) if kind == "gh" else fetch_url(locator)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                subprocess.CalledProcessError) as e:
            print(f"  FAIL {slug}: {e}", file=sys.stderr)
            entries.append({"slug": slug, "kind": kind, "locator": locator,
                            "error": str(e)})
            continue
        with open(os.path.join(OUT, fname), "wb") as f:
            f.write(data)
        entries.append({
            "slug": slug, "kind": kind, "locator": locator, "file": fname,
            "bytes": len(data), "sha256": sha256(data),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"  {slug:32s} {len(data):>9,} B  {sha256(data)[:12]}")

    with open(MANIFEST, "w") as f:
        json.dump({
            "note": ("Verbatim sources for PRIOR_ART.md. "
                     "Re-check with: python scripts/fetch_audit_sources.py --verify"),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": entries,
        }, f, indent=2)
    ok = sum(1 for e in entries if "error" not in e)
    print(f"\n{ok}/{len(entries)} sources archived -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
