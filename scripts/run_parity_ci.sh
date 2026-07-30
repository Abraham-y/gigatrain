#!/usr/bin/env bash
# Full parity CI: unit tests, corpus parity across configs, and fuzzing.
# Milestone 2 gate from CLAUDE.md — must pass before any optimization work.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="${PARITY_WORK_DIR:-$(mktemp -d)}"
mkdir -p "$WORK"

echo "== cargo tests (includes ports of HF's own trainer tests) =="
(cd gigatrain && cargo test --release --quiet)
(cd gigatrain && cargo build --release --quiet)

echo "== generating synthetic corpus =="
if [ ! -f "$WORK/synth.txt" ]; then
  python3 - "$WORK/synth.txt" <<'EOF'
import sys
import corpus
_, texts = corpus.make_corpus(12_000_000)
docs = [t for t in texts if len(t) > 200]
open(sys.argv[1], "w").write("\n\n".join(docs))
EOF
fi

echo "== fetching real corpora =="
[ -f "$WORK/war_and_peace.txt" ] || \
  curl -sL -o "$WORK/war_and_peace.txt" https://www.gutenberg.org/files/2600/2600-0.txt
[ -f "$WORK/hongloumeng.txt" ] || \
  curl -sL -o "$WORK/hongloumeng.txt" https://www.gutenberg.org/cache/epub/24264/pg24264.txt

echo "== parity: synthetic, 32k vocab, special tokens =="
python3 scripts/parity_check.py --files "$WORK/synth.txt" --vocab-size 32000 \
  --special "<|endoftext|>" --special "<pad>"

echo "== parity: multilingual (en+zh), max-token-length =="
python3 scripts/parity_check.py --files "$WORK/war_and_peace.txt" "$WORK/hongloumeng.txt" \
  --vocab-size 8000 --max-token-length 16

echo "== parity: min-frequency =="
python3 scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 5000 --min-frequency 4

echo "== parity: colliding special tokens (ID-reuse path) =="
python3 scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 2000 --special "th" --special "e" --special "<eos>"

echo "== parity: limit-alphabet =="
python3 scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 3000 --limit-alphabet 60

echo "== fuzz: 1000 random word tables =="
python3 scripts/parity_fuzz.py --trials 1000 --seed 7

echo
echo "ALL PARITY CHECKS PASSED"
