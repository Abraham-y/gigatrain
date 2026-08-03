#!/usr/bin/env bash
# Full parity CI: unit tests, corpus parity across configs, and fuzzing.
# Milestone 2 gate from CLAUDE.md — must pass before any optimization work.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="${PARITY_WORK_DIR:-$(mktemp -d)}"
mkdir -p "$WORK"

# Pick an interpreter that actually has `tokenizers` installed; `python3` on
# PATH varies between shells (framework vs conda).
if [ -z "${PYTHON:-}" ]; then
  for candidate in python3 /opt/miniconda3/bin/python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c "import tokenizers" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "${PYTHON:-}" ]; then
  echo "error: no python with the 'tokenizers' package found; set PYTHON=..." >&2
  exit 1
fi
echo "using python: $($PYTHON -c 'import sys, tokenizers; print(sys.executable, "tokenizers", tokenizers.__version__)')"

echo "== cargo tests (includes ports of HF's own trainer tests) =="
(cd gigatrain && cargo test --release --quiet)
(cd gigatrain && cargo build --release --quiet)

echo "== generating synthetic corpus =="
if [ ! -f "$WORK/synth.txt" ]; then
  "$PYTHON" - "$WORK/synth.txt" <<'EOF'
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
"$PYTHON" scripts/parity_check.py --files "$WORK/synth.txt" --vocab-size 32000 \
  --special "<|endoftext|>" --special "<pad>"

echo "== parity: multilingual (en+zh), max-token-length =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" "$WORK/hongloumeng.txt" \
  --vocab-size 8000 --max-token-length 16

echo "== parity: min-frequency =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 5000 --min-frequency 4

echo "== parity: colliding special tokens (ID-reuse path) =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 2000 --special "th" --special "e" --special "<eos>"

echo "== parity: limit-alphabet =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 3000 --limit-alphabet 60

echo "== parity: ByteLevel (GPT-2 regex), the production configuration =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" \
  --vocab-size 5000 --pretokenizer bytelevel

echo "== parity: ByteLevel multilingual (en+zh) =="
"$PYTHON" scripts/parity_check.py --files "$WORK/war_and_peace.txt" "$WORK/hongloumeng.txt" \
  --vocab-size 8000 --pretokenizer bytelevel

echo "== ByteLevel pretokenizer differential vs HF (BMP sweep + corpora) =="
"$PYTHON" scripts/check_bytelevel_parity.py "$WORK/war_and_peace.txt"

echo "== determinism: output must not depend on thread count =="
GT=gigatrain/target/release/gigatrain
$GT --vocab-size 3000 "$WORK/war_and_peace.txt" 2>/dev/null > "$WORK/threads_ref.merges"
for t in 1 2 3 7 16; do
  $GT --vocab-size 3000 --threads $t "$WORK/war_and_peace.txt" 2>/dev/null \
    | cmp -s - "$WORK/threads_ref.merges" \
    || { echo "FAIL: output differs at --threads $t"; exit 1; }
done
$GT --vocab-size 3000 --pretokenizer bytelevel "$WORK/war_and_peace.txt" 2>/dev/null \
  > "$WORK/threads_bl_ref.merges"
for t in 1 2 3 7 16; do
  $GT --vocab-size 3000 --pretokenizer bytelevel --threads $t "$WORK/war_and_peace.txt" \
    2>/dev/null | cmp -s - "$WORK/threads_bl_ref.merges" \
    || { echo "FAIL: bytelevel output differs at --threads $t"; exit 1; }
done
echo "  identical across 1,2,3,7,16 threads ($(wc -l < "$WORK/threads_ref.merges") merges)"

echo "== determinism: decorated modes must be reproducible =="
# HF is non-deterministic with continuing_subword_prefix/end_of_word_suffix
# (see PARITY.md), so these modes are checked for self-consistency rather
# than against HF.
for mode in "--wordpiece" "--end-of-word-suffix </w>"; do
  ref=$($GT --vocab-size 2000 $mode "$WORK/war_and_peace.txt" 2>/dev/null | shasum | cut -d' ' -f1)
  for t in 1 4 16; do
    got=$($GT --vocab-size 2000 $mode --threads $t "$WORK/war_and_peace.txt" 2>/dev/null | shasum | cut -d' ' -f1)
    [ "$ref" = "$got" ] || { echo "FAIL: $mode output differs at --threads $t"; exit 1; }
  done
  echo "  $mode reproducible across 1,4,16 threads"
done

echo "== determinism: parallel range readers (forced on a small corpus) =="
# A second reader is only allocated per 64 MiB of input, so on CI-sized files
# `readers` is always 1 and read_range's skip/overshoot rules at range
# boundaries never execute. GIGATRAIN_MIN_RANGE lowers that threshold so the
# splitting path is actually covered without a 128 MB download.
for mode in "" "--pretokenizer bytelevel"; do
  label="${mode:-whitespace}"
  ref=$($GT --vocab-size 3000 $mode "$WORK/war_and_peace.txt" 2>/dev/null | shasum | cut -d' ' -f1)
  for mr in 65536 262144 1048576; do
    for t in 1 3 8; do
      got=$(GIGATRAIN_MIN_RANGE=$mr $GT --vocab-size 3000 $mode --threads $t \
            "$WORK/war_and_peace.txt" 2>/dev/null | shasum | cut -d' ' -f1)
      [ "$ref" = "$got" ] || {
        echo "FAIL: $label differs at GIGATRAIN_MIN_RANGE=$mr --threads $t"
        exit 1
      }
    done
  done
  echo "  $label identical across 3 range sizes x 3 thread counts"
done

echo "== CLI guards: inputs the merge-output format cannot represent =="
# Merges print as "left<space>right", so a decoration containing a space
# corrupts every line rather than producing a detectable error. The Python
# API returns pairs and is unaffected.
printf 'aaaa bbbb aaaa ####\n' > "$WORK/guard.txt"
for flag in --continuing-subword-prefix --end-of-word-suffix; do
  if $GT --vocab-size 100 "$flag" " " "$WORK/guard.txt" >/dev/null 2>&1; then
    echo "FAIL: $flag with a space was accepted; merge output is unparseable"
    exit 1
  fi
  echo "  $flag rejects a value containing a space"
done
# --wordpiece and --continuing-subword-prefix both set the same field; last
# flag silently won before, so the result depended on argument order.
for order in "--wordpiece --continuing-subword-prefix @@" \
             "--continuing-subword-prefix @@ --wordpiece"; do
  if $GT --vocab-size 100 $order "$WORK/guard.txt" >/dev/null 2>&1; then
    echo "FAIL: conflicting flags accepted ($order)"
    exit 1
  fi
done
echo "  --wordpiece and --continuing-subword-prefix conflict is rejected"

echo "== fuzz: 1000 random word tables =="
"$PYTHON" scripts/parity_fuzz.py --trials 1000 --seed 7

echo
echo "ALL PARITY CHECKS PASSED"
