#!/usr/bin/env bash
# Self-contained benchmark: gigatrain vs HuggingFace tokenizers, rustbpe and
# SentencePiece, on FineWeb slices. Downloads its own data.
#
# Everything quoted in BENCHMARKS.md came from isolated cloud containers via
# scripts/modal_benchmark.py; this script exists so the numbers can be
# reproduced on any single box without Modal. Thread scaling and allocator
# behaviour are both expected to differ across machines.
#
# Usage:
#   bash scripts/run_full_benchmark.sh [WORKDIR] [SIZES_MB...]
#
# Examples:
#   bash scripts/run_full_benchmark.sh /scratch/bench 100 1000
#   bash scripts/run_full_benchmark.sh /scratch/bench 100 1000 13000
#
# Requirements: rust toolchain, python3, curl, ~3x the largest corpus in disk.
# Recommended: >= 64 GB RAM if including 13000 (HF and SentencePiece will
# thrash or fail below that, which is itself a result worth recording).
set -uo pipefail

WORK="${1:-/tmp/gigatrain-bench}"
shift || true
SIZES=("${@:-100 1000}")
[ $# -gt 0 ] && SIZES=("$@")

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORK"
RESULTS="$WORK/results.tsv"
: > "$RESULTS"

log() { printf '\n=== %s\n' "$*"; }

# ---------------------------------------------------------------- environment
log "environment"
uname -a
python3 -c "import sys; print('python', sys.version.split()[0])"
rustc --version
if [ "$(uname)" = "Linux" ]; then
    nproc | sed 's/^/cores: /'
    free -g | sed -n 2p
    TIMER=(/usr/bin/time -v)
else
    sysctl -n hw.ncpu | sed 's/^/cores: /'
    sysctl -n hw.memsize | awk '{printf "ram: %.0f GB\n", $1/1e9}'
    TIMER=(/usr/bin/time -l)
fi

log "installing python baselines"
python3 -m pip install --quiet --upgrade tokenizers sentencepiece rustbpe maturin || {
    echo "pip install failed; baselines may be missing"; }

log "building gigatrain"
cargo build --release --manifest-path "$REPO/gigatrain/Cargo.toml"
GT="$REPO/gigatrain/target/release/gigatrain"

# ---------------------------------------------------------------------- data
log "preparing corpora"
mkdir -p "$WORK/data"
LARGEST=0
for mb in "${SIZES[@]}"; do [ "$mb" -gt "$LARGEST" ] && LARGEST=$mb; done
# Each FineWeb parquet is ~2 GB and yields ~3.2 GB of text (measured; the
# 4-5 GB estimate undersized the 20 GB corpus once — see BENCHMARKS.md).
NPARQ=$(( (LARGEST / 3000) + 1 ))
for i in $(seq 0 $((NPARQ - 1))); do
    f="$WORK/data/fineweb_$(printf '%03d' $i).parquet"
    if [ ! -f "$f" ]; then
        # -f + temp-then-move + magic check: an HTTP error body cached as a
        # .parquet once corrupted a whole sweep (docs/CORRECTIONS.md, D).
        curl -fsSL -o "$f.part" \
          "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/$(printf '%03d' $i)_00000.parquet" \
          || { echo "error: download failed for $f"; rm -f "$f.part"; exit 1; }
        [ "$(head -c 4 "$f.part")" = "PAR1" ] \
          || { echo "error: $f.part is not a parquet file"; rm -f "$f.part"; exit 1; }
        mv "$f.part" "$f"
    fi
done
python3 "$REPO/scripts/slice_fineweb.py" "$WORK"/data/*.parquet \
    --sizes-mb "${SIZES[@]}" --out-dir "$WORK/data"

# ------------------------------------------------------------------ measure
# record TOOL SIZE SECONDS PEAK_KB
measure() {
    local tool=$1 size=$2 out rc; shift 2
    out=$("${TIMER[@]}" "$@" 2>&1 >/dev/null)
    rc=$?
    # A crashed or OOM-killed trainer must not be recorded as a timing row
    # (standing rule 5: only a timeout may print as a timeout, and only a
    # measurement may print as a measurement).
    if [ $rc -ne 0 ]; then
        printf '%s\t%s\tFAILED(rc=%s)\tNA\n' "$tool" "$size" "$rc" | tee -a "$RESULTS"
        return
    fi
    local secs peak
    if [ "$(uname)" = "Linux" ]; then
        secs=$(grep -o 'Elapsed (wall clock).*' <<<"$out" | awk '{print $NF}')
        secs=$(awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1 }' <<<"$secs")
        peak=$(grep -o 'Maximum resident set size.*' <<<"$out" | awk '{print $NF}')
    else
        secs=$(grep -o '[0-9.]* real' <<<"$out" | awk '{print $1}')
        peak=$(( $(grep -o '[0-9]* *maximum resident' <<<"$out" | awk '{print $1}') / 1024 ))
    fi
    printf '%s\t%s\t%s\t%s\n' "$tool" "$size" "${secs:-NA}" "${peak:-NA}" | tee -a "$RESULTS"
}

for mb in "${SIZES[@]}"; do
    corpus="$WORK/data/fineweb_${mb}mb.txt"
    [ -f "$corpus" ] || { echo "missing $corpus, skipping"; continue; }
    log "corpus ${mb} MB"

    # Warm the page cache so the first tool measured is not penalised.
    cat "$corpus" > /dev/null

    measure gigatrain-bytelevel "$mb" "$GT" --vocab-size 32000 \
        --pretokenizer bytelevel --special "<|endoftext|>" "$corpus"
    measure gigatrain-whitespace "$mb" "$GT" --vocab-size 32000 \
        --special "<|endoftext|>" "$corpus"
    measure hf "$mb" python3 "$REPO/scripts/hf_train_cli.py" \
        --vocab-size 32000 --special "<|endoftext|>" "$corpus"
    measure rustbpe "$mb" python3 "$REPO/scripts/rustbpe_train_cli.py" \
        --vocab-size 32000 "$corpus"
    measure sentencepiece "$mb" python3 "$REPO/scripts/sp_train_cli.py" \
        --vocab-size 32000 --model-prefix "$WORK/sp_${mb}" "$corpus"
done

# --------------------------------------------------------------- thread scan
log "gigatrain thread scaling (1 GB, ByteLevel)"
scan_corpus="$WORK/data/fineweb_1000mb.txt"
if [ -f "$scan_corpus" ]; then
    for t in 1 2 4 8 16 32 64; do
        printf 'threads=%-3s ' "$t"
        "$GT" --vocab-size 32000 --threads "$t" --pretokenizer bytelevel \
            "$scan_corpus" 2>&1 >/dev/null | grep -o 'phase1: [0-9.]*[a-z]*'
    done
fi

log "results"
column -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
echo "raw results: $RESULTS"
echo "NOTE: gigatrain output is byte-identical to HuggingFace (see"
echo "scripts/run_parity_ci.sh). rustbpe and SentencePiece produce different"
echo "tokenizers by design, so those rows compare speed and memory only."
