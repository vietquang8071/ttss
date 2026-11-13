#!/usr/bin/env bash
# Benchmark all .bin files; export metrics and PNG previews by default.
# Usage: ./run_full.sh [DATASET_DIR] [RESULTS_DIR]
# Example: ./run_full.sh ../data/processed ../results/laplacian
# EXPORT_IMAGES=0 ./run_full.sh ...  # CSV-only
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# > 2 )); then
  echo "Usage: $0 [DATASET_DIR] [RESULTS_DIR]" >&2
  exit 2
fi

DATASET_DIR="${1:-${DATASET_DIR:-$SCRIPT_DIR/../data/processed}}"
RESULTS_DIR="${2:-${RESULTS_DIR:-$SCRIPT_DIR/../results/laplacian}}"
OUTPUT_DIR="$RESULTS_DIR/full"
EXPORT_IMAGES="${EXPORT_IMAGES:-1}"
if [[ "$EXPORT_IMAGES" != "0" && "$EXPORT_IMAGES" != "1" ]]; then
  echo "Error: EXPORT_IMAGES must be 0 or 1" >&2
  exit 2
fi

for SIZE in 512 1024 2048; do
  if [[ ! -d "$DATASET_DIR/$SIZE" ]]; then
    echo "Error: missing dataset directory: $DATASET_DIR/$SIZE" >&2
    exit 1
  fi
  shopt -s nullglob
  files=("$DATASET_DIR/$SIZE/"*.bin)
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    echo "Error: no .bin files in $DATASET_DIR/$SIZE" >&2
    exit 1
  fi
  echo "Size $SIZE: ${#files[@]} image(s)"
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required." >&2
  exit 1
fi

make -C "$SCRIPT_DIR" laplacian_benchmark
mkdir -p "$OUTPUT_DIR"
printf '\nRunning full benchmark...\n'
args=(
  --dataset "$DATASET_DIR"
  --exe "$SCRIPT_DIR/laplacian_benchmark"
  --out "$OUTPUT_DIR"
)
if [[ "$EXPORT_IMAGES" == "1" ]]; then
  args+=(--export-images)
fi
python3 "$SCRIPT_DIR/run_all.py" "${args[@]}"

printf '\nOutputs:\n  Raw results: %s\n  Summary:     %s\n' \
  "$OUTPUT_DIR/raw_results.csv" "$OUTPUT_DIR/summary.csv"
if [[ "$EXPORT_IMAGES" == "1" ]]; then
  printf '  PNG previews: %s\n' "$OUTPUT_DIR/previews"
fi
