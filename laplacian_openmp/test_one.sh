#!/usr/bin/env bash
# Benchmark one preprocessed grayscale image and export CSV, signed response, edges, and preview.
# Usage: ./test_one.sh [SIZE] [IMAGE_NAME_OR_PATH]
# Example: ./test_one.sh 1024 image_01.bin
# Optional: DATASET_DIR=/path/to/processed RESULTS_DIR=/path/to/results ./test_one.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/../data/processed}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results/laplacian}"

if (( $# > 2 )); then
  echo "Usage: $0 [512|1024|2048] [image_name.bin|image_path.bin]" >&2
  exit 2
fi

SIZE="${1:-1024}"
case "$SIZE" in
  512|1024|2048) ;;
  *) echo "Error: SIZE must be 512, 1024, or 2048 (got '$SIZE')." >&2; exit 2 ;;
esac

if (( $# >= 2 )); then
  IMAGE_ARG="$2"
  if [[ -f "$IMAGE_ARG" ]]; then
    INPUT="$IMAGE_ARG"
  elif [[ "$IMAGE_ARG" == */* ]]; then
    INPUT="$IMAGE_ARG"
  else
    INPUT="$DATASET_DIR/$SIZE/${IMAGE_ARG%.bin}.bin"
  fi
else
  # Select the first available image if no filename is provided.
  shopt -s nullglob
  files=("$DATASET_DIR/$SIZE/"*.bin)
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    echo "Error: no .bin files found under '$DATASET_DIR/$SIZE'." >&2
    exit 1
  fi
  INPUT="${files[0]}"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Error: input image not found: $INPUT" >&2
  echo "Set DATASET_DIR to your local dataset/processed directory." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required for PNG visualization." >&2
  exit 1
fi
if ! python3 -c 'import numpy; import PIL' >/dev/null 2>&1; then
  echo "Error: NumPy and Pillow are required. Install: python3 -m pip install numpy pillow" >&2
  exit 1
fi

make -C "$SCRIPT_DIR" laplacian_benchmark

NAME="$(basename -- "$INPUT" .bin)"
OUTPUT_DIR="$RESULTS_DIR/single/$SIZE"
mkdir -p "$OUTPUT_DIR"
RAW="$OUTPUT_DIR/${NAME}_${SIZE}_laplacian.bin"
EDGES="$OUTPUT_DIR/${NAME}_${SIZE}_edges.bin"
CSV="$OUTPUT_DIR/${NAME}_${SIZE}_metrics.csv"
PNG="$OUTPUT_DIR/${NAME}_${SIZE}_preview.png"

printf '\n[1/2] Benchmark: %s (%sx%s)\n' "$INPUT" "$SIZE" "$SIZE"
"$SCRIPT_DIR/laplacian_benchmark" "$INPUT" "$SIZE" \
  --output "$RAW" --edges-output "$EDGES" | tee "$CSV"

printf '\n[2/2] Creating visualization...\n'
python3 "$SCRIPT_DIR/visualize.py" --input "$INPUT" --edges "$EDGES" --size "$SIZE" --out "$PNG"

printf '\nOutputs:\n  Metrics: %s\n  Response: %s\n  Edges:    %s\n  Preview:  %s\n' \
  "$CSV" "$RAW" "$EDGES" "$PNG"
