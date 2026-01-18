#!/usr/bin/env bash
set -euo pipefail

INPUT_GRAPH="$1"
OUTPUT_DIR="$2"
FINAL_DIR="$3"
SAMPLES="${4:-5}"

echo "--- Running VAE Generator ---"
uv run --locked python -m src.VAE.main --input_graph "$INPUT_GRAPH" --output_dir "$OUTPUT_DIR" --final_dir "$FINAL_DIR" --samples "$SAMPLES"
