#!/usr/bin/env bash
set -euo pipefail

EMP_EDGE="$1"
CLUSTERING="$2"
OUTDIR="$3"
NUM_SAMPLES="${4:-1}"

REPO_DIR="/opt/ec-sbm"

for ((i=0; i<NUM_SAMPLES; i++)); do
  echo "--- Generating EC-SBM sample $i ---"
  SAMPLE_OUT="$OUTDIR/sample_$i"
  mkdir -p "$SAMPLE_OUT"

  (cd "$REPO_DIR" && bash scripts/run_ecsbm.sh "/workspace/$EMP_EDGE" "/workspace/$CLUSTERING" "/workspace/$SAMPLE_OUT")

  echo "EC-SBM sample $i done."
done

echo "All $NUM_SAMPLES EC-SBM samples generated in $OUTDIR"
