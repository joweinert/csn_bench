#!/usr/bin/env bash
set -euo pipefail

EMP_EDGE="$1"          # full empirical edge list (tsv)
CLUSTERING="$2"        # clustering file (tsv)
OUTDIR="$3"            # output dir
NUM_SAMPLES="${4:-1}"  # default to 1

GEN_DIR="/opt/lanne2_networks/generate_synthetic_networks"

mkdir -p "$OUTDIR/00_clean"

# Step 0: clean outliers
python "$GEN_DIR/clean_outliers.py" \
  --input-network "$EMP_EDGE" \
  --input-clustering "$CLUSTERING" \
  --output-folder "$OUTDIR/00_clean"

EDGE_BASENAME="$(basename "$EMP_EDGE")"
CLUST_BASENAME="$(basename "$CLUSTERING")"

GC_EDGE="$OUTDIR/00_clean/$EDGE_BASENAME"
GC_CLUSTER="$OUTDIR/00_clean/$CLUST_BASENAME"

for ((i=0; i<NUM_SAMPLES; i++)); do
  echo "--- Generating RECCS sample $i ---"
  SAMPLE_DIR="sample_$i"

  mkdir -p "$OUTDIR/01_sbm/$SAMPLE_DIR" \
           "$OUTDIR/02_reccs/$SAMPLE_DIR" \
           "$OUTDIR/03_outliers/$SAMPLE_DIR"

  # Step 1: SBM on clustered subnetwork
  python "$GEN_DIR/gen_SBM.py" -f "$GC_EDGE" -c "$GC_CLUSTER" -o "$OUTDIR/01_sbm/$SAMPLE_DIR"
  SBM_EDGE="$OUTDIR/01_sbm/$SAMPLE_DIR/syn_sbm.tsv"

  # Step 2: RECCS
  python "$GEN_DIR/reccs.py" \
    -f "$SBM_EDGE" \
    -c "$GC_CLUSTER" \
    -o "$OUTDIR/02_reccs/$SAMPLE_DIR" \
    -ef "$GC_EDGE"

  RECCS_EDGE="$OUTDIR/02_reccs/$SAMPLE_DIR/ce_plusedges_v2.tsv"

  # Step 3: adds outliers back (strategy 1)
  python "$GEN_DIR/outliers_strategy1.py" \
    -f "$EMP_EDGE" \
    -c "$CLUSTERING" \
    -o "$OUTDIR/03_outliers/$SAMPLE_DIR" \
    -s "$RECCS_EDGE"

  echo "RECCS sample $i done."
done

echo "All $NUM_SAMPLES RECCS samples generated in $OUTDIR"
