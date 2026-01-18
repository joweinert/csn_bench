#!/usr/bin/env bash
set -euo pipefail

# ----- Timer Start -----
start_time=$(date +%s)
echo "Start Time: $(date)"

# Always run from repo root
cd "$(dirname "$0")"

# ----- UV: reproducible env from uv.lock -----
uv lock --check
uv sync --locked
uv run --locked python -m src.data.load_data

# ----- Run ID -----
RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${RUN_ID}"

echo "=========================================================="
echo "Starting Run: ${RUN_ID}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=========================================================="

mkdir -p "${RESULTS_DIR}/RECCS"
mkdir -p "${RESULTS_DIR}/EC-SBM"
mkdir -p "${RESULTS_DIR}/VAE"
mkdir -p "${RESULTS_DIR}/final_dataset/VAE"

# ----- Docker: build images -----
docker compose --profile reccs build reccs
docker compose --profile ecsbm build ecsbm

# ----- Settings -----
# Default samples if not provided
NUM_SAMPLES="${1:-30}"

echo "Using ${NUM_SAMPLES} samples per experiment."

echo "=========================================================="
echo "Running Karate Experiments"
echo "=========================================================="
bash scripts/run_reccs.sh data/karate.tsv data/karate.clustering.tsv "${RESULTS_DIR}/RECCS/karate" "$NUM_SAMPLES"
bash scripts/run_ecsbm.sh data/karate.tsv data/karate.clustering.tsv "${RESULTS_DIR}/EC-SBM/karate" "$NUM_SAMPLES"
bash scripts/run_vae.sh data/karate.tsv "${RESULTS_DIR}/VAE/karate" "${RESULTS_DIR}/final_dataset/VAE" "$NUM_SAMPLES"

echo "=========================================================="
echo "Running Polbooks Experiments"
echo "=========================================================="
bash scripts/run_reccs.sh data/polbooks.tsv data/polbooks.clustering.tsv "${RESULTS_DIR}/RECCS/polbooks" "$NUM_SAMPLES"
bash scripts/run_ecsbm.sh data/polbooks.tsv data/polbooks.clustering.tsv "${RESULTS_DIR}/EC-SBM/polbooks" "$NUM_SAMPLES"
bash scripts/run_vae.sh data/polbooks.tsv "${RESULTS_DIR}/VAE/polbooks" "${RESULTS_DIR}/final_dataset/VAE" "$NUM_SAMPLES"

echo "=========================================================="
echo "Running Football Experiments"
echo "=========================================================="
bash scripts/run_reccs.sh data/football.tsv data/football.clustering.tsv "${RESULTS_DIR}/RECCS/football" "$NUM_SAMPLES"
bash scripts/run_ecsbm.sh data/football.tsv data/football.clustering.tsv "${RESULTS_DIR}/EC-SBM/football" "$NUM_SAMPLES"
bash scripts/run_vae.sh data/football.tsv "${RESULTS_DIR}/VAE/football" "${RESULTS_DIR}/final_dataset/VAE" "$NUM_SAMPLES"

echo "=========================================================="
echo "Standardizing Outputs"
echo "=========================================================="
echo "Standardizing outputs..."
uv run python src/data/standardize_outputs.py --run_dir "$RESULTS_DIR"

echo "=========================================================="
echo "Running Analysis & Visualization"
echo "=========================================================="

mkdir -p "$RESULTS_DIR/analysis"
mkdir -p "$RESULTS_DIR/plots"

echo "[1/5] Basic Metrics..."
uv run python -m src.analysis.basic_metrics --run_dir "$RESULTS_DIR"

echo "[2/5] Validation Utility Metrics..."
uv run python -m src.analysis.validation_utility --run_dir "$RESULTS_DIR"

echo "[3/5] Fidelity Metrics..."
uv run python -m src.analysis.fidelity_metrics --run_dir "$RESULTS_DIR"

echo "[4/5] Robustness Analysis..."
uv run python -m src.analysis.robustness_analysis --run_dir "$RESULTS_DIR" --noise_levels 0.1 0.2 --repetitions 3

echo "[5/6] Quality Index..."
uv run python -m src.analysis.quality_index --run_dir "$RESULTS_DIR"

echo "[6/6] Generating Plots..."
uv run python -m src.analysis.generate_plots --run_dir "$RESULTS_DIR"

echo "=========================================================="
echo "Experiment Run $RUN_ID Completed!"
echo "Results: $RESULTS_DIR"
echo "Analysis: $RESULTS_DIR/analysis"
echo "Plots: $RESULTS_DIR/plots"
echo "Quality Index: $RESULTS_DIR/analysis/quality_index.csv"
echo "=========================================================="

# ----- Timer End -----
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "End Time: $(date)"
printf "Total Duration: %02d:%02d:%02d\n" $hours $minutes $seconds
echo "=========================================================="
