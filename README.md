# CSN Research: Synthetic Network Generation Pipeline

This repository contains a robust, cross-platform pipeline for generating synthetic graph datasets using **RECCS**, **EC-SBM**, and a **VAE** (Variational Autoencoder). It automates data loading, containerizes the generators, and provides easy-to-use scripts for reproducible experiments.

## Project Structure

```text
csn_bench/
├── data/                    # Preprocessed inputs (edge lists, clustering)
├── docker/                  # Docker configurations
│   ├── reccs/
│   └── ecsbm/
├── scripts/                 # Modular runner scripts
│   ├── run_reccs.{bat,sh}
│   ├── run_ecsbm.{bat,sh}
│   ├── run_vae.{bat,sh}
├── src/
│   ├── data/                # Data loading & standardization
│   ├── analysis/            # Verification & Analysis scripts (Metrics, Quality Index)
│   ├── VAE/                 # VAE Generator Code
│   └── utils/               # Logging & helpers
├── .python-version          # Python version pinning
├── pyproject.toml           # Project dependencies
├── uv.lock                  # Lock file for reproducible builds
├── run_experiments.bat      # Master entry point (Windows)
└── run_experiments.sh       # Master entry point (Linux/macOS)
```

## Setup

### Requirements
- **Docker**: For RECCS and EC-SBM.
- **uv**: For strict Python dependency management and VAE execution.

### Installation
Sync the local python environment:
```bash
uv sync
```

## Usage

### 1. Run Full Experiment Suite
The master script runs the entire pipeline for all configured datasets (Karate, Polbooks, Football) using all three generators. It creates a timestamped folder in `results/`.

**Windows:**
```bat
.\run_experiments.bat [NumSamples]
```

**Linux/macOS:**
```bash
./run_experiments.sh [NumSamples]
```

*(Default `NumSamples` is 30)*


### 2. Run Individual Generators
You can also run specific experiments using the modular scripts in `scripts/`.

**Arguments:**
- RECCS/EC-SBM: `[EdgeList] [Clustering] [OutputDir] [NumSamples]`
- VAE: `[InputGraph] [ArtifactsDir] [FinalDir] [NumSamples]`

**Argument Descriptions:**
- `EdgeList` / `InputGraph`: Path to the input graph file (e.g., `.tsv`, `.gml`).
- `Clustering`: Path to the ground truth community file (for SBM/RECCS).
- `OutputDir` / `ArtifactsDir`: Directory for raw generator outputs (logs, models, internal states).
- `FinalDir`: Directory where the final standardized GML graphs will be saved.
- `NumSamples`: The number of synthetic graphs to generate per run.

**Example (RECCS):**
```bash
./scripts/run_reccs.sh data/karate.tsv data/karate.clustering.tsv results/test_run/RECCS/karate 10
```

**Example (VAE):**
```bash
./scripts/run_vae.sh data/karate.tsv results/test_run/VAE/karate results/test_run/final_dataset/VAE 10
```

## Data & Analysis

- **Data Loading**: `uv run python -m src.data.load_data` downloads and formats datasets to `data/`.
- **Standardization**: `src/data/standardize_outputs.py` converts outputs to GML in `results/<RUN_ID>/final_dataset`.
- **Analysis Pipeline**:
    - **Fidelity**: Logic, Degree/Clustering Distribution comparison (Wasserstein, KS-Test).
    - **Utility**: ARI, NMI, Jaccard, VI against Ground Truth labels.
    - **Robustness**: Generator stability under varying noise levels.
    - **Quality Index**: A composite score (0-1) ranking generator performance.
- **Visualization**: `src/analysis/generate_plots.py` produces figures in `results/<RUN_ID>/plots`.

## Credits

This project wraps and integrates the following research codes:

- **RECCS**: [lanne2_networks](https://github.com/illinois-or-research-analytics/lanne2_networks)
- **EC-SBM**: [ec-sbm](https://github.com/illinois-or-research-analytics/ec-sbm)
- **VAE**: Custom Graph Variational Autoencoder.
