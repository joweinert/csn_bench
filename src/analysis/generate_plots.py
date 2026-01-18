"""
Generate publication-quality figures for the CSN Project.

This script consolidates all plotting functionality:
1. Metric-based Analysis (Boxplots/Bars from CSVs)
2. Structural Visualizations (Degree/Clustering distributions from Graph files)

Outputs:
- <run_dir>/plots/fidelity_metrics/
- <run_dir>/plots/fidelity_visuals/
- <run_dir>/plots/utility/
- <run_dir>/plots/robustness/
- <run_dir>/plots/summary/
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Reuse existing data loaders for the "Visual Fidelity" section
from src.data.load_data import (
    POLICY_ANALYSIS,
    load_graph_from_path,
    load_synthetic_graphs,
    resolve_empirical_graph_path,
    simplify_for_metrics,
)

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Set a professional style
sns.set_theme(context="paper", style="whitegrid", font_scale=1.5) # Increased font_scale
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 0, # Hide titles via rcParams (or just don't set them)
    "axes.labelsize": 16, # Bigger labels
    "xtick.labelsize": 14, # Bigger ticks
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,  # Editable text in vector graphics
    "ps.fonttype": 42,
})

COLORS = sns.color_palette("colorblind")
METHOD_PALETTE = {}  # Will be populated dynamically to ensure consistent colors


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save figure in PNG (for preview) and PDF (for LaTeX)."""
    _ensure_dir(out_path.parent)
    # Tight layout often fixes cut-off labels
    fig.tight_layout()
    
    # Save PDF
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    # Save PNG
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def _get_method_palette(methods: List[str]) -> Dict[str, Tuple]:
    """Assign consistent colors to methods."""
    # Ensure baseline methods (like EC-SBM) or "Empirical" get specific colors if desired
    # For now, just map unique methods to the colorblind palette
    unique_methods = sorted(list(set(methods)))
    palette = {}
    for i, m in enumerate(unique_methods):
        palette[m] = COLORS[i % len(COLORS)]
    return palette


# -----------------------------------------------------------------------------
# 1. Fidelity Metrics (Boxplots)
# -----------------------------------------------------------------------------

def plot_fidelity_metrics(run_dir: Path, datasets: List[str]) -> None:
    csv_path = run_dir / "analysis" / "fidelity_metrics.csv"
    if not csv_path.exists():
        logger.warning("Fidelity metrics CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    # Normalize column names for display
    metric_map = {
        "deg_wasserstein": "Degree Dist. (Wasserstein)",
        "clust_wasserstein": "Clustering Dist. (Wasserstein)",
        "spath_wasserstein": "Shortest Path (Wasserstein)",
        "louvain_mod_abs_diff": "Modularity Diff (Abs)",
        "alg_conn_abs_diff": "Alg. Connectivity Diff (Abs)",
    }

    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)
    
    for ds in datasets:
        d_ds = df[df["dataset"] == ds].copy()
        if d_ds.empty:
            continue
            
        # Create a subplot for each metric group
        # Group 1: Distributions (Wasserstein)
        # Group 2: Scalars (Modularity, etc)
        
        cols_present = [c for c in metric_map.keys() if c in d_ds.columns]
        if not cols_present:
            continue
            
        # Assign colors
        global METHOD_PALETTE
        METHOD_PALETTE = _get_method_palette(d_ds["method"].unique())

        # Plot each metric individually for cleanliness
        for col in cols_present:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=d_ds, x="method", y=col, hue="method",
                palette=METHOD_PALETTE, dodge=False, ax=ax,
                linewidth=1.2, fliersize=3
            )
            # ax.set_title(f"{ds.title()}: {metric_map[col]}") # Removed title
            ax.set_xlabel("")
            ax.set_ylabel("Distance (Lower is Better)")
            
            # remove legend if redundant
            ax.legend([], [], frameon=False) 
            
            # Format Y axis for small numbers
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 3))
            
            _save_fig(fig, out_dir / f"{ds}_{col}")


# -----------------------------------------------------------------------------
# 2. Visual Fidelity (Distributions) - Replaces visualize.py
# -----------------------------------------------------------------------------

def _plot_ccdf(ax, data: np.ndarray, label: str, color: str, style: str = "-"):
    """Helper to plot Complementary Cumulative Distribution Function."""
    if len(data) == 0:
        return
    data_sorted = np.sort(data)
    p = 1.0 * np.arange(len(data)) / (len(data) - 1)
    # CCDF is 1 - CDF
    y = 1 - p
    ax.plot(data_sorted, y, color=color, linestyle=style, label=label, linewidth=1.5, alpha=0.8)

def plot_visual_distributions(run_dir: Path, datasets: List[str], data_dir: Path) -> None:
    final_dataset_path = run_dir / "final_dataset"
    if not final_dataset_path.exists():
        logger.warning("final_dataset not found. Skipping visual distributions.")
        return

    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)
    
    for ds in datasets:
        # Load Empirical
        emp_path = resolve_empirical_graph_path(data_dir, ds)
        if not emp_path:
            logger.warning(f"Empirical graph for {ds} not found. Skipping.")
            continue
            
        try:
            G_emp = simplify_for_metrics(load_graph_from_path(emp_path, policy=POLICY_ANALYSIS))
        except Exception as e:
            logger.warning(f"Failed to load empirical {ds}: {e}")
            continue

        deg_emp = np.array([d for _, d in G_emp.degree()])
        clust_emp = np.array(list(nx.clustering(G_emp).values()))

        # Setup Figure: 2 subplots (Degree, Clustering)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Empirical
        _plot_ccdf(axes[0], deg_emp, "Empirical", "black", "-")
        _plot_ccdf(axes[1], clust_emp, "Empirical", "black", "-")

        # Load Methods
        methods = sorted([p.name for p in final_dataset_path.iterdir() if p.is_dir()])
        palette = _get_method_palette(methods)

        for method in methods:
            graphs = load_synthetic_graphs(str(final_dataset_path), method, ds, policy=POLICY_ANALYSIS)
            if not graphs:
                continue
            
            # Aggregate stats from all samples to smooth the line
            all_degs = []
            all_clusts = []
            
            # Use max 5 samples to keep it fast
            for _, G_syn_raw in list(graphs.items())[:5]:
                G_syn = simplify_for_metrics(G_syn_raw)
                all_degs.extend([d for _, d in G_syn.degree()])
                all_clusts.extend(list(nx.clustering(G_syn).values()))
            
            _plot_ccdf(axes[0], np.array(all_degs), method, palette.get(method, "blue"), "--")
            _plot_ccdf(axes[1], np.array(all_clusts), method, palette.get(method, "blue"), "--")

        # Formatting Degree Plot
        # axes[0].set_title(f"{ds.title()} | Degree Distribution (CCDF)")
        axes[0].set_xlabel("Degree (k)")
        axes[0].set_ylabel("P(K >= k)")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        
        if ds.lower() == "karate":
            axes[0].legend()
        else:
            axes[0].get_legend().remove() if axes[0].get_legend() else None

        # Formatting Clustering Plot
        # axes[1].set_title(f"{ds.title()} | Clustering Coeff. (CCDF)")
        axes[1].set_xlabel("Local Clustering Coefficient")
        axes[1].set_ylabel("P(C >= c)")
        # axes[1].legend() # No legend on the second plot typically needed if first has it, or reuse same logic
        axes[1].get_legend().remove() if axes[1].get_legend() else None


        _save_fig(fig, out_dir / f"{ds}_distributions")


# -----------------------------------------------------------------------------
# 3. Validation Utility (Boxplots)
# -----------------------------------------------------------------------------

def plot_utility_metrics(run_dir: Path, datasets: List[str]) -> None:
    csv_path = run_dir / "analysis" / "validation_utility_metrics.csv"
    if not csv_path.exists():
        logger.warning("Utility CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    # Filter for valid runs
    df = df[(df["algorithm"].notna()) & (df["gt_valid"] == 1.0)]
    if "error" in df.columns:
        df = df[(df["error"].isna()) | (df["error"] == "")]

    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)

    metrics = ["ARI", "NMI", "Jaccard", "VI"]
    
    for ds in datasets:
        d_ds = df[df["dataset"] == ds].copy()
        if d_ds.empty:
            continue
            
        palette = _get_method_palette(d_ds["method"].unique())
        
        # Plot each metric, faceting by Algorithm (Louvain/LabelProp/etc)
        for metric in metrics:
            if metric not in d_ds.columns:
                continue
                
            algorithms = d_ds["algorithm"].unique()
            if len(algorithms) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            
            sns.boxplot(
                data=d_ds, x="algorithm", y=metric, hue="method",
                palette=palette, ax=ax, linewidth=1.2, showfliers=False
            )
            
            # ax.set_title(f"{ds.title()}: {metric} vs Ground Truth")
            ax.set_xlabel("Community Detection Algorithm")
            ax.set_ylabel(metric)
            
            if ds.lower() == "karate":
                ax.legend(title="Generator", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                 ax.legend([], [], frameon=False)
            
            _save_fig(fig, out_dir / f"{ds}_{metric}")


# -----------------------------------------------------------------------------
# 4. Consistency (Heatmap/Bar)
# -----------------------------------------------------------------------------

def plot_consistency(run_dir: Path, datasets: List[str]) -> None:
    csv_path = run_dir / "analysis" / "validation_consistency.csv"
    if not csv_path.exists():
        logger.warning("Consistency CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)

    # We want to plot Spearman Correlations for ARI ranking
    metric = "spearman_ARI"
    if metric not in df.columns:
        return

    for ds in datasets:
        d_ds = df[df["dataset"] == ds].copy()
        if d_ds.empty:
            continue
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=d_ds, x="method", y=metric,
            hue="method", palette=_get_method_palette(d_ds["method"].unique()),
            ax=ax, dodge=False
        )
        
        # ax.set_title(f"{ds.title()}: Consistency (Rank Correlation)")
        ax.set_ylabel("Spearman Rho (ARI Ranking)")
        ax.set_xlabel("")
        ax.set_ylim(-1.1, 1.1) # Correlation is -1 to 1, centering 0
        ax.axhline(0, color="black", linewidth=0.8)
        
        # Barplot doesn't auto-legend usually unless explicitly asked, but just in case
        if ax.get_legend() is not None:
             ax.get_legend().remove()

        _save_fig(fig, out_dir / f"{ds}_consistency")


# -----------------------------------------------------------------------------
# 5. Robustness (Line Plots)
# -----------------------------------------------------------------------------

def plot_robustness(run_dir: Path, datasets: List[str], alg: str = "Louvain") -> None:
    csv_path = run_dir / "analysis" / "robustness_metrics.csv"
    if not csv_path.exists():
        logger.warning("Robustness CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    # Filter
    if "error" in df.columns:
        df = df[(df["error"].isna()) | (df["error"] == "")]
    if "algorithm" in df.columns:
        df = df[df["algorithm"] == alg]
    
    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)
    
    metric = "ARI_stability"
    
    for ds in datasets:
        d_ds = df[df["dataset"] == ds].copy()
        if d_ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        
        sns.lineplot(
            data=d_ds, x="noise_frac", y=metric, hue="method", style="method",
            palette=_get_method_palette(d_ds["method"].unique()),
            markers=True, dashes=False, linewidth=2, ax=ax,
            err_style="band", errorbar=("sd", 1) # Show standard deviation
        )
        
        # ax.set_title(f"{ds.title()}: Stability under Noise ({alg})")
        ax.set_xlabel("Noise Fraction (Edges Rewired)")
        ax.set_ylabel(f"Stability ({metric.split('_')[0]})")
        
        if ds.lower() == "karate":
            ax.legend(title="Generator")
        else:
             ax.legend([], [], frameon=False)
        
        _save_fig(fig, out_dir / f"{ds}_robustness_{alg}")


# -----------------------------------------------------------------------------
# 6. Quality Index (Summary)
# -----------------------------------------------------------------------------

def plot_quality_summary(run_dir: Path) -> None:
    csv_path = run_dir / "analysis" / "quality_index.csv"
    if not csv_path.exists():
        logger.warning("Quality Index CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    out_dir = run_dir / "plots"
    _ensure_dir(out_dir)

    # Faceted bar chart: Dataset on X, Score on Y, Hue = Method
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df, x="dataset", y="Quality_Index", hue="method",
        palette=_get_method_palette(df["method"].unique()),
        ax=ax, edgecolor="black"
    )
    
    # ax.set_title("Overall Generator Quality Index")
    ax.set_ylabel("Composite Score (Higher is Better)")
    ax.set_xlabel("Dataset")
    ax.legend(title="Generator", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.05)
    
    _save_fig(fig, out_dir / "overall_quality_index")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready plots.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--data_dir", default="data", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["karate", "polbooks", "football"])
    parser.add_argument("--robustness_alg", default="Louvain")
    parser.add_argument("--skip_visuals", action="store_true", help="Skip expensive graph loading for visual plots.")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info(f"Generating plots for run: {run_dir}")

    # 1. Metric Boxplots
    logger.info("[1/6] Fidelity Metrics...")
    plot_fidelity_metrics(run_dir, args.datasets)

    # 2. Visual Distributions (Replaces visualize.py)
    if not args.skip_visuals:
        logger.info("[2/6] Visual Distributions (Graph Loading)...")
        plot_visual_distributions(run_dir, args.datasets, args.data_dir)
    else:
        logger.info("[2/6] Skipping Visual Distributions.")

    # 3. Utility Boxplots
    logger.info("[3/6] Utility Metrics...")
    plot_utility_metrics(run_dir, args.datasets)

    # 4. Consistency
    logger.info("[4/6] Consistency...")
    plot_consistency(run_dir, args.datasets)

    # 5. Robustness
    logger.info("[5/6] Robustness...")
    plot_robustness(run_dir, args.datasets, args.robustness_alg)

    # 6. Quality Index
    logger.info("[6/6] Quality Summary...")
    plot_quality_summary(run_dir)

    logger.info("Done! Plots are located in: " + str(run_dir / "plots"))


if __name__ == "__main__":
    main()