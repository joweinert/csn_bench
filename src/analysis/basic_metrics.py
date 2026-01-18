"""A1) Basic structural metrics for each sampled synthetic graph.

This is the "baseline" descriptive layer used by the rest of the analysis.

What it does
- Iterates over every graph file in <run_dir>/final_dataset/<METHOD>/
- Loads graphs using POLICY_ANALYSIS
- Computes structural metrics on a simplified, undirected, simple graph
  via simplify_for_metrics(...)

Output
- <run_dir>/analysis/basic_metrics.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from src.data.load_data import (
    POLICY_ANALYSIS,
    compute_node_overlap,
    iter_final_dataset_graphs,
    load_graph_from_path,
    parse_dataset_and_sample,
    resolve_empirical_graph_path,
    simplify_for_metrics,
)


def _finite_float(x) -> float:
    """Cast to float; return NaN if not finite."""
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def calculate_basic_metrics(
    G: nx.Graph,
    *,
    include_paths: bool = True,
    max_nodes_for_paths: int = 5000,
) -> Dict[str, float]:
    """Compute basic structural metrics for a graph.

    Metrics are computed on simplify_for_metrics(G):
    - undirected
    - simple (no parallel edges)
    - no self-loops

    Optional path metrics (avg shortest path length, diameter) are computed on
    the largest connected component to avoid disconnected-graph errors.
    """
    # Raw diagnostics (before simplification)
    n_raw = G.number_of_nodes()
    m_raw = G.number_of_edges()
    is_directed_raw = int(G.is_directed())
    is_multigraph_raw = int(G.is_multigraph())
    n_selfloops_raw = int(nx.number_of_selfloops(G))

    H = simplify_for_metrics(G)
    n = H.number_of_nodes()
    m = H.number_of_edges()

    # Degree stats
    if n > 0:
        deg = np.fromiter((d for _, d in H.degree()), dtype=float)
        deg_mean = float(deg.mean()) if deg.size else 0.0
        deg_std = float(deg.std(ddof=0)) if deg.size else 0.0
        deg_min = float(deg.min()) if deg.size else 0.0
        deg_max = float(deg.max()) if deg.size else 0.0
    else:
        deg_mean = deg_std = deg_min = deg_max = 0.0

    density = _finite_float(nx.density(H))
    transitivity = _finite_float(nx.transitivity(H))
    avg_clustering = _finite_float(nx.average_clustering(H)) if n >= 2 else 0.0

    # Connected components + LCC fraction
    if n == 0:
        n_components = 0
        lcc_size = 0
        lcc_frac = 0.0
    else:
        n_components = int(nx.number_connected_components(H))
        comps = list(nx.connected_components(H))
        lcc_size = int(max((len(c) for c in comps), default=0))
        lcc_frac = float(lcc_size / n) if n else 0.0

    # Degree assortativity
    assortativity = _finite_float(nx.degree_assortativity_coefficient(H))

    # Optional path metrics on LCC
    asp_lcc = float("nan")
    diameter_lcc = float("nan")
    if include_paths and n > 1 and lcc_size > 1 and lcc_size <= max_nodes_for_paths:
        lcc_nodes = max(nx.connected_components(H), key=len)
        Hlcc = H.subgraph(lcc_nodes).copy()
        asp_lcc = _finite_float(nx.average_shortest_path_length(Hlcc))
        diameter_lcc = _finite_float(nx.diameter(Hlcc))

    return {
        "Nodes": float(n),
        "Edges": float(m),
        "Density": density,
        "Avg Degree": float(2.0 * m / n) if n > 0 else 0.0,
        "Degree Mean": deg_mean,
        "Degree Std": deg_std,
        "Degree Min": deg_min,
        "Degree Max": deg_max,
        "Transitivity": transitivity,
        "Avg Clustering": avg_clustering,
        "Connected Components": float(n_components),
        "LCC Size": float(lcc_size),
        "LCC Fraction": float(lcc_frac),
        "Assortativity": assortativity,
        "Avg Shortest Path (LCC)": asp_lcc,
        "Diameter (LCC)": diameter_lcc,
        # Raw diagnostics
        "_raw_nodes": float(n_raw),
        "_raw_edges": float(m_raw),
        "_raw_is_directed": float(is_directed_raw),
        "_raw_is_multigraph": float(is_multigraph_raw),
        "_raw_selfloops": float(n_selfloops_raw),
    }


def analyze_run(
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    out_file: Optional[str | Path] = None,
    include_paths: bool = True,
    max_nodes_for_paths: int = 5000,
) -> pd.DataFrame:
    """Compute metrics for every graph under <run_dir>/final_dataset."""
    run_dir = Path(run_dir)
    data_dir = Path(data_dir)

    final_dataset_path = run_dir / "final_dataset"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if out_file is None:
        out_file = analysis_dir / "basic_metrics.csv"
    out_file = Path(out_file)

    if not final_dataset_path.exists():
        raise FileNotFoundError(f"final_dataset directory not found: {final_dataset_path}")

    template_cache: Dict[str, nx.Graph] = {}
    rows = []

    for method, graph_path in iter_final_dataset_graphs(final_dataset_path):
        dataset, sample_id = parse_dataset_and_sample(graph_path)

        base_row = {
            "dataset": dataset,
            "method": method,
            "sample_id": sample_id,
            "graph_path": str(graph_path),
        }

        try:
            G = load_graph_from_path(graph_path, policy=POLICY_ANALYSIS)
            metrics = calculate_basic_metrics(G, include_paths=include_paths, max_nodes_for_paths=max_nodes_for_paths)

            # Diagnostic: node overlap vs empirical template (if resolvable)
            node_overlap = float("nan")
            n_common = 0
            template_path = resolve_empirical_graph_path(data_dir, dataset)
            if template_path is not None:
                if dataset not in template_cache:
                    template_cache[dataset] = load_graph_from_path(template_path, policy=POLICY_ANALYSIS)
                G_ref = template_cache[dataset]
                node_overlap, n_common = compute_node_overlap(G_ref, G)

            rows.append({
                **base_row,
                **metrics,
                "node_overlap_ref_frac": float(node_overlap),
                "node_overlap_common": int(n_common),
                "status": "ok",
                "error": "",
            })

        except Exception as e:
            rows.append({
                **base_row,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })

    df = pd.DataFrame(rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved basic metrics to {out_file}")

    print("\nStatus counts:")
    print(df["status"].value_counts(dropna=False))

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calculate basic metrics for all synthetic graphs in a run.")
    p.add_argument("--run_dir", required=True, help="Path to run directory")
    p.add_argument("--data_dir", default="data", help="Path to empirical data directory (for optional node overlap)")
    p.add_argument("--output_file", default=None, help="Path to output CSV (default: <run_dir>/analysis/basic_metrics.csv)")
    p.add_argument("--no_paths", action="store_true", help="Skip expensive path metrics (avg shortest path, diameter)")
    p.add_argument("--max_nodes_for_paths", type=int, default=5000, help="Skip path metrics if LCC larger than this")

    args = p.parse_args()

    analyze_run(
        args.run_dir,
        data_dir=args.data_dir,
        out_file=args.output_file,
        include_paths=not args.no_paths,
        max_nodes_for_paths=args.max_nodes_for_paths,
    )
