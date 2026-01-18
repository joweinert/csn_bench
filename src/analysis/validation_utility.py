# src/analysis/validation_utility.py
from __future__ import annotations

"""A2) Validation utility + validation consistency.

This analysis measures whether synthetic graphs are useful for benchmarking
community detection algorithms.

Outputs (default: <run_dir>/analysis/):

- validation_utility_metrics.csv
    Rows: (dataset, method, sample_id, algorithm) with ARI/NMI/VI/Jaccard.
- validation_consistency.csv
    Rows: (dataset, method) with Spearman rank correlation comparing algorithm
    ranking on the empirical graph to ranking on synthetic graphs.
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import community as community_louvain  # python-louvain
from infomap import Infomap
import igraph as ig

from src.data.load_data import (
    POLICY_ANALYSIS,
    load_clustering,
    load_graph_from_path,
    resolve_empirical_graph_path,
    load_synthetic_graphs,
    induce_labels_on_graph,
    simplify_for_metrics,
)


# ======================================================
# Partition similarity metrics (computed on node intersection)
# ======================================================

def _choose2(k: int) -> int:
    return k * (k - 1) // 2


def variation_of_information(true_labels: Dict[object, object], pred_labels: Dict[object, object]) -> float:
    """Variation of Information (VI). Lower is better.

    Reference: Meila (2007).
    """
    n = len(true_labels)
    if n == 0:
        return float("nan")

    ct = Counter(true_labels.values())
    cp = Counter(pred_labels.values())
    joint = Counter((true_labels[u], pred_labels[u]) for u in true_labels.keys())

    H_true = -sum((v / n) * math.log(v / n) for v in ct.values())
    H_pred = -sum((v / n) * math.log(v / n) for v in cp.values())

    I = 0.0
    for (t, p), v in joint.items():
        I += (v / n) * math.log((v * n) / (ct[t] * cp[p]))

    return float(H_true + H_pred - 2.0 * I)


def jaccard_partition_similarity(true_labels: Dict[object, object], pred_labels: Dict[object, object]) -> float:
    """Pairwise Jaccard similarity between partitions without O(n^2) enumeration."""
    nodes = list(true_labels.keys())
    if not nodes:
        return float("nan")

    ct = Counter(true_labels[u] for u in nodes)
    cp = Counter(pred_labels[u] for u in nodes)
    joint = Counter((true_labels[u], pred_labels[u]) for u in nodes)

    TP = sum(_choose2(v) for v in joint.values())
    T_true = sum(_choose2(v) for v in ct.values())
    T_pred = sum(_choose2(v) for v in cp.values())

    FP = T_pred - TP
    FN = T_true - TP
    denom = TP + FP + FN

    if denom == 0:
        return 1.0
    return float(TP / denom)


def compute_scores(true_labels: Dict[object, object], pred_labels: Dict[object, object]) -> Dict[str, float]:
    """Compute ARI/NMI/VI/Jaccard on the node intersection."""
    common = sorted(set(true_labels) & set(pred_labels))
    if not common:
        return {"ARI": np.nan, "NMI": np.nan, "VI": np.nan, "Jaccard": np.nan, "n_eval_nodes": 0.0}

    t_vals = [true_labels[u] for u in common]
    p_vals = [pred_labels[u] for u in common]

    t_map = {u: true_labels[u] for u in common}
    p_map = {u: pred_labels[u] for u in common}

    return {
        "ARI": float(adjusted_rand_score(t_vals, p_vals)),
        "NMI": float(normalized_mutual_info_score(t_vals, p_vals)),
        "VI": float(variation_of_information(t_map, p_map)),
        "Jaccard": float(jaccard_partition_similarity(t_map, p_map)),
        "n_eval_nodes": float(len(common)),
    }


# ======================================================
# Community detection algorithms (NO fallbacks)
# ======================================================

def metrics_view(G: nx.Graph) -> nx.Graph:
    """Canonical view for CD algorithms and metrics."""
    return simplify_for_metrics(G)


def _require_int_nodes(G: nx.Graph, alg_name: str) -> None:
    for n in G.nodes():
        if not isinstance(n, (int, np.integer)):
            raise TypeError(
                f"{alg_name} requires integer node IDs, found node={n!r} (type={type(n)}). "
                "Fix graph loading/canonicalization so nodes are ints."
            )


def run_louvain(G: nx.Graph, *, seed: int = 42, resolution: float = 1.0) -> Dict[object, int]:
    H = metrics_view(G)
    if H.number_of_nodes() == 0:
        return {}
    return community_louvain.best_partition(H, random_state=seed, resolution=resolution)


def run_label_propagation(G: nx.Graph, *, seed: int = 42) -> Dict[object, int]:
    H = metrics_view(G)
    if H.number_of_nodes() == 0:
        return {}

    comms = list(nx.algorithms.community.asyn_lpa_communities(H, seed=seed))
    labels: Dict[object, int] = {}
    for cid, nodes in enumerate(comms):
        for u in nodes:
            labels[u] = cid
    return labels


def run_infomap(G: nx.Graph) -> Dict[int, int]:
    H = metrics_view(G)
    _require_int_nodes(H, "Infomap")

    im = Infomap("--two-level --silent")
    for u, v in H.edges():
        im.add_link(int(u), int(v))
    im.run()

    return {node.node_id: node.module_id for node in im.tree if node.is_leaf}


def run_spinglass(G: nx.Graph, *, spins: int = 25) -> Dict[int, int]:
    H = metrics_view(G)
    _require_int_nodes(H, "Spinglass")

    nodes = list(H.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in H.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs["name"] = nodes

    clustering = g.community_spinglass(spins=spins)
    labels: Dict[int, int] = {}
    for cid, members in enumerate(clustering):
        for idx in members:
            labels[int(g.vs[idx]["name"])] = cid
    return labels



# ======================================================
# IO helpers
# ======================================================

def _list_methods(final_dataset_path: Path) -> List[str]:
    if not final_dataset_path.exists():
        raise FileNotFoundError(f"final_dataset directory not found: {final_dataset_path}")
    return sorted([p.name for p in final_dataset_path.iterdir() if p.is_dir()])


def _load_gt_labels(data_dir: str | Path, dataset: str) -> Dict[object, object]:
    p = Path(data_dir) / f"{dataset}.clustering.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Ground-truth clustering not found: {p}")
    return load_clustering(p)


def _load_empirical_graph(data_dir: str | Path, dataset: str) -> nx.Graph:
    p = resolve_empirical_graph_path(data_dir, dataset)
    if p is None:
        raise FileNotFoundError(
            f"Empirical graph not found for dataset='{dataset}' under data_dir='{data_dir}'."
        )
    return load_graph_from_path(p, policy=POLICY_ANALYSIS)


# ======================================================
# Main evaluation
# ======================================================

def evaluate_validation_utility(
    datasets: Iterable[str],
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    out_csv: str | Path | None = None,
    out_consistency_csv: str | Path | None = None,
    seed: int = 42,
    include_spinglass: bool = False,
    fail_fast: bool = True,
    min_gt_overlap_frac: float = 0.99,
    skip_gt_for_methods_contains: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run validation utility analysis and compute consistency summary."""

    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    final_dataset_path = run_dir / "final_dataset"
    methods = _list_methods(final_dataset_path)

    if out_csv is None:
        out_csv = analysis_dir / "validation_utility_metrics.csv"
    if out_consistency_csv is None:
        out_consistency_csv = analysis_dir / "validation_consistency.csv"

    algorithms = {
        "Louvain": lambda G: run_louvain(G, seed=seed),
        "LabelProp": lambda G: run_label_propagation(G, seed=seed),
        "Infomap": run_infomap,
    }
    if include_spinglass:
        algorithms["Spinglass"] = run_spinglass

    rows: List[Dict[str, object]] = []
    consistency_rows: List[Dict[str, object]] = []

    for d in datasets:
        gt_full = _load_gt_labels(data_dir, d)
        gt_nodes = set(gt_full.keys())

        # Empirical baseline per algorithm
        G_emp = metrics_view(_load_empirical_graph(data_dir, d))
        gt_emp = induce_labels_on_graph(gt_full, G_emp)
        emp_scores: Dict[str, Dict[str, float]] = {}
        for alg_name, alg_fn in algorithms.items():
            pred_emp = alg_fn(G_emp)
            emp_scores[alg_name] = compute_scores(gt_emp, pred_emp)

        for method in methods:
            method_l = method.lower()
            if any(tok in method_l for tok in skip_gt_for_methods_contains):
                rows.append({
                    "dataset": d,
                    "method": method,
                    "sample_id": None,
                    "algorithm": None,
                    "gt_valid": 0.0,
                    "gt_reason": "method_skipped_non_identity_node_semantics",
                    "node_overlap_frac_gt_wrt_graph": np.nan,
                    "ARI": np.nan,
                    "NMI": np.nan,
                    "VI": np.nan,
                    "Jaccard": np.nan,
                    "n_eval_nodes": 0.0,
                    "error": "",
                })
                continue

            graphs_dict = load_synthetic_graphs(str(final_dataset_path), method, d, policy=POLICY_ANALYSIS)
            if not graphs_dict:
                continue

            for sample_id, G_raw in graphs_dict.items():
                G = metrics_view(G_raw)

                V = set(G.nodes())
                overlap_frac = (len(V & gt_nodes) / len(V)) if V else 0.0
                gt_valid = float(overlap_frac >= min_gt_overlap_frac)
                gt_reason = "ok" if gt_valid else f"node_overlap<{min_gt_overlap_frac}"

                gt_induced = induce_labels_on_graph(gt_full, G)

                for alg_name, alg_fn in algorithms.items():
                    try:
                        pred = alg_fn(G)
                        if gt_valid:
                            scores = compute_scores(gt_induced, pred)
                        else:
                            scores = {"ARI": np.nan, "NMI": np.nan, "VI": np.nan, "Jaccard": np.nan, "n_eval_nodes": 0.0}

                        rows.append({
                            "dataset": d,
                            "method": method,
                            "sample_id": str(sample_id),
                            "algorithm": alg_name,
                            "gt_valid": gt_valid,
                            "gt_reason": gt_reason,
                            "node_overlap_frac_gt_wrt_graph": float(overlap_frac),
                            **scores,
                            "error": "",
                        })
                    except Exception as e:
                        if fail_fast:
                            raise
                        rows.append({
                            "dataset": d,
                            "method": method,
                            "sample_id": str(sample_id),
                            "algorithm": alg_name,
                            "gt_valid": gt_valid,
                            "gt_reason": gt_reason,
                            "node_overlap_frac_gt_wrt_graph": float(overlap_frac),
                            "ARI": np.nan,
                            "NMI": np.nan,
                            "VI": np.nan,
                            "Jaccard": np.nan,
                            "n_eval_nodes": 0.0,
                            "error": f"{type(e).__name__}: {e}",
                        })

        # --- consistency per method (after dataset loop) ---
        df_tmp = pd.DataFrame(rows)
        df_ds = df_tmp[(df_tmp["dataset"] == d) & (df_tmp["algorithm"].notna())].copy()
        df_ds = df_ds[(df_ds.get("error", "") == "")].copy()
        df_ds = df_ds[df_ds.get("gt_valid", 0.0) == 1.0].copy()

        for method in sorted(df_ds["method"].unique()):
            df_m = df_ds[df_ds["method"] == method].copy()
            if df_m.empty:
                continue

            mean_by_alg = df_m.groupby("algorithm").mean(numeric_only=True)
            common_algs = sorted(set(mean_by_alg.index) & set(emp_scores.keys()))
            if len(common_algs) < 2:
                continue

            def _corr(metric: str, *, invert: bool = False) -> float:
                x = np.array([emp_scores[a][metric] for a in common_algs], dtype=float)
                y = np.array([mean_by_alg.loc[a, metric] for a in common_algs], dtype=float)
                if invert:
                    x = -x
                    y = -y
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 2:
                    return float("nan")
                return float(spearmanr(x[m], y[m]).correlation)

            consistency_rows.append({
                "dataset": d,
                "method": method,
                "n_samples": int(df_m["sample_id"].nunique()),
                "algorithms": ",".join(common_algs),
                "spearman_ARI": _corr("ARI"),
                "spearman_NMI": _corr("NMI"),
                "spearman_Jaccard": _corr("Jaccard"),
                "spearman_VI": _corr("VI", invert=True),
                "delta_mean_ARI": float(np.nanmean([mean_by_alg.loc[a, "ARI"] - emp_scores[a]["ARI"] for a in common_algs])),
                "delta_mean_NMI": float(np.nanmean([mean_by_alg.loc[a, "NMI"] - emp_scores[a]["NMI"] for a in common_algs])),
                "delta_mean_Jaccard": float(np.nanmean([mean_by_alg.loc[a, "Jaccard"] - emp_scores[a]["Jaccard"] for a in common_algs])),
                "delta_mean_VI": float(np.nanmean([mean_by_alg.loc[a, "VI"] - emp_scores[a]["VI"] for a in common_algs])),
            })

    df = pd.DataFrame(rows)
    df_cons = pd.DataFrame(consistency_rows)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_consistency_csv = Path(out_consistency_csv)
    out_consistency_csv.parent.mkdir(parents=True, exist_ok=True)
    df_cons.to_csv(out_consistency_csv, index=False)

    print(f"Saved validation utility metrics to {out_csv}")
    print(f"Saved validation consistency summary to {out_consistency_csv}")

    return df, df_cons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation utility + consistency analysis.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=["karate", "polbooks", "football"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_spinglass", action="store_true")
    parser.add_argument("--no_fail_fast", action="store_true")
    parser.add_argument("--min_gt_overlap_frac", type=float, default=0.99)
    parser.add_argument("--skip_methods_contains", nargs="*", default=[])

    args = parser.parse_args()

    evaluate_validation_utility(
        datasets=args.datasets,
        run_dir=Path(args.run_dir).resolve(),
        data_dir=args.data_dir,
        seed=args.seed,
        include_spinglass=args.include_spinglass,
        fail_fast=(not args.no_fail_fast),
        min_gt_overlap_frac=args.min_gt_overlap_frac,
        skip_gt_for_methods_contains=tuple(s.lower() for s in args.skip_methods_contains),
    )
