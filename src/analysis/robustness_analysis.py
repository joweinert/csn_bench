"""A3) Robustness analysis under controlled noise.

We test how stable community detection outcomes are when a graph is perturbed
by random degree-preserving edge rewiring (double-edge swaps).

For each synthetic sample:
- Compute a baseline partition on the original sample.
- For each noise level and repetition, perturb the graph, re-run the algorithm,
  and compute:
    (i) Stability vs. baseline partition (ARI/NMI/VI/Jaccard)
    (ii) (Optional) GT alignment if node identities correspond to ground truth.

Output: <run_dir>/analysis/robustness_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.analysis.validation_utility import (
    compute_scores,
    run_infomap,
    run_label_propagation,
    run_louvain,
    run_spinglass,
)
from src.data.load_data import (
    POLICY_ANALYSIS,
    induce_labels_on_graph,
    load_clustering,
    load_synthetic_graphs,
    simplify_for_metrics,
)

logger = logging.getLogger(__name__)


def metrics_view(G: nx.Graph) -> nx.Graph:
    return simplify_for_metrics(G)


def _list_methods(final_dataset_path: Path) -> List[str]:
    if not final_dataset_path.exists():
        raise FileNotFoundError(f"final_dataset directory not found: {final_dataset_path}")
    return sorted([p.name for p in final_dataset_path.iterdir() if p.is_dir()])


def _load_gt_labels(data_dir: str | Path, dataset: str) -> Dict[int, int]:
    p = Path(data_dir) / f"{dataset}.clustering.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Ground-truth clustering not found: {p}")
    labels = load_clustering(p, node_cast=int)
    return {int(k): v for k, v in labels.items()}


def perturb_graph(
    G: nx.Graph,
    noise_frac: float,
    *,
    seed: int,
    preserve_connectivity: bool,
) -> Tuple[nx.Graph, int]:
    """Degree-preserving perturbation by double-edge swaps.

    noise_frac is interpreted as a fraction of edges to *rewire*.
    Each double-edge swap rewires two edges, so nswap ~ noise_frac * m / 2.

    Returns (noisy_graph, n_swaps_attempted_or_succeeded).
    """
    if not (0.0 <= noise_frac <= 1.0):
        raise ValueError(f"noise_frac must be in [0,1], got {noise_frac}")

    H = metrics_view(G).copy()
    m = H.number_of_edges()
    n = H.number_of_nodes()

    if noise_frac == 0.0 or m < 2 or n < 4:
        return H, 0

    nswap = max(1, int(round((noise_frac * m) / 2.0)))

    if preserve_connectivity:
        if not nx.is_connected(H):
            raise nx.NetworkXError(
                "preserve_connectivity=True requires a connected graph (after simplification)."
            )
        # connected_double_edge_swap modifies in place and returns number of swaps performed
        n_success = nx.connected_double_edge_swap(H, nswap=nswap, seed=seed)
        return H, int(n_success)

    max_tries = max(100, 20 * nswap)
    H2 = nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=seed)
    return H2, int(nswap)


def _run_alg(alg_name: str, alg_fn, G: nx.Graph, *, seed: int) -> Dict[object, int]:
    if alg_name in ("Louvain", "LabelProp"):
        return alg_fn(G, seed=seed)
    return alg_fn(G)


def evaluate_robustness(
    datasets: Iterable[str],
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    noise_levels: List[float] | None = None,
    repetitions: int = 3,
    seed: int = 42,
    include_spinglass: bool = False,
    include_baseline: bool = True,
    preserve_connectivity: bool = False,
    min_gt_overlap_frac: float = 0.99,
    skip_gt_for_methods_contains: Tuple[str, ...] = (),
    out_csv: str | Path | None = None,
    fail_fast: bool = True,
) -> pd.DataFrame:
    """Run robustness analysis and write robustness_metrics.csv."""

    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]

    noise_levels = sorted(set(float(x) for x in noise_levels))
    if include_baseline and 0.0 not in noise_levels:
        noise_levels = [0.0] + noise_levels

    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    final_dataset_path = run_dir / "final_dataset"
    methods = _list_methods(final_dataset_path)

    if out_csv is None:
        out_csv = analysis_dir / "robustness_metrics.csv"

    algorithms = {
        "Louvain": run_louvain,
        "LabelProp": run_label_propagation,
        "Infomap": run_infomap,
    }
    if include_spinglass:
        algorithms["Spinglass"] = run_spinglass

    rows: List[Dict[str, object]] = []

    for d in datasets:
        gt_full = _load_gt_labels(data_dir, d)
        gt_nodes = set(gt_full.keys())

        for method in methods:
            graphs_dict = load_synthetic_graphs(str(final_dataset_path), method, d, policy=POLICY_ANALYSIS)
            if not graphs_dict:
                continue

            method_l = method.lower()
            method_skips_gt = any(tok in method_l for tok in skip_gt_for_methods_contains)

            for sample_id, G_raw in graphs_dict.items():
                base = {
                    "dataset": d,
                    "method": method,
                    "sample_id": str(sample_id),
                }

                try:
                    G_base = metrics_view(G_raw)
                    V = set(G_base.nodes())
                    overlap_frac = (len(V & gt_nodes) / len(V)) if V else 0.0

                    gt_valid = (not method_skips_gt) and (overlap_frac >= min_gt_overlap_frac) and bool(gt_full)
                    if not gt_full:
                        gt_reason = "no_gt_labels"
                    elif method_skips_gt:
                        gt_reason = "method_skipped_non_identity_node_semantics"
                    elif overlap_frac < min_gt_overlap_frac:
                        gt_reason = f"node_overlap<{min_gt_overlap_frac}"
                    else:
                        gt_reason = "ok"

                    # Baseline partitions per algorithm (for stability)
                    base_partitions = {
                        alg_name: _run_alg(alg_name, alg_fn, G_base, seed=seed)
                        for alg_name, alg_fn in algorithms.items()
                    }

                    for frac in noise_levels:
                        for rep in range(repetitions):
                            frac_key = int(round(frac * 10_000))
                            noise_seed = seed + 1_000_000 * frac_key + rep

                            G_noisy, n_swaps = perturb_graph(
                                G_base,
                                frac,
                                seed=noise_seed,
                                preserve_connectivity=preserve_connectivity,
                            )

                            gt_induced = induce_labels_on_graph(gt_full, G_noisy)

                            for alg_name, alg_fn in algorithms.items():
                                pred_noisy = _run_alg(alg_name, alg_fn, G_noisy, seed=seed)

                                # Stability: compare baseline vs noisy (always meaningful)
                                pred_base = base_partitions[alg_name]
                                stab = compute_scores(pred_base, pred_noisy)

                                # GT alignment: only if valid
                                if gt_valid:
                                    gt_scores = compute_scores(gt_induced, pred_noisy)
                                else:
                                    gt_scores = {"ARI": np.nan, "NMI": np.nan, "VI": np.nan, "Jaccard": np.nan, "n_eval_nodes": 0.0}

                                rows.append({
                                    **base,
                                    "algorithm": alg_name,

                                    "noise_frac": float(frac),
                                    "repetition": int(rep),
                                    "n_nodes": float(G_base.number_of_nodes()),
                                    "n_edges": float(G_base.number_of_edges()),
                                    "n_swaps": float(n_swaps),
                                    "preserve_connectivity": float(1.0 if preserve_connectivity else 0.0),

                                    "gt_valid": float(1.0 if gt_valid else 0.0),
                                    "gt_reason": gt_reason,
                                    "node_overlap_frac_gt_wrt_graph": float(overlap_frac),

                                    "ARI_gt": gt_scores["ARI"],
                                    "NMI_gt": gt_scores["NMI"],
                                    "VI_gt": gt_scores["VI"],
                                    "Jaccard_gt": gt_scores["Jaccard"],
                                    "n_eval_nodes_gt": gt_scores["n_eval_nodes"],

                                    "ARI_stability": stab["ARI"],
                                    "NMI_stability": stab["NMI"],
                                    "VI_stability": stab["VI"],
                                    "Jaccard_stability": stab["Jaccard"],
                                    "n_eval_nodes_stability": stab["n_eval_nodes"],

                                    "error": "",
                                })

                except Exception as e:
                    if fail_fast:
                        raise
                    rows.append({
                        **base,
                        "algorithm": None,
                        "noise_frac": np.nan,
                        "repetition": np.nan,
                        "n_nodes": np.nan,
                        "n_edges": np.nan,
                        "n_swaps": np.nan,
                        "preserve_connectivity": float(1.0 if preserve_connectivity else 0.0),
                        "gt_valid": np.nan,
                        "gt_reason": "error",
                        "node_overlap_frac_gt_wrt_graph": np.nan,
                        "ARI_gt": np.nan,
                        "NMI_gt": np.nan,
                        "VI_gt": np.nan,
                        "Jaccard_gt": np.nan,
                        "n_eval_nodes_gt": np.nan,
                        "ARI_stability": np.nan,
                        "NMI_stability": np.nan,
                        "VI_stability": np.nan,
                        "Jaccard_stability": np.nan,
                        "n_eval_nodes_stability": np.nan,
                        "error": f"{type(e).__name__}: {e}",
                    })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info("Saved robustness metrics to %s", out_csv)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Run robustness analysis.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=["karate", "polbooks", "football"])
    p.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preserve_connectivity", action="store_true")
    p.add_argument("--min_gt_overlap_frac", type=float, default=0.99)
    p.add_argument("--include_spinglass", action="store_true")
    p.add_argument("--no_fail_fast", action="store_true")
    p.add_argument("--skip_methods_contains", nargs="*", default=[])

    args = p.parse_args()

    evaluate_robustness(
        datasets=args.datasets,
        run_dir=Path(args.run_dir).resolve(),
        data_dir=args.data_dir,
        noise_levels=args.noise_levels,
        repetitions=args.repetitions,
        seed=args.seed,
        preserve_connectivity=args.preserve_connectivity,
        min_gt_overlap_frac=args.min_gt_overlap_frac,
        include_spinglass=args.include_spinglass,
        skip_gt_for_methods_contains=tuple(s.lower() for s in args.skip_methods_contains),
        fail_fast=(not args.no_fail_fast),
    )
