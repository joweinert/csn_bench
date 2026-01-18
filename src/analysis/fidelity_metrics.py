"""A1) Fidelity metrics: empirical vs synthetic structural similarity.

For each dataset and each synthetic sample under <run_dir>/final_dataset/<method>/,
this script compares the synthetic graph to the empirical template graph.

Output: <run_dir>/analysis/fidelity_metrics.csv

Conventions
- Graphs are simplified via simplify_for_metrics(...) to ensure comparability.
- Metrics that require connectivity are computed on the Largest Connected
  Component (LCC) of the simplified graph.
- All distances are defined so that *lower is better*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import community as community_louvain  # python-louvain
import netlsd

from src.analysis.validation_utility import run_louvain
from src.data.load_data import (
    POLICY_ANALYSIS,
    compute_node_overlap,
    load_graph_from_path,
    load_synthetic_graphs,
    resolve_empirical_graph_path,
    simplify_for_metrics,
)


def metrics_view(G: nx.Graph) -> nx.Graph:
    return simplify_for_metrics(G)


def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    H = metrics_view(G)
    if H.number_of_nodes() == 0:
        return H
    if nx.is_connected(H):
        return H
    nodes = max(nx.connected_components(H), key=len)
    return H.subgraph(nodes).copy()


def _safe_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2), bounded in [0,1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    sp = p.sum()
    sq = q.sum()
    if sp <= 0 and sq <= 0:
        return 0.0
    if sp > 0:
        p = p / sp
    if sq > 0:
        q = q / sq

    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def js_divergence_of_community_sizes(sizes_a: List[int], sizes_b: List[int]) -> float:
    """JSD between *distributions of community sizes*.

    We treat the multiset of community sizes (e.g., [10, 5, 5, 2, ...]) as data
    and compare the corresponding histograms over size values.
    """
    if not sizes_a and not sizes_b:
        return 0.0
    if not sizes_a or not sizes_b:
        return 1.0

    max_size = max(max(sizes_a), max(sizes_b))
    ha = np.bincount(sizes_a, minlength=max_size + 1)
    hb = np.bincount(sizes_b, minlength=max_size + 1)
    return _js_divergence(ha, hb)


def _community_sizes_from_partition(part: Dict[object, int]) -> List[int]:
    if not part:
        return []
    counts = Counter(part.values())
    return [int(v) for v in counts.values() if v > 0]


def _shortest_path_lengths_distribution(
    G: nx.Graph,
    *,
    rng: np.random.Generator,
    exact_max_nodes: int,
    sample_sources: int,
) -> np.ndarray:
    """Shortest path length distribution on the LCC.

    - If |V(LCC)| <= exact_max_nodes: compute all-pairs shortest path lengths.
    - Else: sample `sample_sources` BFS roots and aggregate distances.

    This is an explicit scalability switch controlled by parameters.
    """
    H = lcc_subgraph(G)
    n = H.number_of_nodes()
    if n <= 1:
        return np.array([], dtype=int)

    if n <= exact_max_nodes:
        lengths: List[int] = []
        for _, dist in nx.all_pairs_shortest_path_length(H):
            for d in dist.values():
                if d > 0:
                    lengths.append(int(d))
        return np.asarray(lengths, dtype=int)

    nodes = np.asarray(list(H.nodes()))
    k = int(min(sample_sources, len(nodes)))
    sources = rng.choice(nodes, size=k, replace=False)

    lengths = []
    for s in sources:
        dist = nx.single_source_shortest_path_length(H, s)
        for d in dist.values():
            if d > 0:
                lengths.append(int(d))
    return np.asarray(lengths, dtype=int)


def _algebraic_connectivity_lcc(G: nx.Graph) -> float:
    H = lcc_subgraph(G)
    if H.number_of_nodes() <= 1:
        return 0.0
    return float(nx.algebraic_connectivity(H))


def _netlsd_signature(G: nx.Graph, *, n_times: int = 256) -> np.ndarray:
    H = lcc_subgraph(G)
    if H.number_of_nodes() == 0:
        return np.zeros(n_times, dtype=float)
    timescales = np.logspace(-2, 2, num=n_times)
    sig = netlsd.heat(H, timescales=timescales, normalization="complete")
    return np.asarray(sig, dtype=float).reshape(-1)


@dataclass(frozen=True)
class _TemplateFeatures:
    degrees: np.ndarray
    clustering: np.ndarray
    spaths: np.ndarray
    alg_conn: float
    louvain_mod: float
    louvain_comm_sizes: List[int]
    netlsd_sig: np.ndarray
    n_nodes: int
    n_edges: int


def _load_template_features(
    data_dir: str | Path,
    dataset: str,
    *,
    seed: int,
    exact_paths_max_nodes: int,
    spath_sample_sources: int,
    netlsd_n_times: int,
) -> Tuple[nx.Graph, _TemplateFeatures]:
    p = resolve_empirical_graph_path(Path(data_dir), dataset)
    if p is None:
        raise FileNotFoundError(f"Empirical graph not found for dataset='{dataset}' under data_dir='{data_dir}'.")

    G_emp_raw = load_graph_from_path(p, policy=POLICY_ANALYSIS)
    G_emp = metrics_view(G_emp_raw)

    rng = np.random.default_rng(seed)

    deg = np.asarray([d for _, d in G_emp.degree()], dtype=float)
    clust = np.asarray(list(nx.clustering(G_emp).values()), dtype=float) if G_emp.number_of_nodes() > 0 else np.array([], dtype=float)
    spaths = _shortest_path_lengths_distribution(
        G_emp,
        rng=rng,
        exact_max_nodes=exact_paths_max_nodes,
        sample_sources=spath_sample_sources,
    )

    alg_conn = _algebraic_connectivity_lcc(G_emp)

    part = run_louvain(G_emp, seed=seed)
    louvain_mod = float(community_louvain.modularity(part, G_emp)) if G_emp.number_of_edges() > 0 else 0.0
    comm_sizes = _community_sizes_from_partition(part)

    sig = _netlsd_signature(G_emp, n_times=netlsd_n_times)

    return G_emp, _TemplateFeatures(
        degrees=deg,
        clustering=clust,
        spaths=spaths,
        alg_conn=alg_conn,
        louvain_mod=louvain_mod,
        louvain_comm_sizes=comm_sizes,
        netlsd_sig=sig,
        n_nodes=int(G_emp.number_of_nodes()),
        n_edges=int(G_emp.number_of_edges()),
    )


def compute_fidelity_metrics(
    datasets: Iterable[str],
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    out_csv: str | Path | None = None,
    seed: int = 42,
    exact_paths_max_nodes: int = 2500,
    spath_sample_sources: int = 64,
    netlsd_n_times: int = 256,
    fail_fast: bool = True,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    final_dataset_path = run_dir / "final_dataset"
    if not final_dataset_path.exists():
        raise FileNotFoundError(f"final_dataset directory not found: {final_dataset_path}")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if out_csv is None:
        out_csv = analysis_dir / "fidelity_metrics.csv"

    methods = sorted([p.name for p in final_dataset_path.iterdir() if p.is_dir()])

    rows: List[Dict[str, object]] = []

    for d in datasets:
        G_emp, tmpl = _load_template_features(
            data_dir,
            d,
            seed=seed,
            exact_paths_max_nodes=exact_paths_max_nodes,
            spath_sample_sources=spath_sample_sources,
            netlsd_n_times=netlsd_n_times,
        )

        for method in methods:
            graphs = load_synthetic_graphs(str(final_dataset_path), method, d, policy=POLICY_ANALYSIS)
            if not graphs:
                continue

            for sample_id, G_raw in graphs.items():
                base = {
                    "dataset": d,
                    "method": method,
                    "sample_id": str(sample_id),
                }

                try:
                    G = metrics_view(G_raw)

                    # Diagnostics
                    overlap_frac, overlap_common = compute_node_overlap(G_emp, G)

                    rng = np.random.default_rng(seed + (abs(hash((d, method, str(sample_id)))) % 10_000_000))

                    deg = np.asarray([dd for _, dd in G.degree()], dtype=float)
                    clust = np.asarray(list(nx.clustering(G).values()), dtype=float) if G.number_of_nodes() > 0 else np.array([], dtype=float)
                    spaths = _shortest_path_lengths_distribution(
                        G,
                        rng=rng,
                        exact_max_nodes=exact_paths_max_nodes,
                        sample_sources=spath_sample_sources,
                    )

                    alg_conn = _algebraic_connectivity_lcc(G)

                    part = run_louvain(G, seed=seed)
                    louvain_mod = float(community_louvain.modularity(part, G)) if G.number_of_edges() > 0 else 0.0
                    comm_sizes = _community_sizes_from_partition(part)

                    sig = _netlsd_signature(G, n_times=netlsd_n_times)

                    rows.append({
                        **base,
                        "n_nodes": float(G.number_of_nodes()),
                        "n_edges": float(G.number_of_edges()),
                        "node_overlap_ref_frac": float(overlap_frac),
                        "node_overlap_common": float(overlap_common),

                        "deg_wasserstein": _safe_wasserstein(tmpl.degrees, deg),
                        "clust_wasserstein": _safe_wasserstein(tmpl.clustering, clust),
                        "spath_wasserstein": _safe_wasserstein(tmpl.spaths, spaths),
                        "alg_conn_abs_diff": float(abs(tmpl.alg_conn - alg_conn)),
                        "louvain_mod_abs_diff": float(abs(tmpl.louvain_mod - louvain_mod)),
                        "louvain_comm_size_js": float(js_divergence_of_community_sizes(tmpl.louvain_comm_sizes, comm_sizes)),
                        "netlsd_l2": float(np.linalg.norm(tmpl.netlsd_sig - sig)),

                        "status": "ok",
                        "error": "",
                    })

                except Exception as e:
                    if fail_fast:
                        raise
                    rows.append({
                        **base,
                        "n_nodes": float("nan"),
                        "n_edges": float("nan"),
                        "node_overlap_ref_frac": float("nan"),
                        "node_overlap_common": float("nan"),
                        "deg_wasserstein": float("nan"),
                        "clust_wasserstein": float("nan"),
                        "spath_wasserstein": float("nan"),
                        "alg_conn_abs_diff": float("nan"),
                        "louvain_mod_abs_diff": float("nan"),
                        "louvain_comm_size_js": float("nan"),
                        "netlsd_l2": float("nan"),
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved fidelity metrics to {out_csv}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute fidelity metrics for a run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=["karate", "polbooks", "football"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exact_paths_max_nodes", type=int, default=2500)
    p.add_argument("--spath_sample_sources", type=int, default=64)
    p.add_argument("--netlsd_n_times", type=int, default=256)
    p.add_argument("--no_fail_fast", action="store_true")

    args = p.parse_args()

    compute_fidelity_metrics(
        datasets=args.datasets,
        run_dir=Path(args.run_dir).resolve(),
        data_dir=args.data_dir,
        seed=args.seed,
        exact_paths_max_nodes=args.exact_paths_max_nodes,
        spath_sample_sources=args.spath_sample_sources,
        netlsd_n_times=args.netlsd_n_times,
        fail_fast=(not args.no_fail_fast),
    )
