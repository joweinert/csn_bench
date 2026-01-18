"""A4) Composite quality index.

Combines A1 (fidelity), A2 (validation utility + consistency), and A3
(robustness) into a single score per (dataset, method).

All sub-scores are normalized *within each dataset* to [0, 1] so that datasets
with different scales remain comparable. The final index is a weighted sum.

Inputs (expected under <run_dir>/analysis/)
- fidelity_metrics.csv
- validation_utility_metrics.csv
- validation_consistency.csv
- robustness_metrics.csv

Output
- <run_dir>/analysis/quality_index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _minmax(series: pd.Series, *, lower_is_better: bool) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)

    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if s_max == s_min:
        out = pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    else:
        out = (s - s_min) / (s_max - s_min)
        out = out.fillna(0.0)

    return (1.0 - out) if lower_is_better else out


def _norm_within_dataset(df: pd.DataFrame, col: str, *, lower_is_better: bool) -> pd.Series:
    return df.groupby("dataset", group_keys=False)[col].apply(lambda s: _minmax(s, lower_is_better=lower_is_better))


def _require(path: Path, weight: float, name: str) -> None:
    if weight <= 0:
        return
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name} file (weight>0): {path}")


def compute_quality_index(
    run_dir: str | Path,
    *,
    w_fidelity: float = 0.35,
    w_utility: float = 0.35,
    w_consistency: float = 0.10,
    w_robustness: float = 0.20,
    strict_filter_errors: bool = True,
    out_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compute composite quality index.

    Weights are normalized internally (only non-negative weights allowed).

    Fidelity: lower distances are better.
    Utility: higher ARI/NMI/Jaccard and lower VI are better.
    Consistency: higher Spearman correlations are better.
    Robustness: uses stability metrics by default (higher ARI/NMI/Jaccard, lower VI).
    """

    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"

    fidelity_csv = analysis_dir / "fidelity_metrics.csv"
    utility_csv = analysis_dir / "validation_utility_metrics.csv"
    consistency_csv = analysis_dir / "validation_consistency.csv"
    robustness_csv = analysis_dir / "robustness_metrics.csv"

    _require(fidelity_csv, w_fidelity, "fidelity")
    _require(utility_csv, w_utility, "validation utility")
    _require(consistency_csv, w_consistency, "validation consistency")
    _require(robustness_csv, w_robustness, "robustness")

    # -------------------------
    # Fidelity
    # -------------------------
    if w_fidelity > 0:
        df_fid = pd.read_csv(fidelity_csv)
        if "status" in df_fid.columns:
            df_fid = df_fid[df_fid["status"] == "ok"].copy()
        agg_fid = df_fid.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True)

        # Prefer these columns if present
        fid_cols = [
            "deg_wasserstein",
            "clust_wasserstein",
            "spath_wasserstein",
            "alg_conn_abs_diff",
            "louvain_mod_abs_diff",
            "louvain_comm_size_js",
            "netlsd_l2",
        ]
        fid_cols = [c for c in fid_cols if c in agg_fid.columns]
        if not fid_cols:
            agg_fid["Score_Fidelity"] = 0.0
        else:
            tmp = pd.DataFrame({c: _norm_within_dataset(agg_fid, c, lower_is_better=True) for c in fid_cols})
            agg_fid["Score_Fidelity"] = tmp.mean(axis=1)

        base = agg_fid[["dataset", "method", "Score_Fidelity"]].copy()
    else:
        base = pd.DataFrame(columns=["dataset", "method", "Score_Fidelity"])

    # -------------------------
    # Utility
    # -------------------------
    if w_utility > 0:
        df_util = pd.read_csv(utility_csv)
        # keep only rows that correspond to actual algorithm runs
        df_util = df_util[df_util["algorithm"].notna()].copy()
        if strict_filter_errors and "error" in df_util.columns:
            df_util = df_util[(df_util["error"].isna()) | (df_util["error"] == "")].copy()
        if "gt_valid" in df_util.columns:
            df_util = df_util[df_util["gt_valid"] == 1.0].copy()

        agg_util = df_util.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True)

        hi = [c for c in ["ARI", "NMI", "Jaccard"] if c in agg_util.columns]
        lo = [c for c in ["VI"] if c in agg_util.columns]

        parts: List[pd.Series] = []
        for c in hi:
            parts.append(_norm_within_dataset(agg_util, c, lower_is_better=False))
        for c in lo:
            parts.append(_norm_within_dataset(agg_util, c, lower_is_better=True))

        agg_util["Score_Utility"] = pd.concat(parts, axis=1).mean(axis=1) if parts else 0.0

        base = base.merge(agg_util[["dataset", "method", "Score_Utility"]], on=["dataset", "method"], how="outer")
    else:
        base["Score_Utility"] = 0.0

    base["Score_Fidelity"] = base.get("Score_Fidelity", 0.0)
    base["Score_Utility"] = base.get("Score_Utility", 0.0)

    # -------------------------
    # Consistency
    # -------------------------
    if w_consistency > 0:
        df_cons = pd.read_csv(consistency_csv)
        # Aggregate correlations (avoid overfitting to a single metric)
        cols = [c for c in ["spearman_ARI", "spearman_NMI", "spearman_Jaccard", "spearman_VI"] if c in df_cons.columns]
        if not cols:
            df_cons["Score_Consistency"] = 0.0
        else:
            # Spearman correlations are in [-1, 1]; map to [0, 1] by (rho+1)/2 before minmax.
            tmp = df_cons.copy()
            for c in cols:
                tmp[c] = (tmp[c].astype(float) + 1.0) / 2.0
            tmp["cons_raw"] = tmp[cols].mean(axis=1)
            # treat as higher-is-better already
            tmp["Score_Consistency"] = _norm_within_dataset(tmp, "cons_raw", lower_is_better=False)

            df_cons = tmp[["dataset", "method", "Score_Consistency"]].copy()

        base = base.merge(df_cons[["dataset", "method", "Score_Consistency"]], on=["dataset", "method"], how="left")
    else:
        base["Score_Consistency"] = 0.0

    base["Score_Consistency"] = base["Score_Consistency"].fillna(0.0)

    # -------------------------
    # Robustness (stability)
    # -------------------------
    if w_robustness > 0:
        df_rob = pd.read_csv(robustness_csv)
        if strict_filter_errors and "error" in df_rob.columns:
            df_rob = df_rob[(df_rob["error"].isna()) | (df_rob["error"] == "")].copy()

        # focus on degradation (exclude baseline)
        if "noise_frac" in df_rob.columns:
            df_rob = df_rob[df_rob["noise_frac"] > 0.0].copy()

        agg_rob = df_rob.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True)

        hi = [c for c in ["ARI_stability", "NMI_stability", "Jaccard_stability"] if c in agg_rob.columns]
        lo = [c for c in ["VI_stability"] if c in agg_rob.columns]

        parts: List[pd.Series] = []
        for c in hi:
            parts.append(_norm_within_dataset(agg_rob, c, lower_is_better=False))
        for c in lo:
            parts.append(_norm_within_dataset(agg_rob, c, lower_is_better=True))

        agg_rob["Score_Robustness"] = pd.concat(parts, axis=1).mean(axis=1) if parts else 0.0

        base = base.merge(agg_rob[["dataset", "method", "Score_Robustness"]], on=["dataset", "method"], how="left")
    else:
        base["Score_Robustness"] = 0.0

    base["Score_Robustness"] = base["Score_Robustness"].fillna(0.0)

    # -------------------------
    # Final index
    # -------------------------
    w = np.array([w_fidelity, w_utility, w_consistency, w_robustness], dtype=float)
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if w.sum() <= 0:
        raise ValueError("At least one weight must be > 0.")
    w = w / w.sum()

    base["Quality_Index"] = (
        w[0] * base["Score_Fidelity"]
        + w[1] * base["Score_Utility"]
        + w[2] * base["Score_Consistency"]
        + w[3] * base["Score_Robustness"]
    )

    base = base.sort_values(["dataset", "Quality_Index"], ascending=[True, False]).reset_index(drop=True)

    if out_csv is None:
        out_csv = analysis_dir / "quality_index.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_csv, index=False)
    print(f"Saved quality index to {out_csv}")

    return base


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute composite quality index from analysis outputs.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_csv", default=None)

    p.add_argument("--w_fidelity", type=float, default=0.35)
    p.add_argument("--w_utility", type=float, default=0.35)
    p.add_argument("--w_consistency", type=float, default=0.10)
    p.add_argument("--w_robustness", type=float, default=0.20)

    p.add_argument("--no_strict_filter_errors", action="store_true")

    args = p.parse_args()

    compute_quality_index(
        args.run_dir,
        w_fidelity=args.w_fidelity,
        w_utility=args.w_utility,
        w_consistency=args.w_consistency,
        w_robustness=args.w_robustness,
        strict_filter_errors=(not args.no_strict_filter_errors),
        out_csv=args.out_csv,
    )
