import os
import glob
import argparse
from pathlib import Path
from typing import Optional

import networkx as nx

from src.data.load_data import _read_edgelist_tsv


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tsv_to_gml(
    tsv_path: Path,
    gml_path: Path,
    *,
    delimiter: str = "\t",
    node_cast=int,
    has_weight: str = "auto",
) -> None:
    """
    Convert TSV edge list to GML.
    - If TSV has 3rd column, it's interpreted as weight (auto mode).
    - Nodes are cast to int by default because RECCS/EC-SBM pipelines typically
      operate on relabeled integer node IDs.
    """
    G = _read_edgelist_tsv(
        str(tsv_path),
        delimiter=delimiter,
        node_cast=node_cast,
        has_weight=has_weight,
        skip_header="auto",
    )

    ensure_dir(gml_path.parent)
    nx.write_gml(G, str(gml_path))


def organize_reccs(run_dir: Path) -> None:
    """
    Source:
      results/<run_id>/RECCS/<dataset>/03_outliers/sample_<id>/syn_o_un.tsv
    Target:
      results/<run_id>/final_dataset/RECCS/<dataset>_<id>.gml
    """
    reccs_root = run_dir / "RECCS"
    final_dir = run_dir / "final_dataset" / "RECCS"
    ensure_dir(final_dir)

    source_pattern = str(reccs_root / "*" / "03_outliers" / "sample_*" / "syn_o_un.tsv")
    files = sorted(glob.glob(source_pattern))
    print(f"Found {len(files)} RECCS files in {reccs_root}.")

    for f in files:
        p = Path(f)
        sample_part = p.parent.name  # sample_0
        dataset_part = p.parent.parent.parent.name  # <dataset>

        sample_id = sample_part.split("_")[-1]
        out_name = f"{dataset_part}_{sample_id}.gml"
        out_path = final_dir / out_name

        try:
            tsv_to_gml(p, out_path, node_cast=int, has_weight="auto")
            print(f"Converted: {p} -> {out_path}")
        except Exception as e:
            print(f"Error converting RECCS {p}: {type(e).__name__}: {e}")


def organize_ecsbm(run_dir: Path) -> None:
    """
    Source:
      results/<run_id>/EC-SBM/<dataset>/sample_<id>/ecsbm+o+e/edge.tsv
    Target:
      results/<run_id>/final_dataset/EC-SBM/<dataset>_<id>.gml
    """
    ecsbm_root = run_dir / "EC-SBM"
    final_dir = run_dir / "final_dataset" / "EC-SBM"
    ensure_dir(final_dir)

    source_pattern = str(ecsbm_root / "*" / "sample_*" / "ecsbm+o+e" / "edge.tsv")
    files = sorted(glob.glob(source_pattern))
    print(f"Found {len(files)} EC-SBM files in {ecsbm_root}.")

    for f in files:
        p = Path(f)
        sample_part = p.parent.parent.name  # sample_0
        dataset_part = p.parent.parent.parent.name  # <dataset>

        sample_id = sample_part.split("_")[-1]
        out_name = f"{dataset_part}_{sample_id}.gml"
        out_path = final_dir / out_name

        try:
            tsv_to_gml(p, out_path, node_cast=int, has_weight="auto")
            print(f"Converted: {p} -> {out_path}")
        except Exception as e:
            print(f"Error converting EC-SBM {p}: {type(e).__name__}: {e}")


def main(run_dir: str) -> None:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Standardizing outputs for run: {run_dir.name}")

    print("Standardizing RECCS...")
    organize_reccs(run_dir)

    print("Standardizing EC-SBM...")
    organize_ecsbm(run_dir)

    # VAE outputs directly to final_dataset/<VAE>/... in your pipeline.
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize outputs for a specific run.")
    parser.add_argument("--run_dir", required=True, help="Path to the run results directory (e.g. results/2026...)")
    args = parser.parse_args()
    main(args.run_dir)
