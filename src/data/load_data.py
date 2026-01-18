import os
import glob
import io
import csv
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

try:
    import requests  # optional: only for toy dataset helper
except Exception:
    requests = None


# -----------------------------
# Policies
# -----------------------------

@dataclass(frozen=True)
class GraphPreprocessPolicy:
    """
    Preprocessing options applied AFTER loading.
    Keep explicit so each consumer (analysis vs VAE) opts in deliberately.
    """
    to_undirected: bool = False
    remove_self_loops: bool = False

    keep_lcc: bool = False
    collapse_multiedges: bool = False  # MultiGraph -> Graph (sum weights)

    ensure_weight: bool = False  # set missing 'weight' = 1.0

    # Make nodes 0..N-1 contiguous (needed for VAE/adjacency-based code)
    relabel_to_int: bool = False
    relabel_attribute: Optional[str] = "_orig_node"  # store old label if relabel_to_int

    # Cast node IDs (crucial to keep GT labels and predicted labels compatible)
    node_cast: Optional[Callable[[object], object]] = None  # e.g., int or str


# Canonical choice for this project: ints everywhere.
POLICY_ANALYSIS = GraphPreprocessPolicy(
    to_undirected=False,
    remove_self_loops=False,
    keep_lcc=False,
    collapse_multiedges=False,
    ensure_weight=False,
    relabel_to_int=False,
    node_cast=int,
)

# VAE needs a clean simple graph, LCC, weights, and contiguous 0..N-1 nodes.
POLICY_VAE = GraphPreprocessPolicy(
    to_undirected=True,
    remove_self_loops=True,
    keep_lcc=True,
    collapse_multiedges=True,
    ensure_weight=True,
    relabel_to_int=True,
    relabel_attribute="_orig_node",
    node_cast=int,
)


# -----------------------------
# Path helpers
# -----------------------------

def parse_dataset_and_sample(path: str) -> Tuple[str, str]:
    """Parse '{dataset}_{sample}.gml' using the last underscore as separator."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "0"


def resolve_empirical_graph_path(data_dir: Union[str, Path], dataset: str) -> Optional[str]:
    """Conservative resolution for empirical graphs."""
    data_dir = str(data_dir)
    candidates = [
        os.path.join(data_dir, f"{dataset}.tsv"),
        os.path.join(data_dir, f"{dataset}.txt"),
        os.path.join(data_dir, f"{dataset}.gml"),
        os.path.join(data_dir, f"{dataset}.graphml"),
        os.path.join(data_dir, dataset, f"{dataset}.tsv"),
        os.path.join(data_dir, dataset, "edge.tsv"),
        os.path.join(data_dir, dataset, f"{dataset}.gml"),
        os.path.join(data_dir, dataset, "graph.gml"),
        os.path.join(data_dir, dataset, "graph.graphml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def iter_final_dataset_graphs(final_dataset_dir: Union[str, Path]) -> Iterator[Tuple[str, str]]:
    """
    Yields (method, graph_path) for all .gml/.graphml files under:
      final_dataset/<METHOD>/
    """
    final_dataset_dir = str(final_dataset_dir)
    if not os.path.isdir(final_dataset_dir):
        return
    for method in sorted(os.listdir(final_dataset_dir)):
        mdir = os.path.join(final_dataset_dir, method)
        if not os.path.isdir(mdir):
            continue
        for ext in ("*.gml", "*.graphml"):
            for f in sorted(glob.glob(os.path.join(mdir, ext))):
                yield method, f


# -----------------------------
# Graph IO
# -----------------------------

def _read_edgelist_tsv(
    path: str,
    *,
    delimiter: str = "\t",
    node_cast: Optional[Callable[[object], object]] = None,
    has_weight: str = "auto",  # "auto" | "yes" | "no"
    weight_cast: Callable[[str], float] = float,
    skip_header: str = "auto",  # "auto" | "yes" | "no"
) -> nx.Graph:
    """
    Reads a 2-col or 3-col TSV edge list.
    - 2 cols: u v
    - 3 cols: u v w
    """
    G = nx.Graph()
    node_cast = node_cast or (lambda x: x)

    def _maybe_skip_header(row: List[str]) -> bool:
        if skip_header == "no":
            return False
        if skip_header == "yes":
            return True
        # auto: if node_cast is int and first two tokens aren't ints, treat as header
        if node_cast is int:
            try:
                int(row[0]); int(row[1])
                return False
            except Exception:
                return True
        return False

    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first = True
        for row in reader:
            if not row or len(row) < 2:
                continue
            if first:
                first = False
                if _maybe_skip_header(row):
                    continue

            u = node_cast(row[0])
            v = node_cast(row[1])

            w = None
            if has_weight == "yes" or (has_weight == "auto" and len(row) >= 3):
                try:
                    w = weight_cast(row[2])
                except Exception:
                    w = None

            if w is None:
                G.add_edge(u, v)
            else:
                G.add_edge(u, v, weight=w)

    return G


# Public alias (so other modules don’t import a “private” name)
read_edgelist_tsv = _read_edgelist_tsv


def load_graph_from_path(
    path: Union[str, Path],
    *,
    policy: GraphPreprocessPolicy = POLICY_ANALYSIS,
    gml_label: Optional[str] = "label",
    raise_on_error: bool = True,
) -> nx.Graph:
    """
    Load a graph from disk and apply an explicit preprocessing policy.

    Note: NetworkX read_gml defaults to label='label', renaming nodes using that
    node attribute. :contentReference[oaicite:8]{index=8}
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        if path.endswith(".gml"):
            G = nx.read_gml(path, label=gml_label)
        elif path.endswith(".graphml"):
            G = nx.read_graphml(path)
        elif path.endswith(".tsv") or path.endswith(".txt"):
            G = _read_edgelist_tsv(path, node_cast=policy.node_cast, has_weight="auto")
        else:
            raise ValueError(f"Unsupported file format: {path}")

        return preprocess_graph(G, policy=policy)
    except Exception as e:
        if raise_on_error:
            raise
        warnings.warn(f"Error loading {path}: {type(e).__name__}: {e}")
        return nx.Graph()


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_graph(G: nx.Graph, *, policy: GraphPreprocessPolicy) -> nx.Graph:
    """Apply policy transforms in a deterministic order."""
    H = G.copy()

    # 1) Node casting (fixes str/int mismatches early)
    if policy.node_cast is not None:
        mapping = {}
        changed = False
        for n in H.nodes():
            nn = policy.node_cast(n)
            mapping[n] = nn
            if nn != n:
                changed = True
        if changed:
            H = nx.relabel_nodes(H, mapping, copy=True)

    # 2) Directed -> undirected
    if policy.to_undirected and H.is_directed():
        H = H.to_undirected(as_view=False)

    # 3) Collapse multiedges (sum weights)
    if policy.collapse_multiedges and H.is_multigraph():
        S = nx.Graph()
        S.add_nodes_from(H.nodes(data=True))
        for u, v, data in H.edges(data=True):
            w = data.get("weight", 1.0)
            if S.has_edge(u, v):
                S[u][v]["weight"] = S[u][v].get("weight", 1.0) + w
            else:
                S.add_edge(u, v, weight=w)
        H = S

    # 4) Ensure weights
    if policy.ensure_weight:
        for u, v, data in H.edges(data=True):
            if "weight" not in data:
                H[u][v]["weight"] = 1.0

    # 5) Remove self-loops
    if policy.remove_self_loops:
        H.remove_edges_from(list(nx.selfloop_edges(H)))

    # 6) Keep largest connected component
    if policy.keep_lcc and H.number_of_nodes() > 0:
        if H.is_directed():
            comps = list(nx.weakly_connected_components(H))
        else:
            comps = list(nx.connected_components(H))
        if comps:
            lcc = max(comps, key=len)
            H = H.subgraph(lcc).copy()

    # 7) Relabel to 0..N-1 deterministically (prevents “silent permutation”)
    if policy.relabel_to_int:
        # convert_node_labels_to_integers supports deterministic orderings. :contentReference[oaicite:9]{index=9}
        H = nx.convert_node_labels_to_integers(
            H,
            ordering="sorted",
            label_attribute=policy.relabel_attribute,
        )
 

    return H


def simplify_for_metrics(G: nx.Graph) -> nx.Graph:
    """
    Convenience for analysis metrics:
    - undirected
    - simple graph
    - remove self-loops
    Does NOT drop components.
    """
    H = G.to_undirected(as_view=False) if G.is_directed() else G.copy()
    if H.is_multigraph():
        S = nx.Graph()
        S.add_nodes_from(H.nodes(data=True))
        S.add_edges_from(H.edges())
        H = S
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    return H


# -----------------------------
# Labels / overlap utilities
# -----------------------------

def load_clustering(
    path: Union[str, Path],
    *,
    node_cast: Callable[[object], object] = int,
) -> Dict[object, object]:
    """
    Load node clustering assignments from TSV:
      node_id<TAB>cluster_id
    Canonical: node_id as int (matches POLICY_ANALYSIS).
    """
    df = pd.read_csv(str(path), sep="\t", header=None, names=["node", "cluster"])
    df["node"] = df["node"].apply(node_cast)
    return dict(zip(df["node"], df["cluster"]))


def induce_labels_on_graph(labels: Dict[object, object], G: nx.Graph) -> Dict[object, object]:
    """Restrict a node->label mapping to nodes present in G."""
    nodes = set(G.nodes())
    return {n: labels[n] for n in labels if n in nodes}


def compute_node_overlap(G_ref: nx.Graph, G_other: nx.Graph) -> Tuple[float, int]:
    """Returns (overlap_fraction_wrt_ref, n_common_nodes)."""
    a = set(G_ref.nodes())
    b = set(G_other.nodes())
    inter = a.intersection(b)
    frac = len(inter) / max(1, len(a))
    return float(frac), int(len(inter))


# -----------------------------
# Synthetic loader (backwards compatible)
# -----------------------------

def load_synthetic_graphs(
    final_dataset_dir: str,
    method: str,
    dataset_name: str,
    *,
    policy: GraphPreprocessPolicy = POLICY_ANALYSIS,
) -> Dict[str, nx.Graph]:
    """
    Loads all graphs for a given method and dataset.
    Pattern: final_dataset/<METHOD>/{dataset_name}_*.gml
    """
    method_dir = os.path.join(final_dataset_dir, method)
    if not os.path.exists(method_dir):
        warnings.warn(f"Directory not found for {method}: {method_dir}")
        return {}

    pattern_gml = os.path.join(method_dir, f"{dataset_name}_*.gml")
    pattern_graphml = os.path.join(method_dir, f"{dataset_name}_*.graphml")
    files = sorted(glob.glob(pattern_gml)) + sorted(glob.glob(pattern_graphml))

    graphs: Dict[str, nx.Graph] = {}
    for f in files:
        _, sample_id = parse_dataset_and_sample(f)
        graphs[sample_id] = load_graph_from_path(f, policy=policy)
    return graphs


# -----------------------------
# Toy dataset helper
# -----------------------------

def load_toy_dataset(dataset_name: str, *, policy: GraphPreprocessPolicy = POLICY_VAE) -> nx.Graph:
    """
    Convenience loader for karate/polbooks/football (optional).
    """
    if dataset_name == "karate":
        return preprocess_graph(nx.karate_club_graph(), policy=policy)

    if requests is None:
        raise RuntimeError("requests not available; cannot download datasets.")

    urls = {
        "polbooks": "http://www-personal.umich.edu/~mejn/netdata/polbooks.zip",
        "football": "http://www-personal.umich.edu/~mejn/netdata/football.zip",
    }
    if dataset_name not in urls:
        raise ValueError("Dataset not supported.")

    r = requests.get(urls[dataset_name], timeout=30)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    filename = f"{dataset_name}.gml"
    gml_content = z.read(filename).decode("utf-8")

    lines = gml_content.splitlines()
    if lines and "creator" in lines[0].lower():
        gml_content = "\n".join(lines[1:])

    G = nx.parse_gml(gml_content, label="id")
    return preprocess_graph(G, policy=policy)


def save_graph_and_clustering(G: nx.Graph, dataset_name: str, data_dir: Path) -> None:
    """
    Saves graph as .gml and community structure as .clustering.tsv.
    Extracts 'value', 'club', or 'community' node attributes as cluster labels.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    gml_path = data_dir / f"{dataset_name}.gml"
    
    # Save Graph (GML)
    if not gml_path.exists():
        nx.write_gml(G, str(gml_path))
        print(f"Saved graph: {gml_path}")
    else:
        print(f"Graph exists, skipping: {gml_path}")
        
    # Extract & Save Clustering
    # different datasets use different keys for ground truth communities
    # karate: 'club' (or 'value' in some versions)
    # polbooks: 'value' (often "l", "n", "c")
    # football: 'value' (conference id)
    cluster_attr = None
    for attr in ["value", "club", "community"]:
        sample_node = next(iter(G.nodes(data=True)))[1]
        if attr in sample_node:
            cluster_attr = attr
            break
            
    if cluster_attr:
        clust_path = data_dir / f"{dataset_name}.clustering.tsv"
        if not clust_path.exists():
            with open(clust_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                # No header typically expected by pipeline, just ID<tab>Cluster
                # Sort by node ID for determinism
                for n, data in sorted(G.nodes(data=True), key=lambda x: int(x[0]) if isinstance(x[0], int) or x[0].isdigit() else x[0]):
                    cluster = data[cluster_attr]
                    writer.writerow([n, cluster])
            print(f"Saved clustering: {clust_path}")
        else:
            print(f"Clustering exists, skipping: {clust_path}")
    else:
        print(f"Warning: No known community attribute found for {dataset_name}. Skipping clustering save.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensure datasets are present.")
    parser.add_argument("--data_dir", default="data", help="Directory to save datasets")
    args = parser.parse_args()
    
    DATA_DIR = Path(args.data_dir)
    DATASETS = ["karate", "polbooks", "football"]
    
    print(f"Checking/Downloading datasets to '{DATA_DIR}'...")
    
    for ds in DATASETS:
        # Check if files exist to avoid download
        gml_exists = (DATA_DIR / f"{ds}.gml").exists()
        clust_exists = (DATA_DIR / f"{ds}.clustering.tsv").exists()
        
        if gml_exists and clust_exists:
            print(f"[{ds}] All files present.")
            continue
            
        print(f"[{ds}] Downloading...")
        try:
            G_toy = load_toy_dataset(ds, policy=POLICY_ANALYSIS)
            save_graph_and_clustering(G_toy, ds, DATA_DIR)
            
        except Exception as e:
            print(f"Error handling {ds}: {e}")
            
    print("Done.")

