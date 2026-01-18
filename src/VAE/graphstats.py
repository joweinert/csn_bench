from dataclasses import dataclass
import torch
import networkx as nx
import numpy as np
import community as community_louvain


@dataclass
class GraphStats:
    num_nodes: int
    num_edges: int
    density: float
    q_orig: float
    pos_weight: torch.Tensor
    edge_index_lookup: dict
    all_possible_edges: torch.Tensor
    edge_mask: torch.Tensor
    target_adj_binary: torch.Tensor
    target_weights: torch.Tensor
    features: torch.Tensor
    true_degrees: torch.Tensor

    @classmethod
    def from_networkx(cls, G):
        num_nodes = G.number_of_nodes()
        num_edges = G.size()
        density = nx.density(G)

        # Feature Extraction
        deg = list(nx.degree_centrality(G).values())
        try:
            eig = list(nx.eigenvector_centrality(G, max_iter=500).values())
        except:
            eig = [0.0] * num_nodes
        clust = list(nx.clustering(G).values())
        bet = list(nx.betweenness_centrality(G).values())

        identity_features = np.eye(num_nodes)

        features_np = np.column_stack([deg, eig, clust, bet, identity_features])
        x = torch.tensor(features_np, dtype=torch.float)

        mean = x.mean(dim=0)
        std = x.std(dim=0) + 1e-6
        x_norm = (x - mean) / std

        # Targets + Lookups
        partition = community_louvain.best_partition(G)
        q_orig = community_louvain.modularity(partition, G)

        adj_full = torch.ones((num_nodes, num_nodes))
        adj_full.fill_diagonal_(0)  # No self-loops
        all_possible_edges = adj_full.nonzero().t()

        lookup = {
            i: (u.item(), v.item()) for i, (u, v) in enumerate(all_possible_edges.t())
        }

        target_adj = torch.zeros((num_nodes, num_nodes))
        target_weights = torch.zeros((num_nodes, num_nodes))

        for u, v, d in G.edges(data=True):
            target_adj[u, v] = 1.0
            target_adj[v, u] = 1.0
            w = d.get("weight", 1.0)
            target_weights[u, v] = w
            target_weights[v, u] = w

        target_adj_flat = target_adj[all_possible_edges[0], all_possible_edges[1]]
        target_weights_flat = target_weights[
            all_possible_edges[0], all_possible_edges[1]
        ]
        edge_mask = target_adj_flat > 0

        # Calculates True Degrees Tensor
        # G.degree() returns (node, degree) tuples
        d_dict = dict(G.degree())
        # order by node index 0..N-1
        degrees_list = [d_dict[i] for i in range(num_nodes)]
        true_degrees = torch.tensor(degrees_list, dtype=torch.float)

        num_neg = (num_nodes**2) - num_edges
        pos_w = torch.tensor([num_neg / max(num_edges, 1)])

        return cls(
            num_nodes,
            num_edges,
            density,
            q_orig,
            pos_w,
            lookup,
            all_possible_edges,
            edge_mask,
            target_adj_flat,
            target_weights_flat,
            x_norm,
            true_degrees,
        )
