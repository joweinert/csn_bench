import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import copy
from scipy.stats import ks_2samp
import community as community_louvain
from torch_geometric.nn import VGAE

from src.VAE.encoder import GCNEncoder
from src.VAE.decoder import DualHeadDecoder


def train_vae_model(stats, params, max_epochs=1000):
    beta, gamma, hidden_dim, dropout = params
    latent_dim = 16

    x = stats.features
    feature_dim = x.shape[1]
    edge_index = stats.all_possible_edges[:, stats.edge_mask]

    encoder = GCNEncoder(feature_dim, int(hidden_dim), latent_dim, dropout)
    decoder = DualHeadDecoder(latent_dim)
    model = VGAE(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    true_degrees = stats.true_degrees

    # early stopping variables
    best_loss = float("inf")
    patience = 50  # how many epochs to wait for improvement
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        adj_logits, pred_weights = model.decoder(z, stats.all_possible_edges)

        # Losses
        recon_loss = F.binary_cross_entropy_with_logits(
            adj_logits, stats.target_adj_binary, pos_weight=stats.pos_weight
        )
        weight_loss = F.mse_loss(
            pred_weights[stats.edge_mask], stats.target_weights[stats.edge_mask]
        )
        kl_loss = model.kl_loss()

        # structural losses
        adj_probs = torch.sigmoid(adj_logits)
        pred_degrees = torch.zeros(stats.num_nodes)
        pred_degrees.scatter_add_(0, stats.all_possible_edges[0], adj_probs)
        degree_loss = F.mse_loss(pred_degrees, true_degrees)

        loss = (
            recon_loss
            + (beta * kl_loss)
            + (0.5 * weight_loss)
            + ((gamma * 0.1) * degree_loss)
        )

        loss.backward()
        optimizer.step()

        # early stopping logic
        current_loss = loss.item()

        # best model so far?
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # restores the best model found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def construct_synthetic_graph(
    adj_logits, predicted_weights, stats, enforce_edge_count=True
):
    """
    Universal Constructor: Uses 'Fixed Edge Count' sampling.
    Correctly handles UNDIRECTED sampling by symmetrizing probabilities
    and sampling from unique pairs (u < v).
    """
    # temperature sharpening (standardized)
    temperature = 0.5
    probs = torch.sigmoid(adj_logits / temperature).detach().cpu().numpy()
    weights = predicted_weights.detach().cpu().numpy()

    num_nodes = stats.num_nodes
    
    # 1. Map 1D probs/weights back to (N, N) matrix
    prob_mat = np.zeros((num_nodes, num_nodes))
    weight_mat = np.zeros((num_nodes, num_nodes))
    
    # stats.all_possible_edges is usually all off-diagonal pairs
    # we assume they correspond 1-to-1 with probs
    src, dst = stats.all_possible_edges.cpu().numpy()
    prob_mat[src, dst] = probs
    weight_mat[src, dst] = weights
    
    # 2. Symmetrize (Average direction predictions)
    prob_mat_sym = (prob_mat + prob_mat.T) / 2.0
    weight_mat_sym = (weight_mat + weight_mat.T) / 2.0
    
    # 3. Extract Upper Triangle (Unique Undirected Candidates)
    triu_idx = np.triu_indices(num_nodes, k=1)
    triu_probs = prob_mat_sym[triu_idx]
    triu_weights = weight_mat_sym[triu_idx]
    
    # 4. Sample
    if enforce_edge_count:
        target_num_edges = int(stats.num_edges)  # This is the undirected count
        
        # normalize
        p_sum = triu_probs.sum()
        if p_sum == 0:
            norm_probs = np.ones_like(triu_probs) / len(triu_probs)
        else:
            norm_probs = triu_probs / p_sum
            
        # Sample unique pairs
        # If target > candidates, we can't sample without replacement (shouldn't happen for sparse graphs)
        n_candidates = len(triu_probs)
        n_sample = min(target_num_edges, n_candidates)
        
        sampled_indices = np.random.choice(
            n_candidates, size=n_sample, replace=False, p=norm_probs
        )
    else:
        # standard bernoulli on upper triangle
        rand_vals = np.random.rand(len(triu_probs))
        sampled_indices = np.where(rand_vals < triu_probs)[0]

    # 5. Build Graph
    G_syn = nx.Graph()
    G_syn.add_nodes_from(range(num_nodes))
    
    u_list = triu_idx[0][sampled_indices]
    v_list = triu_idx[1][sampled_indices]
    w_list = triu_weights[sampled_indices]
    
    for u, v, w in zip(u_list, v_list, w_list):
        G_syn.add_edge(u, v, weight=float(w))

    # connectivity repair (minimal interference)
    if not nx.is_connected(G_syn):
        components = list(nx.connected_components(G_syn))
        components.sort(key=len, reverse=True)
        main_comp = set(components[0])

        # Mask out already sampled indices to find bridges
        unsampled_mask = np.ones(len(triu_probs), dtype=bool)
        unsampled_mask[sampled_indices] = False
        unsampled_indices = np.where(unsampled_mask)[0]
        
        # Candidates for bridging
        candidate_probs = triu_probs[unsampled_indices]
        # Top 500 strongest unused edges
        top_k = min(500, len(candidate_probs))
        if top_k > 0:
            best_candidates_rel_idx = np.argsort(candidate_probs)[-top_k:]
            likely_bridge_indices = unsampled_indices[best_candidates_rel_idx]
        else:
            likely_bridge_indices = []

        for other_comp in components[1:]:
            other_comp = set(other_comp)
            best_bridge_edge = None
            best_bridge_prob = -1.0
            
            # search in top likely edges
            for idx in likely_bridge_indices:
                u, v = triu_idx[0][idx], triu_idx[1][idx]
                
                # Check if it connects main_comp <-> other_comp
                is_crossing = (u in main_comp and v in other_comp) or \
                              (v in main_comp and u in other_comp)
                
                if is_crossing:
                    p = triu_probs[idx]
                    if p > best_bridge_prob:
                        best_bridge_prob = p
                        best_bridge_edge = (u, v, float(triu_weights[idx]))
            
            # fallback if no likely edge found (search all? or just force one)
            # For speed, if we miss it here, we might just skip or do a naive link
            # But let's assume top 500 covers it often. 
            
            if best_bridge_edge:
                u, v, w = best_bridge_edge
                G_syn.add_edge(u, v, weight=w)
                main_comp.update(other_comp)
    
    return G_syn


def calculate_degree_ks_test(G_syn, G_orig):
    deg_syn = [d for n, d in G_syn.degree()]
    deg_orig = [d for n, d in G_orig.degree()]
    statistic, _ = ks_2samp(deg_syn, deg_orig)
    return statistic


def calculate_clustering_ks_test(G_syn, G_orig):
    clust_syn = list(nx.clustering(G_syn).values())
    clust_orig = list(nx.clustering(G_orig).values())
    stat, _ = ks_2samp(clust_syn, clust_orig)
    return stat


# self-contained objective
def objective(params, stats, G_orig):
    """
    Accepts param list from gp_minimize,
    injects fixed constants, trains, and evaluates.
    """
    gamma, hidden_dim = params

    beta = 1e-4  # fixed small regularization
    dropout = 0.0  # fixed (Dropout hurts generation fidelity)

    full_params = [beta, gamma, hidden_dim, dropout]

    # train
    model = train_vae_model(stats, full_params, max_epochs=1000)
    model.eval()

    # evaluate (average over 5 realizations)
    realization_scores = []
    x = stats.features
    edge_index = stats.all_possible_edges[:, stats.edge_mask]

    with torch.no_grad():
        z_mu, z_std = model.encoder(x, edge_index)

        for _ in range(5):
            z = model.reparametrize(z_mu, z_std)
            adj_logits, weights = model.decoder(z, stats.all_possible_edges)

            # universal constructor (handles sampling logic)
            G_syn = construct_synthetic_graph(adj_logits, weights, stats)

            # metric 1: modularity (communities)
            try:
                part_syn = community_louvain.best_partition(G_syn)
                Q_syn = community_louvain.modularity(part_syn, G_syn)
            except ValueError:
                Q_syn = 0
            q_err = abs(Q_syn - stats.q_orig)

            # metric 2 & 3: degree & clustering (structure)
            deg_err = calculate_degree_ks_test(G_syn, G_orig)
            clust_err = calculate_clustering_ks_test(G_syn, G_orig)

            # score: balance community preservation vs. structural accuracy
            score = q_err + gamma * (deg_err + clust_err)
            realization_scores.append(score)

    return sum(realization_scores) / 5
