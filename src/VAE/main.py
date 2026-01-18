import networkx as nx
import torch
import os
import argparse
from skopt import gp_minimize
from skopt.space import Real, Integer
import pickle

from src.VAE.graphstats import GraphStats
from src.VAE.train import objective, train_vae_model, construct_synthetic_graph
from src.utils.log import setup_logger, log_startup_banner, log_section, log_step
from src.data.load_data import load_graph_from_path, POLICY_VAE

SPACE = [
    Real(0.1, 50.0, name="gamma"),
    Integer(16, 64, name="hidden_dim"),
]

def main():
    parser = argparse.ArgumentParser(description="VAE Synthetic Network Generator")
    parser.add_argument("--input_graph", required=True, help="Path to input graph (GML/TSV)")
    parser.add_argument("--output_dir", required=True, help="Directory for intermediate artifacts (logs, models)")
    parser.add_argument("--final_dir", required=True, help="Directory for final synthetic GMLs")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "vae_run.log")
    logger = setup_logger(name="VAE", log_file=log_file)
    
    log_startup_banner(logger)
    log_section(logger, "VAE Generator Started")
    
    logger.info(f"Input: {args.input_graph}")
    logger.info(f"Artifacts: {args.output_dir}")
    logger.info(f"Final Output: {args.final_dir}")
    
    with log_step(logger, "Loading Graph"):
        try:
            G_orig = load_graph_from_path(args.input_graph, policy=POLICY_VAE)
            stats = GraphStats.from_networkx(G_orig)
            id_to_orig = nx.get_node_attributes(G_orig, "_orig_node")
            if id_to_orig:
                missing = set(range(stats.num_nodes)) - set(id_to_orig.keys())
                if missing:
                    raise RuntimeError(f"Missing _orig_node for node ids: {sorted(missing)[:10]}")
            logger.info(f"Graph loaded: {stats.num_nodes} nodes, {stats.num_edges} edges")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return

    log_section(logger, "Hyperparameter Tuning")
    
    with log_step(logger, "Running gp_minimize"):
        res = gp_minimize(
            lambda p: objective(p, stats, G_orig), SPACE, n_calls=15, random_state=42
        )
    
    logger.info(f"Best Score: {res.fun:.4f}")
    logger.info(f"Best Parameters: Gamma={res.x[0]:.2f}, Hidden Dim={res.x[1]}")
    
    with open(os.path.join(args.output_dir, "best_params.txt"), "w") as f:
        f.write(f"Gamma: {res.x[0]}\n")
        f.write(f"HiddenDim: {res.x[1]}\n")

    log_section(logger, "Generation")
    logger.info(f"Generating {args.samples} Synthetic Realizations...")

    # Reconstructs full params for final training
    # [lr, gamma, hidden_dim, dropout]
    final_params = [1e-4, res.x[0], res.x[1], 0.0]
    
    with log_step(logger, "Training Final Model"):
        best_model = train_vae_model(stats, final_params, max_epochs=1500)
    
    best_model.eval()

    x = stats.features
    edge_index = stats.all_possible_edges[:, stats.edge_mask]
    
    with log_step(logger, "Sampling Graphs"):
        z_mu, z_std = best_model.encoder(x, edge_index)
        
        dataset_name = os.path.splitext(os.path.basename(args.input_graph))[0]

        for i in range(args.samples):
            with torch.no_grad():
                z = best_model.reparametrize(z_mu, z_std)
                adj_logits, weights = best_model.decoder(z, stats.all_possible_edges)
                G_syn = construct_synthetic_graph(adj_logits, weights, stats)
                G_to_save = nx.relabel_nodes(G_syn, id_to_orig, copy=True) if id_to_orig else G_syn
                out_name = f"{dataset_name}_{i}.gml"
                out_path = os.path.join(args.final_dir, out_name)
                nx.write_gml(G_to_save, out_path)
                logger.info(f"Saved sample {i}: {out_path}")

    log_section(logger, "Saving Model Outputs")
    
    output_data = {
        "adj_logits": adj_logits.detach().cpu().numpy(),
        "z_mu": z_mu.detach().cpu().numpy(),
        "z_std": z_std.detach().cpu().numpy(),
        "final_params": final_params,
        "best_score": res.fun,
        "best_params": res.x
    }
    
    pkl_path = os.path.join(args.output_dir, "model_outputs.pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(output_data, f)
        logger.info(f"Model outputs pickled to: {pkl_path}")
    except Exception as e:
        logger.error(f"Failed to pickle model outputs: {e}")

    logger.info("VAE Pipeline Completed Successfully.")

    logger.info("VAE Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
