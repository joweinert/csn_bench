import torch
import torch.nn as nn


class DualHeadDecoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        # Simple inner-product for structure; MLP for weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(latent_channels * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Ensures weights are positive
        )

    def forward(self, z, edge_index):
        # Structure prediction (return LOGITS, not probabilities)
        row, col = edge_index
        # No sigmoid here! We use BCEWithLogitsLoss later for stability
        adj_logits = (z[row] * z[col]).sum(dim=1)

        # Weight prediction
        z_pair = torch.cat([z[row], z[col]], dim=1)
        weights = self.weight_mlp(z_pair).view(-1)

        return adj_logits, weights
