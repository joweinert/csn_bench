import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout

        # Project input features to match hidden dim for addition
        self.project = torch.nn.Linear(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        # GCN Layer
        hidden = self.conv1(x, edge_index).relu()

        # Residual Connection
        # original node features back into the signal
        # -> preserves unique node identity against GCN smoothing
        x_residual = self.project(x)
        hidden = hidden + x_residual

        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)

        # Output Layers
        return self.conv_mu(hidden, edge_index), self.conv_logstd(hidden, edge_index)
