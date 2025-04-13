import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATWithMLPLinkPred(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6
    ):
        super(GATWithMLPLinkPred, self).__init__()
        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        # MLP for edge prediction
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index):
        # Node embedding
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Extract node features for each edge
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)  # Concatenate node features

        # Pass through MLP
        return self.mlp(edge_features).squeeze(1)
