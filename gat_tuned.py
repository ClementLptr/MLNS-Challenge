import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv
import json

# === GAT Model with skip connections ===
class DeepGATWithSkips(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads1=2, heads2=2, heads3=1):
        super().__init__()
        self.heads1 = heads1
        self.heads2 = heads2
        self.heads3 = heads3

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads1)
        self.gat2 = GATConv(hidden_channels * heads1, hidden_channels, heads=heads2)
        self.gat3 = GATConv(hidden_channels * heads2, hidden_channels, heads=heads3)

        self.skip1 = nn.Linear(in_channels, hidden_channels * heads1)
        self.skip2 = nn.Linear(hidden_channels * heads1, hidden_channels * heads2)
        self.skip3 = nn.Linear(hidden_channels * heads2, hidden_channels * heads3)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * heads3 * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        x1 = F.elu(self.gat1(x, edge_index)) + self.skip1(x)
        x2 = F.elu(self.gat2(x1, edge_index)) + self.skip2(x1)
        x3 = self.gat3(x2, edge_index) + self.skip3(x2)

        src = x3[edge_pairs[:, 0]]
        dst = x3[edge_pairs[:, 1]]
        edge_feat = torch.cat([src, dst], dim=1)

        return self.mlp(edge_feat).squeeze(), x3

# === Data loading utils ===
def load_data(train_path, test_path, node_info_path):
    node_df = pd.read_csv(node_info_path, header=None)
    node_features = torch.tensor(node_df.iloc[:, 1:].values, dtype=torch.float)
    train_edges = pd.read_csv(train_path, sep=' ', header=None)
    test_edges = pd.read_csv(test_path, sep=' ', header=None)
    train_edges.columns = ['src', 'dst', 'label']
    test_edges.columns = ['src', 'dst']
    pos_edges = train_edges[train_edges['label'] == 1][['src', 'dst']].values
    full_edges = np.concatenate([pos_edges, pos_edges[:, [1, 0]]])
    edge_index = torch.tensor(full_edges.T, dtype=torch.long)
    return node_features, edge_index, train_edges, test_edges

# === Objective function for Optuna ===
def objective(trial):
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    heads1 = trial.suggest_int("heads1", 1, 4)
    heads2 = trial.suggest_int("heads2", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = DeepGATWithSkips(x.shape[1], hidden_channels, 1, heads1=heads1, heads2=heads2).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(20):
        model.train()
        optimizer.zero_grad()
        out, _ = model(x, edge_index, train_pairs)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out, _ = model(x, edge_index, val_pairs)
        preds = torch.sigmoid(val_out)
        auc = roc_auc_score(val_labels.numpy(), preds.numpy())
    return auc

# === Main ===
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    x, edge_index, train_edges, test_edges = load_data(
        "data/train_edges_id_remapped.txt", "data/test_edges_id_remapped.txt", "data/node_information.csv"
    )

    train_df, val_df = train_test_split(train_edges, test_size=0.2, stratify=train_edges['label'], random_state=42)
    train_pairs = torch.tensor(train_df[['src', 'dst']].values, dtype=torch.long)
    val_pairs = torch.tensor(val_df[['src', 'dst']].values, dtype=torch.long)
    train_labels = torch.tensor(train_df['label'].values, dtype=torch.float)
    val_labels = torch.tensor(val_df['label'].values, dtype=torch.float)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params
    print("✅ Best trial:")
    print(best_params)

    with open("best_params_gat.json", "w") as f:
        json.dump(best_params, f)
