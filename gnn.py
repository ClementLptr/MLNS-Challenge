import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from torch_geometric.nn import GATConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx

# === GAT model definition ===
class DeepGATWithSkips(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads1=2, heads2=2, heads3=1):
        super().__init__()
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

# === Utils ===
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
    return node_features, edge_index, train_edges, test_edges, full_edges

def compute_pa_scores(edge_list, edge_pairs):
    G = nx.Graph()
    G.add_edges_from(edge_list.tolist())
    pa_dict = {(u, v): p for u, v, p in nx.preferential_attachment(G, edge_pairs.tolist())}
    return [pa_dict.get((u, v), pa_dict.get((v, u), 0)) for u, v in edge_pairs.tolist()]

def get_features(embeddings, edge_pairs, edge_list):
    src_embed = embeddings[edge_pairs[:, 0]]
    dst_embed = embeddings[edge_pairs[:, 1]]
    embed_feats = torch.cat([src_embed, dst_embed], dim=1).numpy()
    pa_feats = np.array(compute_pa_scores(edge_list, edge_pairs))
    return np.hstack([embed_feats, pa_feats.reshape(-1, 1)])

# === Main pipeline ===
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    x, edge_index, train_edges, test_edges, edge_list = load_data(
        "data/train_edges_id_remapped.txt",
        "data/test_edges_id_remapped.txt",
        "data/node_information.csv"
    )

    train_df, val_df = train_test_split(train_edges, test_size=0.2, stratify=train_edges['label'], random_state=42)
    train_pairs = torch.tensor(train_df[['src', 'dst']].values, dtype=torch.long)
    val_pairs = torch.tensor(val_df[['src', 'dst']].values, dtype=torch.long)
    train_labels = torch.tensor(train_df['label'].values, dtype=torch.float)
    val_labels = torch.tensor(val_df['label'].values, dtype=torch.float)

    # Load best params from Optuna
    with open("best_params_gat.json", "r") as f:
        best_params = json.load(f)

    model = DeepGATWithSkips(
        in_channels=x.shape[1],
        hidden_channels=best_params['hidden_channels'],
        out_channels=1,
        heads1=best_params['heads1'],
        heads2=best_params['heads2'],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    # === Train with early stopping ===
    best_auc = 0
    best_embeddings = None
    patience = 5
    trigger = 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out, _ = model(x, edge_index, train_pairs)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out, embeddings = model(x, edge_index, val_pairs)
            val_probs = torch.sigmoid(val_out)
            val_auc = roc_auc_score(val_labels.numpy(), val_probs.numpy())
            val_acc = accuracy_score(val_labels.numpy(), (val_probs > 0.5).int().numpy())

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_embeddings = embeddings
            trigger = 0
        else:
            trigger += 1

        if trigger > patience:
            print("Early stopping.")
            break

    # === Features for Random Forest ===
    train_feats = get_features(best_embeddings, train_pairs, edge_list)
    test_pairs = torch.tensor(test_edges[['src', 'dst']].values, dtype=torch.long)
    test_feats = get_features(best_embeddings, test_pairs, edge_list)

    # === Train RandomForest ===
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(train_feats, train_labels.numpy())
    test_preds = rf.predict_proba(test_feats)[:, 1]

    # === Save submission ===
    submission = pd.DataFrame({
        "id": np.arange(len(test_preds)),
        "predicted": test_preds
    })
    submission.to_csv("final_submission_rf.csv", index=False)
    print("✅ Submission saved to final_submission_rf.csv")
