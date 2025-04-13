import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Node2Vec
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx

# === Model ===
class GraphSAGEWithSkipAndMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)

        self.skip = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.3)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, node_information, edge_index, edge_pairs):
        x1 = F.relu(self.sage1(node_information, edge_index))
        x2 = self.dropout(F.relu(self.sage2(x1, edge_index)))
        x3 = self.sage3(x2, edge_index)

        x_skip = self.skip(node_information)
        x_combined = x3 + x_skip

        src = x_combined[edge_pairs[:, 0]]
        dst = x_combined[edge_pairs[:, 1]]
        edge_feat = torch.cat([src, dst], dim=1)

        return self.mlp(edge_feat).squeeze()

# === Load and preprocess data ===
def load_data(train_path, test_path):
    train_edges = pd.read_csv(train_path, sep=' ', header=None)
    test_edges = pd.read_csv(test_path, sep=' ', header=None)

    train_edges.columns = ['src', 'dst', 'label']
    test_edges.columns = ['src', 'dst']

    full_edges = train_edges[train_edges['label'] == 1][['src', 'dst']].values
    full_edges = np.concatenate([full_edges, full_edges[:, [1, 0]]], axis=0)
    edge_index = torch.tensor(full_edges.T, dtype=torch.long)

    num_nodes = max(edge_index.max().item(), train_edges[['src', 'dst']].values.max(), test_edges[['src', 'dst']].values.max()) + 1

    return edge_index, train_edges, test_edges, num_nodes

# === Structural Features ===
def compute_structural_features(edge_index, num_nodes):
    G = nx.Graph()
    G.add_edges_from(edge_index.T.tolist())
    G.remove_edges_from(nx.selfloop_edges(G))

    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    betweenness = nx.betweenness_centrality(G)
    kcore = nx.core_number(G)
    triangle = nx.triangles(G)

    feat = np.zeros((num_nodes, 7))
    for node in range(num_nodes):
        feat[node] = [
            degrees.get(node, 0),
            clustering.get(node, 0),
            pagerank.get(node, 0),
            eigenvector.get(node, 0),
            betweenness.get(node, 0),
            kcore.get(node, 0),
            triangle.get(node, 0)
        ]
    return torch.tensor(feat, dtype=torch.float)

# === Training logic ===
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.node_information, data.edge_index, data.train_edge_index)
    loss = criterion(out, data.train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, edge_index, labels):
    model.eval()
    with torch.no_grad():
        out = model(data.node_information, data.edge_index, edge_index)
        preds = torch.sigmoid(out)
        auc = roc_auc_score(labels.cpu(), preds.cpu())
        acc = accuracy_score(labels.cpu(), (preds > 0.5).int().cpu())
        return auc, acc

# === Full pipeline ===
def main():
    train_path = 'data/train_edges_id_remapped.txt'
    test_path = 'data/test_edges_id_remapped.txt'

    edge_index, train_edges, test_edges, num_nodes = load_data(train_path, test_path)

    # Node2Vec embeddings
    node2vec = Node2Vec(edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10,
                        num_negative_samples=1, sparse=True).to('cpu')
    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer_n2v = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)
    node2vec.train()
    for epoch in range(1, 101):
        for pos_rw, neg_rw in loader:
            optimizer_n2v.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer_n2v.step()

    embeddings = node2vec.embedding.weight.data.cpu()

    # Structural features
    struct_feats = compute_structural_features(edge_index, num_nodes)
    struct_feats = (struct_feats - struct_feats.mean(0)) / (struct_feats.std(0) + 1e-6)

    # Concatenate Node2Vec + structural features
    node_information = torch.cat([embeddings, struct_feats], dim=1)

    # Split train/val
    train_df, val_df = train_test_split(train_edges, test_size=0.2, random_state=42, stratify=train_edges['label'])
    train_edge_index = torch.tensor(train_df[['src', 'dst']].values, dtype=torch.long)
    train_labels = torch.tensor(train_df['label'].values, dtype=torch.float)
    val_edge_index = torch.tensor(val_df[['src', 'dst']].values, dtype=torch.long)
    val_labels = torch.tensor(val_df['label'].values, dtype=torch.float)

    data = Data(node_information=node_information, edge_index=edge_index)
    data.train_edge_index = train_edge_index
    data.train_labels = train_labels

    model = GraphSAGEWithSkipAndMLP(in_channels=node_information.shape[1], hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    pos_weight_val = (train_labels == 0).sum() / (train_labels == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))

    best_auc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(1, 301):
        loss = train(model, data, optimizer, criterion)
        val_auc, val_acc = evaluate(model, data, val_edge_index, val_labels)
        print(f"Epoch {epoch:03d} - Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_state)

    # Predict on test set
    test_edge_index = torch.tensor(test_edges[['src', 'dst']].values, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        test_logits = model(data.node_information, data.edge_index, test_edge_index)
        test_preds = (torch.sigmoid(test_logits) > 0.5).long()

    submission = pd.DataFrame({
        'ID': range(len(test_preds)),
        'Predicted': test_preds.cpu().numpy()
    })
    submission.to_csv('node2vec_structural_predictions.csv', index=False)
    print("Test predictions saved in 'node2vec_structural_predictions.csv'.")

if __name__ == '__main__':
    main()