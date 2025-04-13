import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import wandb
from dataPreparation import train_val_split
from models import GATWithMLPLinkPred
from supervisedLinkPred import build_train_test_graphs
from train import train_gat

wandb.init(project="challenge-graphs")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# device = "cpu"

train_graph, test_graph = (
    build_train_test_graphs()
)  # Use whatever data for the nodes, not gonna be used

train_graph, val_graph = train_val_split(train_graph)

scaler = StandardScaler()
normalized_node_features = scaler.fit_transform(train_graph.x)
train_graph.x = torch.tensor(normalized_node_features, dtype=torch.float32)
val_graph.x = torch.tensor(normalized_node_features, dtype=torch.float32)

hidden_channels = 32
out_channels = 32
num_epochs = 300
lr = 0.001
dropout = 0.6
num_features = train_graph.num_node_features

model = GATWithMLPLinkPred(num_features, hidden_channels, out_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
loss_fn = nn.BCEWithLogitsLoss()


# def objective(trial):
train_gat(
    device=device,
    model=model,
    train_graph=train_graph,
    val_graph=val_graph,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=num_epochs,
)


# study = optuna.create_study(
#     study_name="GAT-GCN",
#     direction="maximize",
# )
