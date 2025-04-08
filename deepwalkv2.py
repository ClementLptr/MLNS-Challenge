import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networkx import generate_random_paths
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_networkx

from supervisedLinkPred import build_train_test_graphs

train_graph, test_graph = build_train_test_graphs(
    "data/degrees_id_remapped.csv"
)  # Use whatever data for the nodes, not gonna be used
train_graph = to_networkx(train_graph, to_undirected=True)

NUM_NODES = train_graph.number_of_nodes()
EMBEDDING_DIM = 64
PATH_LENGTH = 4  # that means 5 nodes in total
NUM_NEGATIVE_SAMPLES = 2


class DeepWalk(nn.Module):
    def __init__(self):
        super(DeepWalk, self).__init__()
        self.contextEmbedding = nn.Embedding(
            NUM_NODES,
            EMBEDDING_DIM,
        )
        self.centerEmbedding = nn.Embedding(
            NUM_NODES,
            EMBEDDING_DIM,
        )

    def forward(self, x_center: int) -> Tensor:
        """
        x_center: int, the index of the node embedding as center node
        """
        return (
            self.centerEmbedding(x_center) @ self.contextEmbedding.weight.T
        )  # similarity of the whole voc with x, the center word


rw = generate_random_paths(train_graph, sample_size=10, path_length=PATH_LENGTH)
rw_tensor = torch.tensor(list(rw))
center_index = PATH_LENGTH // 2
rw_context = torch.cat(
    (rw_tensor[:, :center_index], rw_tensor[:, center_index + 1 :]), dim=1
)
rw_center = rw_tensor[:, center_index]

y = torch.zeros((rw_tensor.size(0), NUM_NODES))
y[:, rw_context] = 1


train_dataset = TensorDataset(rw_center[:8], y[:8])
test_dataset = TensorDataset(rw_center[8:], y[8:])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(next(iter(train_dataloader)))

n_epochs = 2
model = DeepWalk()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

for i in range(n_epochs):
    train_loss = 0.0
    for X, y in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {i + 1}/{n_epochs}, Loss: {train_loss / len(train_dataloader)}")
