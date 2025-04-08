import torch
from torch.utils.data import TensorDataset, 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networkx import generate_random_paths
from torch import Tensor
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

    def forward(self, x_center: int, x_context: list[int]):
        """
        x_center: int, the index of the node embedding as center node
        x_context: Tensor[int], the indices of the node embeddings as context nodes. Includes postive and negative context
        """
        return (
            self.contextEmbedding[x_context] @ self.centerEmbedding[x_center]
        )  # similarity of the whole voc with x, the center word


rw = generate_random_paths(train_graph, sample_size=3, path_length=PATH_LENGTH)
rw_list = list(rw)
print("rw list", rw_list)
x_center = torch.tensor([path[len(path) // 2] for path in rw_list])
# Take one node at random from the context
rw_list_without_center = [
    path[: len(path) // 2] + path[len(path) // 2 + 1 :] for path in rw_list
]

# print(rw_list_without_center)
y = torch.tensor(
    [
        path[
            torch.randint(low=0, high=len(rw_list_without_center[0]), size=(1,)).item()
        ]
        for path in rw_list_without_center
    ]
)
y_negative = torch.randint(NUM_NODES, (len(x_center), NUM_NEGATIVE_SAMPLES))

x_context = torch.cat([y, y_negative], dim=0)


print("y_positive", y)
print("y_neg", y_negative)

dataset = 

n_epochs = 2
model = DeepWalk()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

for i in range(n_epochs)
    optimizer.zero_grad()
    output = model(X)
