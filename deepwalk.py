import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler
import yaml
from networkx import generate_random_paths
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

import wandb
from supervisedLinkPred import build_train_test_graphs

wandb.init(project="challenge-graphs")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# device = "cpu"

train_graph, test_graph = build_train_test_graphs(
    "data/degrees_id_remapped.csv"
)  # Use whatever data for the nodes, not gonna be used
train_graph = to_networkx(train_graph, to_undirected=True)

NUM_NODES = train_graph.number_of_nodes()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

EMBEDDING_DIM = config["embedding_dim"]
LR = config["lr"]
DATASET_SIZE = config["dataset_size"]
PATH_LENGTH = config["path_length"]
N_EPOCHS = config["n_epochs"]
BATCH_SIZE = config["batch_size"]


class DeepWalk(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super(DeepWalk, self).__init__()
        self.contextEmbedding = nn.Embedding(num_nodes, embedding_dim)
        self.centerEmbedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, x_center: int) -> Tensor:
        """
        x_center: int, the index of the node embedding as center node
        """
        return (
            self.centerEmbedding(x_center) @ self.contextEmbedding.weight.T
        )  # similarity of the whole voc with x, the center word


rw = generate_random_paths(
    train_graph, sample_size=DATASET_SIZE, path_length=PATH_LENGTH
)
rw_tensor = torch.tensor(list(rw))
center_index = PATH_LENGTH // 2
rw_context = torch.cat(
    (rw_tensor[:, :center_index], rw_tensor[:, center_index + 1 :]), dim=1
)
rw_center = rw_tensor[:, center_index].to(device)

y = torch.zeros((rw_tensor.size(0), NUM_NODES)).to(device)
for i in range(rw_tensor.size(0)):
    context = rw_context[i]
    y[i, context] = 1


train_dataset = TensorDataset(rw_center[:8], y[:8])
test_dataset = TensorDataset(rw_center[8:], y[8:])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(next(iter(train_dataloader)))

model = DeepWalk(NUM_NODES, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# profile_dir = "wandb/profiler"

# profiler = torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         # torch.profiler.ProfilerActivity.CUDA,
#     ],
#     # schedule=schedule,  # see the profiler docs for details on scheduling
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
#     with_stack=True,
#     profile_memory=True,
#     record_shapes=True,
# )


wandb.watch(model, log="all", criterion=loss_fn, log_freq=10000, log_graph=True)

for i in tqdm(range(N_EPOCHS)):
    train_loss = 0.0
    for batch_index, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # with profiler:
    with torch.no_grad():
        test_loss = 0.0
        for X, y in test_dataloader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    # profiler.export_chrome_trace(
    #     f"{profile_dir}/{i}.pt.trace.json"
    # )  # Export the trace to a file
    # profiler.export_memory_timeline(
    #     f"{profile_dir}/{i}.pt.memory.json"
    # )  # Export the memory timeline to a file
    # profiler.export_stacks(
    #     f"{profile_dir}/{i}.pt.stacks.json",  # Export the stacks to a file
    # )  # Export the stacks to a file

    wandb.log(
        {
            "train_loss": train_loss / len(train_dataloader),
            "test_loss": test_loss / len(test_dataloader),
        }
    )

    if i % 5 == 0:
        print(
            f"Epoch {i + 1}/{N_EPOCHS}, Train Loss: {train_loss / len(train_dataloader)}"
        )
        print(f"Test Loss: {test_loss / len(test_dataloader)}")


# # create a wandb Artifact
# profile_art = wandb.Artifact("trace", type="profile")
# # add the pt.trace.json files to the Artifact
# profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# # log the artifact
