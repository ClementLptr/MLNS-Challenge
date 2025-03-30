import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph(filepath: str):
    data = Data()
    # Read CSV and name the first column as ID
    x = pd.read_csv(filepath, header=None)
    column_names = ["ID"] + [f"feature_{i}" for i in range(1, len(x.columns))]
    x.columns = column_names
    x = x.drop("ID", axis=1).values

    data.x = torch.tensor(x)

    with open("data/train_edges_id_remapped.txt", "r") as f:
        edges = f.readlines()
        edges = [line.split() for line in edges]
        edges = [edge for edge in edges if edge[2] == "1"]
        edges = [(int(edge[0]), int(edge[1])) for edge in edges]
        source, target = [edge[0] for edge in edges], [edge[1] for edge in edges]
        data.edge_index = torch.tensor([source, target])

    with open("data/test_edges_id_remapped.txt", "r") as f:
        edges = f.readlines()
        edges = [line.split() for line in edges]
        source, target = (
            [int(edge[0]) for edge in edges],
            [int(edge[1]) for edge in edges],
        )
        data.test_edges = torch.tensor([source, target])

    return data


def train_test_split(
    data: Data, train_ratio: float = 0.8, negative_samples_factor: int = 1
) -> tuple[Data, Data]:
    """
    Splits the graph data into training and testing sets.
    Args:
        data (Data): The graph data.
        train_ratio (float): The proportion of edges to include in the training set.
        negative_samples_factor (int): The factor by which to increase the number of negative samples.
    Returns:
        Data, Data: The training and testing graph data.
    """
    num_edges = data.edge_index.size(1)
    num_train_edges = int(num_edges * train_ratio)
    num_nodes = data.x.size(0)

    train_edge_index = data.edge_index[:, :num_train_edges]
    test_edge_index = data.edge_index[:, num_train_edges:]

    fake_edge_index = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, negative_samples_factor * test_edge_index.size(1)),
    )

    train_data = Data(x=data.x, edge_index=train_edge_index)
    test_data = Data(x=data.x, edge_index=test_edge_index)
    test_data.fake_edge_index = fake_edge_index

    return train_data, test_data


def remap_node_ids() -> None:
    """
    For some reason, the indices of the nodes in the graph are not "continuous".
    To simplify operations on graphs, we need to map the node indices to ranges from 0 to n-1.
    Args:
        data (Data): The graph data.
    Returns:
        dict: A dictionary mapping the original node indices to range(0, n).
    """
    nodes = np.loadtxt("data/node_information.csv", dtype=float, delimiter=",")
    train_edges = np.loadtxt("data/train.txt", dtype=int, delimiter=" ")
    test_edges = np.loadtxt("data/test.txt", dtype=int, delimiter=" ")

    nodes[:, 0] = np.arange(len(nodes))

    mapping = json.load(open("data/node_id_mapping.json", "r"))
    mapping = {int(v): int(k) for k, v in mapping.items()}

    # Apply mapping to train edges
    for i in range(len(train_edges)):
        train_edges[i, 0] = mapping[train_edges[i, 0]]
        train_edges[i, 1] = mapping[train_edges[i, 1]]

    # Apply mapping to test edges
    for i in range(len(test_edges)):
        test_edges[i, 0] = mapping[test_edges[i, 0]]
        test_edges[i, 1] = mapping[test_edges[i, 1]]

    # Save the remapped edge files
    np.savetxt("data/train_edges_id_remapped.txt", train_edges, fmt="%d", delimiter=" ")
    np.savetxt("data/test_edges_id_remapped.txt", test_edges, fmt="%d", delimiter=" ")
    np.savetxt("data/node_information_id_remapped.csv", nodes, fmt="%d", delimiter=",")


def node_id_mapping() -> None:
    """
    Creates a mapping of the new Ids to the original ones to convert back before submission
    Returns the mapping dictionary and saves it to a JSON file
    """
    nodes = np.loadtxt("data/node_information.csv", dtype=float, delimiter=",")
    mapping = {i: int(nodes[i, 0]) for i in range(len(nodes))}
    with open("data/node_id_mapping.json", "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    # data = build_graph()
    # node_id_mapping()
    remap_node_ids()
    # remap_node_ids("data/node_information.csv")
