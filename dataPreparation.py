import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_train_test_graphs(node_features_filepath: str) -> tuple[Data, Data]:
    """Builds the train and test graphs from the given CSV file
    Args:
        - node_features_filepath (str): The path to the CSV file containing the node features, built by the compute_node_features function.
    Returns:
        - train_data (data): the training graph data. contains an edge_index attribute with the edges AND a fake_edge_index attribute
        with fake edges (built by using the labels in the train.txt file)

        - test_data (Data): The testing graph data. Contains only a single edge_index attribute with the edges that we will make predictions on.
    """
    # Read CSV and name the first column as ID
    x = pd.read_csv(node_features_filepath, header="infer")
    x.drop(columns=["node_id"], inplace=True)
    x = x.values

    train_data, test_data = (
        Data(x=torch.tensor(x, dtype=torch.float)),
        Data(x=torch.tensor(x, dtype=torch.float)),
    )

    with open("data/train_edges_id_remapped.txt", "r") as f:
        lines = f.readlines()
        edges = [line.split() for line in lines]
    true_edges = [(int(edge[0]), int(edge[1])) for edge in edges if edge[2] == "1"]
    fake_edges = [(int(edge[0]), int(edge[1])) for edge in edges if edge[2] == "0"]
    true_source, true_target = (
        [edge[0] for edge in true_edges],
        [edge[1] for edge in true_edges],
    )
    fake_source, fake_target = (
        [edge[0] for edge in fake_edges],
        [edge[1] for edge in fake_edges],
    )
    train_data.edge_index = torch.tensor([true_source, true_target])
    train_data.fake_edge_index = torch.tensor([fake_source, fake_target])

    with open("data/test_edges_id_remapped.txt", "r") as f:
        lines = f.readlines()
        edges = [line.split() for line in lines]
        source, target = (
            [int(edge[0]) for edge in edges],
            [int(edge[1]) for edge in edges],
        )

    test_data.edge_index = torch.tensor([source, target])

    return train_data, test_data


def train_val_split(train_data: Data, train_ratio: float = 0.8) -> tuple[Data, Data]:
    """
    Splits the graph data into training and testing sets randomly.
    Args:
        - data (Data): The graph data. The real edges are in the edge_index attribute.  The fake edges are in the
        fake_edge_index attribute. Fake edges are built from the train.txt file, where 50% are labeled as 1 and 50% as 0.
        - train_ratio (float): The proportion of edges to include in the training set.
    Returns:
        - Data, Data: The training and testing graph data.
    """
    num_edges = train_data.edge_index.size(1)
    num_fake_edges = train_data.fake_edge_index.size(1)

    edge_perm = torch.randperm(num_edges)
    fake_edge_perm = torch.randperm(num_fake_edges)

    num_train_edges = int(num_edges * train_ratio)
    num_train_fake_edges = int(num_fake_edges * train_ratio)

    train_edge_index = train_data.edge_index[:, edge_perm[:num_train_edges]]
    val_edge_index = train_data.edge_index[:, edge_perm[num_train_edges:]]

    fake_train_edge_index = train_data.fake_edge_index[
        :, fake_edge_perm[:num_train_fake_edges]
    ]
    fake_val_edge_index = train_data.fake_edge_index[
        :, fake_edge_perm[num_train_fake_edges:]
    ]

    train_data_new = Data(
        x=train_data.x,
        edge_index=train_edge_index,
        fake_edge_index=fake_train_edge_index,
    )
    val_data = Data(
        x=train_data.x, edge_index=val_edge_index, fake_edge_index=fake_val_edge_index
    )

    return train_data_new, val_data


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
