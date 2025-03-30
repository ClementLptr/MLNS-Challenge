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

    with open("data/train.txt", "r") as f:
        edges = f.readlines()
        edges = [line.split() for line in edges]
        edges = [edge for edge in edges if edge[2] == "1"]
        edges = [(int(edge[0]), int(edge[1])) for edge in edges]
        source, target = [edge[0] for edge in edges], [edge[1] for edge in edges]
        data.edge_index = torch.tensor([source, target])

    with open("data/test.txt", "r") as f:
        edges = f.readlines()
        edges = [line.split() for line in edges]
        source, target = (
            [int(edge[0]) for edge in edges],
            [int(edge[1]) for edge in edges],
        )
        data.test_edges = torch.tensor([source, target])

    return data


def train_test_split(data: Data, train_ratio: float = 0.8) -> tuple[Data, Data]:
    """
    Splits the graph data into training and testing sets.
    Args:
        data (Data): The graph data.
        train_ratio (float): The proportion of edges to include in the training set.
    Returns:
        Data, Data: The training and testing graph data.
    """
    num_edges = data.edge_index.size(1)
    num_train_edges = int(num_edges * train_ratio)

    train_edge_index = data.edge_index[:, :num_train_edges]
    test_edge_index = data.edge_index[:, num_train_edges:]

    train_data = Data(x=data.x, edge_index=train_edge_index)
    test_data = Data(x=data.x, edge_index=test_edge_index)

    return train_data, test_data


if __name__ == "__main__":
    data = build_graph("data/node_information.csv")
