import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from utils import build_train_graph, train_test_split


def preferential_attachment_scores(data: Data) -> Tensor:
    """
    Calculate the preferential attachment score for each pair of nodes in the graph

    Args:
        data (Data): The graph data.

    Returns:
        list[tuple[int, int]]: A list of tuples containing the node pairs with the highest preferential attachment scores.
    """
    degrees = data.edge_index[1].bincount(minlength=data.num_nodes) + data.edge_index[
        0
    ].bincount(minlength=data.num_nodes)
    attachment_coef = torch.outer(degrees, degrees)
    return attachment_coef


def jaccard_index_scores(data: Data) -> Tensor:
    """
    Calculate the Jaccard index for each pair of nodes in the graph
    Args:
        data (Data): The graph data.
    Returns:
        Tensor: A tensor containing the Jaccard index scores for each pair of nodes.
    """
    adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    adj[data.edge_index[1], data.edge_index[0]] = 1

    degrees = adj.sum(dim=1)
    denominator = degrees.unsqueeze(1) + degrees.unsqueeze(0) - adj
    # Avoid division by zero by setting those positions to 1 (will result in 0 after division)
    denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    jaccard_index = (adj @ adj.T) / denominator
    return jaccard_index


def preferential_attachment_predictions(
    train_data: Data, test_data: Data, proportion: float
) -> None:
    """
    Predicts links on the test set using the preferential attachment method.
    Args:
        train_data (Data): The training graph data.
        test_data (Data): The test graph data.
        proportion (float): The proportion of edges to predict.
    """
    attachment_coef = preferential_attachment_scores(train_data)
    # Get the top k pairs with the highest scores
    k = int(proportion * (train_data.num_edges * train_data.num_edges - 1) // 2)
    assert k <= len(test_data.edge_index[0]), (
        "k exceeds the number of edges in the test set"
    )

    # Get the values and indices of the top-k coefficients
    top_k_indices = torch.topk(attachment_coef.flatten(), k=k).indices

    y_pred = [
        tuple([i // train_data.num_nodes, i % train_data.num_nodes])
        for i in top_k_indices.tolist()
    ]

    precision = len(
        set(y_pred) & set(tuple(row) for row in test_data.edge_index.t().tolist())
    ) / len(y_pred)

    recall = len(
        set(y_pred) & set(tuple(row) for row in test_data.edge_index.t().tolist())
    ) / len(test_data.edge_index[0])
    f1 = 2 * (precision * recall) / (precision + recall) + 1e-10
    print(
        f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, f1: {f1:.4f}"
    )


def jaccard_index_predictions(
    train_data: Data, test_data: Data, proportion: float
) -> None:
    """
    Predicts links on the test set using the preferential attachment method.
    Args:
        train_data (Data): The training graph data.
        test_data (Data): The test graph data.
        proportion (float): The proportion of edges to predict.
    """
    jaccard_coef = jaccard_index_scores(train_data)
    # Get the top k pairs with the highest scores
    k = int(proportion * (train_data.num_edges * train_data.num_edges - 1) // 2)
    assert k <= len(test_data.edge_index[0]), (
        "k exceeds the number of edges in the test set"
    )

    # Get the values and indices of the top-k coefficients
    top_k_indices = torch.topk(jaccard_coef.flatten(), k=k).indices

    y_pred = [
        tuple([i // train_data.num_nodes, i % train_data.num_nodes])
        for i in top_k_indices.tolist()
    ]

    precision = len(
        set(y_pred) & set(tuple(row) for row in test_data.edge_index.t().tolist())
    ) / len(y_pred)

    recall = len(
        set(y_pred) & set(tuple(row) for row in test_data.edge_index.t().tolist())
    ) / len(test_data.edge_index[0])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    print(
        f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, f1: {f1:.4f}"
    )


if __name__ == "__main__":
    data = build_train_graph("data/node_information.csv")
    train_data, test_data = train_test_split(data, train_ratio=0.8)
    preferential_attachment_predictions(train_data, test_data, proportion=1e-4)
    jaccard_index_predictions(train_data, test_data, proportion=1e-4)
