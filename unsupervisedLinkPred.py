import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from dataPreparation import build_train_test_graphs, train_test_split
from utils import convert_preds_to_submission


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


# TODO: Add the possibility to use a threshold on the attachment coeff to predict instead of a proportion
def preferential_attachment_predictions(
    train_data: Data, test_data: Data, proportion: float
) -> list[tuple[int, int]]:
    """
    Predicts links on the test set using the preferential attachment method.
    Args:
        train_data (Data): The training graph data.
        test_data (Data): The test graph data.
        proportion (float): The proportion of edges of test_data.edge_index to that will be labeled as positive.
    """
    attachment_coef = preferential_attachment_scores(train_data)
    # Get the top k pairs with the highest scores

    k = int(proportion * test_data.edge_index.shape[1])

    test_attachment_coef = attachment_coef[
        test_data.edge_index[0], test_data.edge_index[1]
    ]  # Get the scores for the test edges. Output is 1D!
    top_k_indices = torch.topk(test_attachment_coef, k=k).indices.tolist()

    y_pred = [
        (test_data.edge_index[0][index].item(), test_data.edge_index[1][index].item())
        for index in top_k_indices
    ]

    return y_pred


def jaccard_index_predictions(
    train_data: Data, test_data: Data, proportion: float
) -> list[tuple[int, int]]:
    """
    Predicts links on the test set using the preferential attachment method.
    Args:
        train_data (Data): The training graph data.
        test_data (Data): The test graph data.
        proportion (float): The proportion of edges to predict.
    """
    attachment_coef = jaccard_index_scores(train_data)
    # Get the top k pairs with the highest scores
    prediction_edge_index = torch.cat(
        [test_data.edge_index, test_data.fake_edge_index], dim=1
    )
    k = int(proportion * prediction_edge_index.shape[1])

    test_attachment_coef = attachment_coef[
        prediction_edge_index[0], prediction_edge_index[1]
    ]  # Get the scores for the test edges. Output is 1D!
    top_k_indices = torch.topk(test_attachment_coef, k=k).indices.tolist()

    y_pred = [
        (prediction_edge_index[0][index].item(), prediction_edge_index[1][index].item())
        for index in top_k_indices
    ]

    return y_pred


if __name__ == "__main__":
    train_data, test_data = build_train_test_graphs(
        "data/node_information_id_remapped.csv"
    )
    # train_data, test_data = train_test_split(
    #     data, train_ratio=0.8, negative_samples_factor=1
    # )
    preds = preferential_attachment_predictions(train_data, test_data, proportion=0.5)
    convert_preds_to_submission(preds)
    # jaccard_index_predictions(train_data, test_data, proportion=1e-4)
