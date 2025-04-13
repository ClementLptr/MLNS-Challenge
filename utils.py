from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


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


def convert_preds_to_submission(y_pred: list[tuple[int, int]]):
    """
    Converts the predictions to the submission format as a CSV file.
    Args:
        y_pred (list[tuple[int, int]]): The predicted edges.
    """
    y_pred = [(min(edge), max(edge)) for edge in y_pred]
    y_pred_set = set(y_pred)
    with open(
        f"submissions/submission{datetime.now().strftime('%m-%d_%H-%M-%S')}.csv", "w"
    ) as submission_file:
        submission_file.write("ID,Predicted\n")
        with open("data/test_edges_id_remapped.txt", "r") as test_file:
            for id, line in enumerate(test_file):
                test_edge = tuple(line.split())
                test_edge = (int(test_edge[0]), int(test_edge[1]))
                if test_edge in y_pred_set:
                    submission_file.write(f"{id},1\n")
                else:
                    submission_file.write(f"{id},0\n")


def convert_pred_1D_array_to_submission(y_pred: np.ndarray):
    """
    Converts the predictions to the submission format as a CSV file.
    Args:
        y_pred (np.array): The predicted edges.
    """
    with open(
        f"submissions/submission_{datetime.now().strftime('%m-%d_%H-%M')}.csv", "w"
    ) as submission_file:
        submission_file.write("ID,Predicted\n")
        for id, pred in enumerate(y_pred):
            submission_file.write(f"{id},{int(pred)}\n")
