from datetime import datetime
from typing import Literal, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.isolate import number_of_isolates
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from dataPreparation import build_train_test_graphs, train_val_split
from utils import convert_pred_1D_array_to_submission, convert_preds_to_submission


# TODO: store the results in a loadable file to avoid recomputing the features everytime
def compute_node_features(data: Data, output_filepath: str) -> None:
    """
    Build a dataset for supervised link prediction. Stores the dataset in a csv file.
    Args:
        data (Data): The input graph data.
    """
    # Get the edge indices and features
    graph = to_networkx(data, to_undirected=True)
    degrees = dict(graph.degree())
    # cluster_coeff = dict(nx.clustering(graph))
    # betweenness = dict(nx.betweenness_centrality(graph))
    # closedness = dict(nx.closeness_centrality(graph))
    # pagerank = dict(nx.pagerank(graph))
    # eigenvector = dict(nx.eigenvector_centrality(graph))
    # katz = dict(nx.katz_centrality(graph))

    node_features = pd.DataFrame(
        {
            "degree": degrees,
            # "clustering_coefficient": cluster_coeff,
            # "betweenness_centrality": betweenness,
            # "closeness_centrality": closedness,
            # "pagerank": pagerank,
            # "eigenvector_centrality": eigenvector,
            # "katz_centrality": katz,      # Convergence error
        }
    )
    node_features.insert(0, "node_id", list(graph.nodes()))

    node_features.to_csv(output_filepath, index=False)


# TODO: Add features relating to the pair of nodes
def build_dataset(
    data: Data, method: Literal["concatenate", "product"] = "product"
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset for edge prediction. The input is a concatenation of each node features,
    as well as features relating to the pair of nodes (nb of common neighbors, jaccard coefficient, etc.)
    Args:
        - data (Data): The input graph data. Is train or val data.
    Returns:
        - X (np.ndarray): The dataset with positive and negative samples.
        - y (np.ndarray): The labels for the samples (1 when there is an edge, 0 otherwise).
    """
    node_features = data.x.numpy()
    edge_index = data.edge_index.numpy()
    fake_edge_index = data.fake_edge_index.numpy()
    if method == "concatenate":
        X_positive = np.array(
            [
                np.concatenate(
                    [
                        node_features[edge_index[0, i], :],
                        node_features[edge_index[1, i], :],
                    ]
                )
                for i in range(edge_index.shape[1])
            ]
        )
        X_negative = np.array(
            [
                np.concatenate(
                    [
                        node_features[fake_edge_index[0, i], :],
                        node_features[fake_edge_index[1, i], :],
                    ]
                )
                for i in range(fake_edge_index.shape[1])
            ]
        )
    elif method == "product":
        X_positive = np.array(
            [
                node_features[edge_index[0, i]] * node_features[edge_index[1, i]]
                for i in range(edge_index.shape[1])
            ]
        )
        X_negative = np.array(
            [
                node_features[fake_edge_index[0, i]]
                * node_features[fake_edge_index[1, i]]
                for i in range(fake_edge_index.shape[1])
            ]
        )
    X = np.vstack((X_positive, X_negative))
    y = np.concatenate(
        (np.ones(X_positive.shape[0]), np.zeros(X_negative.shape[0])), axis=0
    )
    return X, y


def build_test_dataset(
    test_data: Data, method: Literal["concatenate", "product"] = "product"
) -> np.ndarray:
    """Build a dataset for edge prediction. The input is a concatenation of each node features,
    as well as features relating to the pair of nodes (nb of common neighbors, jaccard coefficient, etc.)
    Args:
        - test_data (Data): The input graph data.
    Returns:
        - X (np.ndarray): The test dataset that the model will make predictions on.
    """
    node_features = test_data.x.numpy()
    edge_index = test_data.edge_index.numpy()
    if method == "concatenate":
        X_test = np.array(
            [
                np.concatenate(
                    [node_features[edge_index[0, i]], node_features[edge_index[1, i]]]
                )
                for i in range(edge_index.shape[1])
            ]
        )
    elif method == "product":
        X_test = np.array(
            [
                node_features[edge_index[0, i]] * node_features[edge_index[1, i]]
                for i in range(edge_index.shape[1])
            ]
        )
    return X_test


def train_and_fit_model(
    train_data,
    model: Union[AdaBoostClassifier, DecisionTreeClassifier, RandomForestClassifier],
    method: Literal["product", "concatenate"] = "product",
    n_folds: int = 5,
):
    """
    Evaluates the model using cross-validation.
    Args:
        - train_data (Data): The training graph data.
        - method (Literal["product", "concatenate"]): The method to use for building the dataset.
        - model (RandomForestClassifier): The model to evaluate.
        - n_folds (int): The number of folds for cross-validation.
    Returns:
        - model (RandomForestClassifier): The trained model.
        - train_accuracies (dict): The accuracies on the training set (mean, std, values).
        - val_accuracies (dict): The accuracies on the validation set (mean, std, values).
    """
    train_accuracies_values = np.zeros(n_folds)
    val_accuracies_values = np.zeros(n_folds)

    for k in range(n_folds):
        train_data, val_data = train_val_split(train_data)
        X_train, y_train = build_dataset(train_data, method=method)
        X_val, y_val = build_dataset(val_data, method=method)
        model.fit(X_train, y_train)
        train_accuracies_values[k] = model.score(X_train, y_train)
        val_accuracies_values[k] = model.score(X_val, y_val)

    train_accuracies = {
        "mean": np.mean(train_accuracies_values),
        "std": np.std(train_accuracies_values),
        "values": train_accuracies_values,
    }

    val_accuracies = {
        "mean": np.mean(val_accuracies_values),
        "std": np.std(val_accuracies_values),
        "values": val_accuracies_values,
    }

    return model, train_accuracies, val_accuracies


if __name__ == "__main__":
    train_data, test_data = build_train_test_graphs("data/degrees_id_remapped.csv")
    train_data = to_networkx(train_data, to_undirected=True)
    print(number_of_isolates(train_data))
    # compute_node_features(train_data, "data/degrees_id_remapped.csv")
    # model = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=10,
    # )
    base_model = DecisionTreeClassifier(max_leaf_nodes=2)
    # model = AdaBoostClassifier(
    #     estimator=base_model,
    #     n_estimators=100,
    #     learning_rate=0.3,
    # )
    model, train_accuracies, val_accuracies = train_and_fit_model(
        train_data,
        model=base_model,
        method="product",
        n_folds=5,
    )
    print("Train accuracies: ", train_accuracies)
    print("Validation accuracies: ", val_accuracies)
    X_test = build_test_dataset(test_data, method="product")
    y_pred = model.predict(X_test)
    convert_pred_1D_array_to_submission(y_pred)
