from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from dataPreparation import build_train_test_graphs, train_test_split
from utils import convert_preds_to_submission


# TODO: store the results in a loadable file to avoid recomputing the features everytime
def compute_node_features(data: Data) -> pd.DataFrame:
    """
    Build a dataset for supervised link prediction.
    Args:
        data (Data): The input graph data.
    Returns:
        np.ndarray: The dataset with positive and negative samples.
    """
    # Get the edge indices and features
    graph = to_networkx(data, to_undirected=True)
    degrees = dict(graph.degree())
    cluster_coeff = dict(nx.clustering(graph))
    betweenness = dict(nx.betweenness_centrality(graph))
    closedness = dict(nx.closeness_centrality(graph))
    pagerank = dict(nx.pagerank(graph))
    eigenvector = dict(nx.eigenvector_centrality(graph))
    # katz = dict(nx.katz_centrality(graph))

    node_features = pd.DataFrame(
        {
            "degree": degrees,
            "clustering_coefficient": cluster_coeff,
            "betweenness_centrality": betweenness,
            "closeness_centrality": closedness,
            "pagerank": pagerank,
            "eigenvector_centrality": eigenvector,
            # "katz_centrality": katz,      # Convergence error
        }
    )

    return node_features


# TODO: Add features relating to the pair of nodes
def build_dataset(
    data: Data, negative_samples_factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset for edge prediction. The input is a concatenation of each node features,
    as well as features relating to the pair of nodes (nb of common neighbors, jaccard coefficient, etc.)
    Args:
        - data (Data): The input graph data. Is train or val data.
        - negative_samples_factor (int): The number of negative samples per positive sample.
    Returns:
        - X (np.ndarray): The dataset with positive and negative samples.
        - y (np.ndarray): The labels for the samples (1 when there is an edge, 0 otherwise).
    """
    node_features = compute_node_features(data)
    edge_index = data.edge_index.numpy()
    fake_edges = np.random.randint(
        0, data.num_nodes, size=(2, negative_samples_factor * edge_index.shape[1])
    )
    X_positive = np.array(
        [
            node_features.iloc[edge_index[0, i]].tolist()
            + node_features.iloc[edge_index[1, i]].tolist()
            for i in range(edge_index.shape[1])
        ]
    )
    X_negative = np.array(
        [
            node_features.iloc[fake_edges[0, i]].tolist()
            + node_features.iloc[fake_edges[1, i]].tolist()
            for i in range(fake_edges.shape[1])
        ]
    )
    X = np.vstack((X_positive, X_negative))
    y = np.concatenate(
        (np.ones(X_positive.shape[0]), np.zeros(X_negative.shape[0])), axis=0
    )
    return X, y


def build_test_dataset(
    test_data: Data,
) -> tuple[np.ndarray]:
    """Build a dataset for edge prediction. The input is a concatenation of each node features,
    as well as features relating to the pair of nodes (nb of common neighbors, jaccard coefficient, etc.)
    Args:
        - test_data (Data): The input graph data.
    Returns:
        - X (np.ndarray): The test dataset that the model will make predictions on.
    """
    node_features = compute_node_features(test_data)
    edge_index = test_data.edge_index.numpy()
    X_test = np.array(
        [
            node_features.iloc[edge_index[0, i]].tolist()
            + node_features.iloc[edge_index[1, i]].tolist()
            for i in range(edge_index.shape[1])
        ]
    )
    return X_test


if __name__ == "__main__":
    train_data, test_data = build_train_test_graphs("data/node_information.csv")
    train_data, val_data = train_test_split(train_data)
    # node_features = compute_node_features(train_data)
    # print(node_features)
    X_train, y_train = build_dataset(train_data, 1)
    X_val, y_val = build_dataset(val_data, 1)
    X_test = build_test_dataset(test_data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    with open(f"submissions/submission{datetime.now()}.csv", "w") as submission_file:
        submission_file.write("ID,Predicted\n")
        for id, pred in enumerate(y_pred):
            submission_file.write(f"{id},{int(pred)}\n")
    print("Train accuracy: ", model.score(X_train, y_train))
    print("Validation accuracy: ", model.score(X_val, y_val))
    # print(val_data)
