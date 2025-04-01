import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from dataPreparation import build_train_graph


def compute_node_features(data: Data) -> np.ndarray:
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
            # "katz_centrality": katz,
        }
    )

    return node_features


if __name__ == "__main__":
    data = build_train_graph("data/node_information.csv")
    node_features = compute_node_features(data)
    print(node_features)
