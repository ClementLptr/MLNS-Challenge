import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from dataPreparation import build_train_test_graphs


def plot_graph(data: Data):
    G = to_networkx(data, to_undirected=True)

    # Use spring layout with more space between nodes
    pos = nx.spring_layout(G, k=0.15, iterations=50)

    # Draw with improved parameters
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos=pos,
        node_color="skyblue",
        node_size=5,
        edge_color="gray",
        width=0.5,
        alpha=0.7,
        with_labels=False,
    )

    plt.title(f"Graph visualization (showing {G.number_of_nodes()} nodes)")
    plt.tight_layout()
    plt.show()


def compute_graph_metrics(data):
    G = to_networkx(data, to_undirected=True)

    metrics = {}

    # Basic graph information
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)

    # Degree statistics
    degrees = dict(G.degree())
    # metrics["degree_dict"] = degrees
    if degrees:
        metrics["max_degree"] = max(degrees.values())
        metrics["min_degree"] = min(degrees.values())
        metrics["average_degree"] = sum(degrees.values()) / len(degrees)
    else:
        metrics["max_degree"] = metrics["min_degree"] = metrics["average_degree"] = None

    # Clustering coefficients
    metrics["average_clustering"] = nx.average_clustering(G)
    metrics["transitivity"] = nx.transitivity(G)

    # Centrality measures
    try:
        betweenness = nx.betweenness_centrality(G)
        # metrics["betweenness_centrality"] = betweenness
        metrics["max_betweenness"] = max(betweenness.values()) if betweenness else None
        metrics["min_betweenness"] = min(betweenness.values()) if betweenness else None
        metrics["average_betweenness"] = (
            sum(betweenness.values()) / len(betweenness) if betweenness else None
        )
    except Exception:
        metrics["betweenness_centrality"] = metrics["max_betweenness"] = None

    try:
        closeness = nx.closeness_centrality(G)
        # metrics["closeness_centrality"] = closeness
        metrics["max_closeness"] = max(closeness.values()) if closeness else None
        metrics["min_closeness"] = min(closeness.values()) if closeness else None
        metrics["average_closeness"] = (
            sum(closeness.values()) / len(closeness) if closeness else None
        )
    except Exception:
        metrics["closeness_centrality"] = metrics["max_closeness"] = None

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        # metrics["eigenvector_centrality"] = eigenvector
        metrics["max_eigenvector"] = max(eigenvector.values()) if eigenvector else None
        metrics["min_eigenvector"] = min(eigenvector.values()) if eigenvector else None
        metrics["average_eigenvector"] = (
            sum(eigenvector.values()) / len(eigenvector) if eigenvector else None
        )
    except Exception:
        metrics["eigenvector_centrality"] = metrics["max_eigenvector"] = None

    metrics["self_loops"] = G.number_of_selfloops()

    # # Shortest path metrics (compute on the largest connected component if not connected)
    # if nx.is_connected(G):
    #     metrics["average_shortest_path_length"] = nx.average_shortest_path_length(G)
    #     metrics["diameter"] = nx.diameter(G)
    #     metrics["is_connected"] = True
    # else:
    #     largest_cc = max(nx.connected_components(G), key=len)
    #     G_lcc = G.subgraph(largest_cc)
    #     metrics["average_shortest_path_length_lcc"] = nx.average_shortest_path_length(
    #         G_lcc
    #     )
    #     metrics["diameter_lcc"] = nx.diameter(G_lcc)
    #     metrics["is_connected"] = False

    # # Assortativity
    # try:
    #     metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
    # except Exception:
    #     metrics["degree_assortativity"] = None

    # Connected components
    metrics["number_connected_components"] = nx.number_connected_components(G)
    metrics["largest_component_size"] = len(max(nx.connected_components(G), key=len))

    # Community detection and modularity
    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        metrics["num_communities"] = len(communities)
        # metrics["communities"] = [list(c) for c in communities]
        metrics["modularity"] = nx.algorithms.community.quality.modularity(
            G, communities
        )
    except Exception:
        metrics["num_communities"] = metrics["communities"] = metrics["modularity"] = (
            None
        )

    # Spectral properties: Algebraic connectivity (Fiedler value)
    try:
        if nx.is_connected(G):
            metrics["algebraic_connectivity"] = nx.algebraic_connectivity(G)
        else:
            metrics["algebraic_connectivity"] = nx.algebraic_connectivity(G_lcc)
    except Exception:
        metrics["algebraic_connectivity"] = None

    # plot degrees distribution
    plt.hist(
        list(degrees.values()),
        bins=max(degrees.values()) - min(degrees.values()) + 1,
        color="blue",
        alpha=0.7,
    )
    plt.title("Degree Distribution")
    plt.yscale("log")
    # plt.xscale("log")
    plt.savefig("results/degree_distribution.png")

    return metrics


if __name__ == "__main__":
    data = build_train_test_graphs("data/node_information_id_remapped.csv")
    # plot_graph(data)
    metrics = compute_graph_metrics(data)
    print(metrics)
