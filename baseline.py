"""Some trivial methods to gauge the difficulty of the dataset"""

from torch_geometric.data import Data


def baseline(data: Data, degree_threshold: int):
    """
    We predict the existence of an edge"""
