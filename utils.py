import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph(filepath: str):
    data = Data()
    # Read CSV and name the first column as ID
    x = pd.read_csv(filepath, header=None)
    column_names = ["ID"] + [f"feature_{i}" for i in range(1, len(x.columns))]
    x.columns = column_names

    min_id = x["ID"].min()
    max_id = x["ID"].max()
    # Check for missing IDs
    all_ids = set(range(min_id, max_id + 1))
    existing_ids = set(x["ID"])
    missing_ids = all_ids - existing_ids

    # Create rows with zeros for missing IDs
    if missing_ids:
        missing_df = pd.DataFrame(0, index=range(len(missing_ids)), columns=x.columns)
        missing_df["ID"] = list(missing_ids)
        x = pd.concat([x, missing_df]).sort_values("ID").reset_index(drop=True)

    x = x.drop("ID", axis=1).values

    data.x = torch.tensor(x)

    with open("data/train.txt") as f:
        edges = f.readlines()
        edges = [line.split() for line in edges]
        edges = [edge for edge in edges if edge[2] == "1"]
        edges = [(int(edge[0]), int(edge[1])) for edge in edges]
        source, target = [edge[0] for edge in edges], [edge[1] for edge in edges]
        data.edge_index = torch.tensor([source, target])

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


from datetime import datetime


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
