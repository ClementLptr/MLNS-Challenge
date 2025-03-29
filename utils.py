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


data = build_graph("data/node_information.csv")
print(data.x)
print(data.edge_index)
