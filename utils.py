from datetime import datetime

import numpy as np


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
