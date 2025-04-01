def precision(y_pred: list[tuple[int, int]], y_true: list[tuple[int, int]]) -> float:
    """
    Calculate precision of predictions.
    Args:
        y_pred (list[tuple[int, int]]): Predicted edges.
        y_true (list[tuple[int, int]]): True edges.
    Returns:
        float: Precision score.
    """
    # Convert edges to tuples of (min, max) to avoid inconsistencies
    y_pred = [(min(edge), max(edge)) for edge in y_pred]
    y_true = [(min(edge), max(edge)) for edge in y_true]
    precision = len(set(y_pred) & set(y_true)) / len(y_pred)
    return precision


def recall(y_pred: list[tuple[int, int]], y_true: list[tuple[int, int]]) -> float:
    """
    Calculate recall of predictions.
    Args:
        y_pred (list[tuple[int, int]]): Predicted edges.
        y_true (list[tuple[int, int]]): True edges.
    Returns:
        float: Recall score.
    """
    # Convert edges to tuples of (min, max) to avoid inconsistencies
    y_pred = [(min(edge), max(edge)) for edge in y_pred]
    y_true = [(min(edge), max(edge)) for edge in y_true]
    recall = len(set(y_pred) & set(y_true)) / len(y_true)
    return recall


def f1_score(y_pred: list[tuple[int, int]], y_true: list[tuple[int, int]]) -> float:
    """
    Calculate F1 score of predictions.
    Args:
        y_pred (list[tuple[int, int]]): Predicted edges.
        y_true (list[tuple[int, int]]): True edges.
    Returns:
        float: F1 score.
    """
    # Convert edges to tuples of (min, max) to avoid inconsistencies
    y_pred = [(min(edge), max(edge)) for edge in y_pred]
    y_true = [(min(edge), max(edge)) for edge in y_true]
    f1_score = (2 * precision(y_pred, y_true) * recall(y_pred, y_true)) / (
        precision(y_pred, y_true) + recall(y_pred, y_true) + 1e-10
    )
    return f1_score
