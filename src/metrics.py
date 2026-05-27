from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    probabilities: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        metrics["auc_ovr"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
    except ValueError:
        metrics["auc_ovr"] = float("nan")
    return metrics


def confusion_matrix_counts(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).astype(int).tolist()
