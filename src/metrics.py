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


def predictive_entropy(probabilities: list[float] | np.ndarray) -> float:
    """Normalised Shannon entropy of a softmax distribution, scaled to [0, 1].

    0.0 means all probability mass sits on one class (maximally confident);
    1.0 means the distribution is uniform over all classes (maximally
    uncertain).

    Caveat (see issue #17): a classically overconfident out-of-distribution
    prediction has LOW entropy despite being wrong — the softmax still
    collapses onto one class even for garbage input. Entropy alone cannot
    catch that failure mode; it is a cheap *additional* signal, not a
    replacement for an OOD detector. `src/api/main.py`'s `/predict` combines
    it with a reconstruction-error cross-check against the autoencoder
    (already used for `/anomaly`) to flag `likely_ood`.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    n = p.shape[-1]
    if n <= 1:
        return 0.0
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / float(np.log(n))
