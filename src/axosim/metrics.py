from __future__ import annotations

import numpy as np


def soma_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_soma = np.asarray(prediction, dtype=np.float64)[..., 1]
    target_soma = np.asarray(target, dtype=np.float64)[..., 1]
    return float(np.sqrt(np.mean((pred_soma - target_soma) ** 2)))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have same shape, got {scores.shape} and {labels.shape}")

    positive = labels > 0.5
    negative = ~positive
    pos_count = int(positive.sum())
    neg_count = int(negative.sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_positive = positive[order].astype(np.float64, copy=False)
    tie_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_scores)) + 1))
    tie_ends = np.concatenate((tie_starts[1:], [len(scores)]))
    positive_per_tie = np.add.reduceat(sorted_positive, tie_starts)
    average_ranks = (tie_starts + 1 + tie_ends) / 2.0
    pos_rank_sum = float(np.dot(positive_per_tie, average_ranks))
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def spike_auc(prediction: np.ndarray, target: np.ndarray) -> float:
    return binary_auc(np.asarray(prediction)[..., 0], np.asarray(target)[..., 0])
