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
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    tied_ranks = _average_tied_ranks(scores, ranks)
    pos_rank_sum = tied_ranks[positive].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def spike_auc(prediction: np.ndarray, target: np.ndarray) -> float:
    return binary_auc(np.asarray(prediction)[..., 0], np.asarray(target)[..., 0])


def _average_tied_ranks(scores: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    result = ranks.copy()
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for score_idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == score_idx
            result[mask] = ranks[mask].mean()
    return result
