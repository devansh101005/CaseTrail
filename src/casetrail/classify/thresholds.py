"""Per-statute decision thresholds chosen on dev (one-vs-rest multi-label decisions; LLD section 7).

For each statute we sort the dev documents by score and pick the cut that maximises that statute's F1.
Maximising every per-statute F1 separately maximises their mean, i.e. macro-F1.
"""
import numpy as np


def best_threshold(scores: np.ndarray, gold: np.ndarray) -> float:
    """Threshold t such that predicting 'score >= t' gives the best F1 on these documents."""
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    true_pos = np.cumsum(gold[order])
    predicted = np.arange(1, len(scores) + 1)
    total_pos = gold.sum()
    if total_pos == 0:
        return float(sorted_scores[0]) + 1.0  # never predict a statute that has no dev examples
    f1 = 2 * true_pos / (predicted + total_pos)
    # Only cut between different scores, otherwise ties would be split
    valid = np.append(sorted_scores[:-1] > sorted_scores[1:], True)
    f1 = np.where(valid, f1, -1.0)
    cut = int(np.argmax(f1))
    if cut + 1 < len(scores):
        return float((sorted_scores[cut] + sorted_scores[cut + 1]) / 2)
    return float(sorted_scores[cut]) - 1e-9


def tune_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One threshold per statute (column)."""
    return np.array([best_threshold(scores[:, c], labels[:, c]) for c in range(scores.shape[1])])


def predict_sets(scores: np.ndarray, thresholds: np.ndarray, statutes: list) -> list:
    decisions = scores >= thresholds
    return [{statutes[c] for c in np.flatnonzero(row)} for row in decisions]
