"""Paired randomisation (permutation) tests (Lecture: Evaluation - statistical significance; Smucker et al., 2007).

Null hypothesis: systems A and B are equally good, so for every query it is a coin flip which system produced
which output. We flip those coins many times, recompute the metric difference, and the p-value is the share of
flips whose |difference| is at least the observed one. For pooled metrics (micro-F1@K, macro-F1) the swap is
done on the per-query (or per-document) outputs and the pooled metric is recomputed after every swap.
"""
import numpy as np

from casetrail.eval.metrics import multilabel_f1


def randomisation_test_mean(values_a: list, values_b: list, permutations: int = 10000, seed: int = 13) -> dict:
    """Two-sided test on the mean of a per-query metric (e.g. average precision, nDCG@10)."""
    diffs = np.array(values_a) - np.array(values_b)
    observed = diffs.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(diffs)))
    permuted = (signs * diffs).mean(axis=1)
    p_value = (np.sum(np.abs(permuted) >= abs(observed) - 1e-12) + 1) / (permutations + 1)
    return {"difference": float(observed), "p_value": float(p_value)}


def micro_f1_counts(run: dict, qrels: dict, k: int) -> np.ndarray:
    """Per query [correct, retrieved, relevant] at cut-off k."""
    rows = []
    for query_id, relevant in qrels.items():
        top = run.get(query_id, [])[:k]
        rows.append([len(set(top) & relevant), len(top), len(relevant)])
    return np.array(rows, dtype=np.float64)


def micro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """counts [..., queries, 3] -> micro-F1 over the query axis."""
    correct, retrieved, relevant = counts[..., 0].sum(-1), counts[..., 1].sum(-1), counts[..., 2].sum(-1)
    precision = correct / np.maximum(retrieved, 1)
    recall = correct / np.maximum(relevant, 1)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def randomisation_test_micro_f1(counts_a: np.ndarray, counts_b: np.ndarray, permutations: int = 10000,
                                seed: int = 13) -> dict:
    observed = float(micro_f1_from_counts(counts_a) - micro_f1_from_counts(counts_b))
    rng = np.random.default_rng(seed)
    swap = rng.random((permutations, len(counts_a))) < 0.5
    perm_a = np.where(swap[..., None], counts_b, counts_a)
    perm_b = np.where(swap[..., None], counts_a, counts_b)
    permuted = micro_f1_from_counts(perm_a) - micro_f1_from_counts(perm_b)
    p_value = (np.sum(np.abs(permuted) >= abs(observed) - 1e-12) + 1) / (permutations + 1)
    return {"difference": observed, "p_value": float(p_value)}


def randomisation_test_macro_f1(pred_a: list, pred_b: list, gold: list, labels: list, permutations: int = 1000,
                                seed: int = 13) -> dict:
    """Swap whole documents' predicted label sets between A and B; macro-F1 recomputed each time (slower)."""
    observed = multilabel_f1(pred_a, gold, labels)["macro_f1"] - multilabel_f1(pred_b, gold, labels)["macro_f1"]
    rng = np.random.default_rng(seed)
    extreme = 0
    for i in range(permutations):
        swap = rng.random(len(gold)) < 0.5
        perm_a = [b if s else a for a, b, s in zip(pred_a, pred_b, swap)]
        perm_b = [a if s else b for a, b, s in zip(pred_a, pred_b, swap)]
        diff = multilabel_f1(perm_a, gold, labels)["macro_f1"] - multilabel_f1(perm_b, gold, labels)["macro_f1"]
        if abs(diff) >= abs(observed) - 1e-12:
            extreme += 1
    return {"difference": float(observed), "p_value": (extreme + 1) / (permutations + 1)}

