"""Metrics against hand-computed values (and scikit-learn as a reference for the multi-label F1)."""
import math

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from casetrail.eval.metrics import (average_precision, best_k, mean_over_queries, micro_prf_at_k, multilabel_f1,
                                    ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank)

RUN = {"q1": ["a", "b", "c", "d"], "q2": ["x", "y", "z", "w"]}
QRELS = {"q1": {"a", "c"}, "q2": {"w", "v", "u"}}


def test_micro_prf_at_k_pools_counts_over_queries():
    # K = 2: q1 gets a (1 correct of 2), q2 gets none -> P = 1/4, R = 1/5
    result = micro_prf_at_k(RUN, QRELS, 2)
    assert result["precision"] == 0.25 and result["recall"] == 0.2
    assert abs(result["f1"] - 2 * 0.25 * 0.2 / 0.45) < 1e-12
    # K = 4: correct = 2 + 1 = 3, retrieved 8, relevant 5
    result = micro_prf_at_k(RUN, QRELS, 4)
    assert result["precision"] == 3 / 8 and result["recall"] == 3 / 5


def test_best_k_picks_highest_f1():
    k, score = best_k(RUN, QRELS, [1, 2, 3, 4])
    assert k == 4 and abs(score - micro_prf_at_k(RUN, QRELS, 4)["f1"]) < 1e-12


def test_ranking_metrics_by_hand():
    ranked, relevant = ["a", "b", "c", "d"], {"a", "c"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert average_precision(ranked, relevant) == (1 / 1 + 2 / 3) / 2
    assert reciprocal_rank(["b", "a"], relevant) == 0.5
    expected_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(ranked, relevant, 10) - expected_ndcg) < 1e-12


def test_mean_over_queries():
    assert mean_over_queries(RUN, QRELS, precision_at_k, 4) == (2 / 4 + 1 / 4) / 2
    assert mean_over_queries(RUN, QRELS, reciprocal_rank) == (1.0 + 1 / 4) / 2


def test_multilabel_f1_matches_sklearn():
    labels = ["s1", "s2", "s3"]
    gold = [{"s1", "s2"}, {"s2"}, {"s3"}, {"s1"}]
    pred = [{"s1"}, {"s2", "s3"}, set(), {"s1", "s2"}]
    ours = multilabel_f1(pred, gold, labels)
    binarizer = MultiLabelBinarizer(classes=labels)
    y_true, y_pred = binarizer.fit_transform(gold), binarizer.transform(pred)
    assert abs(ours["macro_f1"] - f1_score(y_true, y_pred, average="macro", zero_division=0)) < 1e-12
    assert abs(ours["micro_f1"] - f1_score(y_true, y_pred, average="micro", zero_division=0)) < 1e-12
