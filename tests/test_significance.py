"""Randomisation tests on hand-made cases."""
from casetrail.eval.metrics import micro_prf_at_k
from casetrail.eval.significance import (micro_f1_counts, micro_f1_from_counts, randomisation_test_macro_f1,
                                         randomisation_test_mean, randomisation_test_micro_f1)

RUN = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]}
QRELS = {"q1": {"a", "c"}, "q2": {"z", "w"}}


def test_identical_systems_are_not_significant():
    result = randomisation_test_mean([0.5, 0.2, 0.9], [0.5, 0.2, 0.9], permutations=500)
    assert result["difference"] == 0 and result["p_value"] == 1.0


def test_clear_difference_is_significant():
    better = [0.9] * 30
    worse = [0.1] * 30
    assert randomisation_test_mean(better, worse, permutations=2000)["p_value"] < 0.01


def test_vectorised_micro_f1_matches_metrics_module():
    counts = micro_f1_counts(RUN, QRELS, 2)
    assert abs(float(micro_f1_from_counts(counts)) - micro_prf_at_k(RUN, QRELS, 2)["f1"]) < 1e-12


def test_micro_f1_test_runs():
    counts_a = micro_f1_counts(RUN, QRELS, 3)
    counts_b = micro_f1_counts({"q1": ["b", "x", "y"], "q2": ["a", "b", "c"]}, QRELS, 3)
    result = randomisation_test_micro_f1(counts_a, counts_b, permutations=200)
    assert result["difference"] > 0 and 0 < result["p_value"] <= 1


def test_macro_f1_test_identical_predictions():
    gold = [{"s1"}, {"s2"}, {"s1", "s2"}]
    pred = [{"s1"}, set(), {"s2"}]
    result = randomisation_test_macro_f1(pred, pred, gold, ["s1", "s2"], permutations=50)
    assert result["difference"] == 0 and result["p_value"] == 1.0
