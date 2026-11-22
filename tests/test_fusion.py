"""Fusion on hand-made runs."""
from casetrail.fusion.combine import ids_only, min_max, rrf, weighted_sum

KEYWORD = {"q": [("a", 10.0), ("b", 5.0), ("c", 1.0)]}
MEANING = {"q": [("c", 0.9), ("a", 0.8), ("d", 0.1)]}


def test_rrf_by_hand():
    fused = rrf([KEYWORD, MEANING], k=60)["q"]
    scores = dict(fused)
    assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(scores["c"] - (1 / 63 + 1 / 61)) < 1e-12
    assert ids_only({"q": fused})["q"][:2] == ["a", "c"]


def test_min_max_and_weighted_sum():
    assert min_max(KEYWORD["q"]) == {"a": 1.0, "b": 4 / 9, "c": 0.0}
    fused = dict(weighted_sum([KEYWORD, MEANING], [0.5, 0.5])["q"])
    assert abs(fused["a"] - (0.5 * 1.0 + 0.5 * 0.875)) < 1e-12
    assert abs(fused["d"] - 0.0) < 1e-12   # last in the neural list, absent from BM25
    only_keyword = ids_only(weighted_sum([KEYWORD, MEANING], [1.0, 0.0]))["q"]
    assert only_keyword[:3] == ["a", "b", "c"]
