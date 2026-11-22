"""Graph construction on hand-made inputs."""
from casetrail.graph.build_graph import citation_edges, legal_reference_edges, similarity_edges


def test_citation_edges_from_train_labels():
    queries = {"q1": {"relevant": ["c1", "c2"]}, "q2": {"relevant": ["c2"]}}
    ids, edges = citation_edges(queries)
    assert sorted(ids) == ["c1", "c2", "q1", "q2"]
    assert sorted(edges) == sorted([(ids["q1"], ids["c1"], 1.0), (ids["q1"], ids["c2"], 1.0),
                                    (ids["q2"], ids["c2"], 1.0)])


def test_reference_edges_drop_references_seen_once():
    tokens = {"d1": [["sec_302", "murder", "sec_302"], ["art_21"]], "d2": [["sec_302", "sec_34"]]}
    judgments, references, edges = legal_reference_edges(tokens, min_df=2)
    assert list(references) == ["sec_302"]
    assert sorted(edges) == [(judgments["d1"], 0, 2.0), (judgments["d2"], 0, 1.0)]


def test_similarity_edge_weights_sum_to_one_per_node():
    run = {"a": [("b", 3.0), ("c", 1.0)], "b": [("a", 2.0), ("c", 0.0)], "c": []}
    ids, edges = similarity_edges(run, ["a", "b", "c"])
    from_a = [w for source, target, w in edges if source == ids["a"]]
    assert from_a == [0.75, 0.25]
    assert [(s, t) for s, t, w in edges if s == ids["b"]] == [(ids["b"], ids["a"])]
