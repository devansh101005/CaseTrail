"""PageRank and HITS on tiny graphs with known answers (networkx used only as a reference check)."""
import networkx as nx
import numpy as np

from casetrail.graph.hits import hits
from casetrail.graph.pagerank import pagerank


def test_pagerank_symmetric_cycle_is_uniform():
    scores, _ = pagerank(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    assert np.allclose(scores, 1 / 3)


def test_pagerank_two_nodes_by_hand():
    # Edges 1 -> 0, 2 -> 0, 0 -> 1 (no dangling nodes), d = 0.85, N = 3, so (1 - d) / N = 0.05:
    # r2 = 0.05 (nobody links to 2), r1 = 0.05 + 0.85 r0, r0 = 0.05 + 0.85 (r1 + r2); solve for r0
    scores, _ = pagerank(3, [(1, 0, 1.0), (2, 0, 1.0), (0, 1, 1.0)], damping=0.85, tol=1e-12, max_iter=1000)
    r2 = 0.15 / 3
    r0 = (0.05 + 0.85 * (0.05 + r2)) / (1 - 0.85 ** 2)
    r1 = 0.05 + 0.85 * r0
    assert np.allclose(scores, [r0, r1, r2])
    assert abs(scores.sum() - 1.0) < 1e-9


def test_pagerank_dangling_node_and_reference():
    edges = [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 1.0), (3, 2, 1.0)]  # node 2 is dangling
    scores, _ = pagerank(4, edges, tol=1e-12, max_iter=1000)
    assert abs(scores.sum() - 1.0) < 1e-9
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_weighted_edges_from(edges)
    reference = nx.pagerank(graph, alpha=0.85, tol=1e-12, weight="weight")
    assert np.allclose(scores, [reference[i] for i in range(4)], atol=1e-8)
    assert np.argmax(scores) == 2


def test_hits_bipartite_against_reference():
    # judgments 0..2 (hubs) -> statutes 0..1 (authorities)
    edges = [(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0), (2, 0, 1.0)]
    hub, authority, _ = hits(3, 2, edges, tol=1e-12, max_iter=1000)
    graph = nx.DiGraph()
    graph.add_edges_from([(f"j{h}", f"s{a}") for h, a, w in edges])
    ref_hub, ref_auth = nx.hits(graph, tol=1e-12, max_iter=1000)
    # networkx normalises to sum 1; ours to L2 norm 1, so compare directions
    ours_auth = authority / authority.sum()
    assert np.allclose(ours_auth, [ref_auth["s0"], ref_auth["s1"]], atol=1e-6)
    ours_hub = hub / hub.sum()
    assert np.allclose(ours_hub, [ref_hub[f"j{i}"] for i in range(3)], atol=1e-6)
    assert np.argmax(hub) == 1  # judgment 1 cites both statutes
