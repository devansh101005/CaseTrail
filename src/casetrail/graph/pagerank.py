"""PageRank by power iteration (Lecture: PageRank - random surfer, damping, dangling nodes).

A random surfer follows an out-link with probability d (choosing links in proportion to their weight)
and jumps to a random node with probability 1 - d. Dangling nodes (no out-links) spread their score
evenly over all nodes. The PageRank vector is the stationary distribution of this walk:
  r_new[v] = (1 - d) / N + d * ( sum_{u -> v} r[u] * w(u, v) / W(u)  +  sum_{dangling u} r[u] / N )
"""
import numpy as np


def pagerank(num_nodes: int, edges: list, damping: float = 0.85, tol: float = 1e-8, max_iter: int = 100) -> tuple:
    """edges = [(source, target, weight)]. Returns (scores summing to 1, iterations used)."""
    if num_nodes == 0:
        return np.zeros(0), 0
    sources = np.array([edge[0] for edge in edges], dtype=np.int64)
    targets = np.array([edge[1] for edge in edges], dtype=np.int64)
    weights = np.array([edge[2] for edge in edges], dtype=np.float64)
    out_weight = np.bincount(sources, weights=weights, minlength=num_nodes)
    dangling = out_weight == 0
    # Fraction of u's score that flows along each edge u -> v
    edge_share = weights / np.where(out_weight[sources] > 0, out_weight[sources], 1.0)

    scores = np.full(num_nodes, 1.0 / num_nodes)
    for iteration in range(1, max_iter + 1):
        flow = np.bincount(targets, weights=scores[sources] * edge_share, minlength=num_nodes)
        dangling_mass = scores[dangling].sum()
        new_scores = (1.0 - damping) / num_nodes + damping * (flow + dangling_mass / num_nodes)
        change = np.abs(new_scores - scores).sum()
        scores = new_scores
        if change < tol:
            return scores, iteration
    return scores, max_iter
