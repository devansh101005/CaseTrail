"""HITS hubs and authorities (Lecture: Link analysis - HITS).

A good hub points to good authorities, and a good authority is pointed to by good hubs:
  a = A^T h,  h = A a,  then L2-normalise both, repeated until the vectors stop changing.
On our bipartite graph the hubs are judgments and the authorities are statutes (A[j, s] = edge weight).
"""
import numpy as np


def hits(num_hubs: int, num_authorities: int, edges: list, tol: float = 1e-8, max_iter: int = 100) -> tuple:
    """edges = [(hub, authority, weight)]. Returns (hub scores, authority scores, iterations); both have L2 norm 1."""
    hub_ids = np.array([edge[0] for edge in edges], dtype=np.int64)
    auth_ids = np.array([edge[1] for edge in edges], dtype=np.int64)
    weights = np.array([edge[2] for edge in edges], dtype=np.float64)
    hub = np.ones(num_hubs) / np.sqrt(max(num_hubs, 1))
    authority = np.ones(num_authorities) / np.sqrt(max(num_authorities, 1))
    for iteration in range(1, max_iter + 1):
        # a = A^T h : each authority collects the scores of the hubs pointing to it
        new_authority = np.bincount(auth_ids, weights=weights * hub[hub_ids], minlength=num_authorities)
        new_authority /= max(np.linalg.norm(new_authority), 1e-300)
        # h = A a : each hub collects the scores of the authorities it points to
        new_hub = np.bincount(hub_ids, weights=weights * new_authority[auth_ids], minlength=num_hubs)
        new_hub /= max(np.linalg.norm(new_hub), 1e-300)
        change = np.abs(new_hub - hub).sum() + np.abs(new_authority - authority).sum()
        hub, authority = new_hub, new_authority
        if change < tol:
            return hub, authority, iteration
    return hub, authority, max_iter
