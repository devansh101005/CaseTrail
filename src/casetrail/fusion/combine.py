"""Score fusion of several rankers (keyword BM25 + neural meaning; LLD/HLD component C7).

Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank_r(d)).
It only uses ranks, so it needs no score normalisation. Weighted sum: min-max normalise each ranker's
scores per query to [0, 1] and add them with weights (explainable: "40% keyword, 60% meaning").
A run with scores is {query_id: [(doc_id, score), ...]} best first.
"""


def rrf(runs: list, k: int = 60) -> dict:
    fused = {}
    for query_id in runs[0]:
        totals = {}
        for run in runs:
            for rank, (doc_id, score) in enumerate(run.get(query_id, []), start=1):
                totals[doc_id] = totals.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[query_id] = sorted(totals.items(), key=lambda item: -item[1])
    return fused


def min_max(ranked: list) -> dict:
    if not ranked:
        return {}
    scores = [score for doc_id, score in ranked]
    low, high = min(scores), max(scores)
    spread = high - low if high > low else 1.0
    return {doc_id: (score - low) / spread for doc_id, score in ranked}


def weighted_sum(runs: list, weights: list) -> dict:
    """A document missing from one ranker's list gets 0 from that ranker."""
    fused = {}
    for query_id in runs[0]:
        totals = {}
        for run, weight in zip(runs, weights):
            for doc_id, value in min_max(run.get(query_id, [])).items():
                totals[doc_id] = totals.get(doc_id, 0.0) + weight * value
        fused[query_id] = sorted(totals.items(), key=lambda item: -item[1])
    return fused


def ids_only(run: dict) -> dict:
    return {query_id: [doc_id for doc_id, score in ranked] for query_id, ranked in run.items()}
