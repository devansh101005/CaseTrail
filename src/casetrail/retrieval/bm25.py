"""BM25 and tf-idf cosine scoring over an inverted index (Lectures: tf-idf; Scoring and result assembly).

Scoring is term-at-a-time: for every query term we walk its postings list and add its contribution
into one score accumulator per document, then take the top K with a heap.
"""
import heapq
import math
from collections import Counter

import numpy as np


def bm25_idf(df: int, num_docs: int) -> float:
    # Robertson-Sparck Jones idf with +1 inside the log so it is never negative (Lucene form)
    return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))


def query_weights(query_terms: list, binary: bool = False) -> dict:
    """Query term frequencies; binary=True counts every distinct term once (useful for very long queries)."""
    counts = Counter(query_terms)
    if binary:
        return {term: 1 for term in counts}
    return dict(counts)


def bm25_scores(index, query: dict, k1: float, b: float) -> np.ndarray:
    """score(d) = sum_t qtf_t * idf_t * tf_td (k1 + 1) / (tf_td + k1 (1 - b + b |d| / avgdl))."""
    num_docs = index.num_docs()
    scores = np.zeros(num_docs, dtype=np.float64)
    # Length normalisation part of the denominator, one value per document
    length_norm = k1 * (1.0 - b + b * index.doc_len / index.avg_len)
    for term, qtf in query.items():
        doc_ids, tfs = index.postings(term)
        if len(doc_ids) == 0:
            continue
        idf = bm25_idf(len(doc_ids), num_docs)
        scores[doc_ids] += qtf * idf * tfs * (k1 + 1.0) / (tfs + length_norm[doc_ids])
    return scores


def tfidf_doc_norms(index) -> np.ndarray:
    """Euclidean length of every document's tf-idf vector, w_td = (1 + log tf) * log(N / df)."""
    num_docs = index.num_docs()
    idf = np.log(num_docs / np.maximum(index.df, 1))
    # Postings are stored term by term, so repeating each idf df times lines it up with the postings arrays
    posting_idf = np.repeat(idf, index.df)
    weights = (1.0 + np.log(index.tfs)) * posting_idf
    return np.sqrt(np.bincount(index.doc_ids, weights=weights ** 2, minlength=num_docs))


def tfidf_cosine_scores(index, query: dict, doc_norms: np.ndarray) -> np.ndarray:
    """Cosine similarity between the query and every document in the vector space model."""
    num_docs = index.num_docs()
    scores = np.zeros(num_docs, dtype=np.float64)
    query_norm_sq = 0.0
    for term, qtf in query.items():
        doc_ids, tfs = index.postings(term)
        if len(doc_ids) == 0:
            continue
        idf = math.log(num_docs / len(doc_ids))
        query_weight = (1.0 + math.log(qtf)) * idf
        query_norm_sq += query_weight ** 2
        scores[doc_ids] += query_weight * (1.0 + np.log(tfs)) * idf
    if query_norm_sq == 0.0:
        return scores
    return scores / (math.sqrt(query_norm_sq) * np.maximum(doc_norms, 1e-12))


def reduce_query(query: dict, index, top_terms: int) -> dict:
    """Query reduction: keep the top_terms query terms by qtf * idf (0 = keep all terms)."""
    if top_terms <= 0:
        return query
    num_docs = index.num_docs()
    weighted = []
    for term, qtf in query.items():
        df = index.doc_freq(term)
        if df > 0:
            weighted.append((qtf * bm25_idf(df, num_docs), term))
    best = heapq.nlargest(top_terms, weighted)
    return {term: query[term] for weight, term in best}


def top_k(scores: np.ndarray, k: int, exclude: set = None) -> list:
    """Heap-based top-K: [(doc_id, score)] in decreasing score order, skipping excluded doc ids."""
    exclude = exclude or set()
    best = heapq.nlargest(k + len(exclude), range(len(scores)), key=scores.__getitem__)
    ranked = [(doc_id, float(scores[doc_id])) for doc_id in best if doc_id not in exclude]
    return ranked[:k]
