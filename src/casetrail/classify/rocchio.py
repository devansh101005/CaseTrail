"""Rocchio classification with statute centroids (Lecture: Vector space classification - Rocchio).

Every document is a length-normalised tf-idf vector, w_td = (1 + log tf) * log(N / df).
The centroid of statute c is the mean vector of its training documents; a document's score for c is
the cosine between the document and the centroid.
"""
import numpy as np

from casetrail.classify.naive_bayes import term_label_sums


def tfidf_vectors(vectors: list, idf: np.ndarray) -> list:
    """(term_ids, unit-length tf-idf weights) for every document."""
    weighted = []
    for term_ids, tfs in vectors:
        weights = (1.0 + np.log(tfs)) * idf[term_ids] if len(term_ids) else tfs
        norm = np.linalg.norm(weights)
        weighted.append((term_ids, weights / norm if norm > 0 else weights))
    return weighted


def train_rocchio(train_tfidf: list, labels: np.ndarray, num_terms: int) -> np.ndarray:
    """Unit-length centroids [num_terms x num_statutes]."""
    centroids = term_label_sums(train_tfidf, labels, num_terms) / np.maximum(labels.sum(axis=0), 1)
    return centroids / np.maximum(np.linalg.norm(centroids, axis=0), 1e-12)


def rocchio_scores(doc_tfidf: list, centroids: np.ndarray) -> np.ndarray:
    scores = np.zeros((len(doc_tfidf), centroids.shape[1]), dtype=np.float64)
    for d, (term_ids, weights) in enumerate(doc_tfidf):
        scores[d] = weights @ centroids[term_ids]
    return scores
