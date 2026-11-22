"""k-nearest-neighbour statute classification (Lecture: Vector space classification - kNN).

The neighbours of a document are the training documents with the highest tf-idf cosine, found with the
inverted index. A statute's score is the summed similarity of the neighbours labelled with it.
"""
import numpy as np

from casetrail.retrieval.bm25 import tfidf_cosine_scores


def nearest_neighbours(train_index, train_norms: np.ndarray, query: dict, depth: int) -> tuple:
    """(neighbour doc ids, cosine similarities), best first."""
    similarity = tfidf_cosine_scores(train_index, query, train_norms)
    depth = min(depth, len(similarity))
    candidates = np.argpartition(-similarity, depth - 1)[:depth]
    order = candidates[np.argsort(-similarity[candidates])]
    return order, similarity[order]


def knn_scores(neighbours: list, labels: np.ndarray, k: int) -> np.ndarray:
    """neighbours[d] = (ids, similarities) at some depth >= k; score[d, c] = sum of similarities of the
    top-k neighbours labelled c."""
    scores = np.zeros((len(neighbours), labels.shape[1]), dtype=np.float64)
    for d, (ids, sims) in enumerate(neighbours):
        scores[d] = sims[:k] @ labels[ids[:k]]
    return scores
