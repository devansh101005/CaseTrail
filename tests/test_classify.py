"""Naive Bayes, Rocchio, kNN and threshold tuning on tiny hand-made datasets."""
import math

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from casetrail.classify.knn import knn_scores, nearest_neighbours
from casetrail.classify.naive_bayes import naive_bayes_scores, train_naive_bayes
from casetrail.classify.rocchio import rocchio_scores, tfidf_vectors, train_rocchio
from casetrail.classify.thresholds import best_threshold, predict_sets
from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import tfidf_doc_norms

# Vocabulary: 0 dowry, 1 death, 2 knife, 3 theft. Statutes: 0 = 304B (dowry death), 1 = 379 (theft)
TRAIN = [(np.array([0, 1]), np.array([2.0, 1.0])), (np.array([0]), np.array([3.0])),
         (np.array([2, 1]), np.array([1.0, 1.0])), (np.array([3]), np.array([2.0]))]
LABELS = np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.float64)


def dense(vectors: list, num_terms: int) -> np.ndarray:
    matrix = np.zeros((len(vectors), num_terms))
    for d, (ids, values) in enumerate(vectors):
        matrix[d, ids] = values
    return matrix


def test_naive_bayes_log_odds_match_sklearn():
    model = train_naive_bayes(TRAIN, LABELS, 4, alpha=1.0)
    test = [(np.array([0, 1]), np.array([1.0, 1.0])), (np.array([3]), np.array([1.0]))]
    ours = naive_bayes_scores(test, model)
    for c in range(2):
        reference = MultinomialNB(alpha=1.0).fit(dense(TRAIN, 4), LABELS[:, c])
        log_proba = reference.predict_log_proba(dense(test, 4))
        assert np.allclose(ours[:, c], log_proba[:, 1] - log_proba[:, 0])
    assert ours[0, 0] > 0 > ours[1, 0]   # the dowry document is 304B, the theft one is not


def test_naive_bayes_smoothing_by_hand():
    model = train_naive_bayes(TRAIN, LABELS, 4, alpha=1.0)
    # count(dowry, 304B) = 5, total tokens in 304B docs = 6, V = 4 -> theta = 6 / 10
    # count(dowry, not 304B) = 0, tokens = 4                      -> theta = 1 / 8
    assert abs(model["weights"][0, 0] - (math.log(6 / 10) - math.log(1 / 8))) < 1e-12


def test_rocchio_prefers_nearest_centroid():
    idf = np.log(4 / np.array([2.0, 2.0, 1.0, 1.0]))
    centroids = train_rocchio(tfidf_vectors(TRAIN, idf), LABELS, 4)
    scores = rocchio_scores(tfidf_vectors([(np.array([0]), np.array([1.0]))], idf), centroids)
    assert scores[0, 0] > scores[0, 1]


def test_knn_votes_by_similarity():
    index = InvertedIndex().build(["a", "b", "c", "d"], [["dowry", "dowry", "death"], ["dowry"], ["knife", "death"],
                                                         ["theft", "theft"]])
    labels = np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=np.float64)
    ids, sims = nearest_neighbours(index, tfidf_doc_norms(index), {"theft": 1}, depth=2)
    assert ids[0] == 3 and abs(sims[0] - 1.0) < 1e-9
    scores = knn_scores([(ids, sims)], labels, k=1)
    assert scores[0, 1] > 0 and scores[0, 0] == 0


def test_best_threshold_maximises_f1():
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    gold = np.array([1, 1, 0, 1, 0])
    threshold = best_threshold(scores, gold)
    # cut after 0.3 -> TP 3, predicted 4 -> F1 = 6/7, better than cutting after 0.8 (F1 = 0.8)
    assert 0.1 < threshold <= 0.3
    assert predict_sets(np.array([[0.35, 0.05]]), np.array([threshold, 0.2]), ["s1", "s2"]) == [{"s1"}]
