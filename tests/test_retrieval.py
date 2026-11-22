"""Inverted index, BM25, tf-idf and passages on a tiny hand-made corpus."""
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import (bm25_idf, bm25_scores, query_weights, reduce_query, tfidf_cosine_scores,
                                      tfidf_doc_norms, top_k)
from casetrail.retrieval.chunking import passage_doc_scores, split_passages

DOCS = [
    ["dowry", "death", "bride", "dowry"],
    ["murder", "knife", "death"],
    ["bail", "granted"],
    ["dowry", "harassment", "cruelty", "bride", "husband"],
]


def toy_index():
    return InvertedIndex().build(["d0", "d1", "d2", "d3"], DOCS)


def test_postings_sorted_and_df_matches():
    index = toy_index()
    doc_ids, tfs = index.postings("dowry")
    assert list(doc_ids) == [0, 3] and list(tfs) == [2, 1]
    for term, term_id in index.term_to_id.items():
        start, end = index.offsets[term_id], index.offsets[term_id + 1]
        ids = index.doc_ids[start:end]
        assert index.df[term_id] == end - start
        assert list(ids) == sorted(ids)
    assert list(index.doc_len) == [4, 3, 2, 5] and index.avg_len == 3.5


def test_save_load_round_trip(tmp_path: Path):
    index = toy_index()
    index.save(tmp_path / "idx")
    loaded = InvertedIndex().load(tmp_path / "idx")
    assert loaded.doc_names == index.doc_names
    assert loaded.term_to_id == index.term_to_id
    assert np.array_equal(loaded.doc_ids, index.doc_ids) and np.array_equal(loaded.tfs, index.tfs)


def test_bm25_matches_hand_computation():
    index = toy_index()
    k1, b = 1.2, 0.75
    scores = bm25_scores(index, {"dowry": 1}, k1, b)
    # doc 0: tf = 2, |d| = 4, avgdl = 3.5, df = 2, N = 4
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 4 / 3.5))
    assert abs(scores[0] - expected) < 1e-12
    assert scores[1] == 0 and scores[2] == 0


def test_bm25_ranking_agrees_with_rank_bm25_reference():
    # rank_bm25 uses a different idf (log((N - df + 0.5) / (df + 0.5)), which is 0 when df = N/2), so we pad the
    # corpus with filler docs and compare rankings for queries whose terms share one df: then the idf difference
    # is a constant factor and cannot change the order.
    docs = DOCS + [["filler", "text"], ["other", "words"], ["more", "filler"], ["unrelated"]]
    index = InvertedIndex().build([f"d{i}" for i in range(len(docs))], docs)
    reference = BM25Okapi(docs, k1=1.2, b=0.75)
    for query in [["dowry"], ["death"], ["bride", "dowry"]]:
        ours = bm25_scores(index, query_weights(query), 1.2, 0.75)
        theirs = reference.get_scores(query)
        matched = [i for i in range(len(docs)) if ours[i] > 0]
        assert sorted(matched, key=lambda i: -ours[i]) == sorted(matched, key=lambda i: -theirs[i])


def test_bm25_idf_never_negative():
    assert bm25_idf(df=100, num_docs=100) > 0


def test_tfidf_cosine_of_document_with_itself_is_one():
    index = toy_index()
    norms = tfidf_doc_norms(index)
    scores = tfidf_cosine_scores(index, query_weights(DOCS[3]), norms)
    assert abs(scores[3] - 1.0) < 1e-9
    assert np.argmax(scores) == 3


def test_query_reduction_keeps_rare_terms():
    index = toy_index()
    query = {"dowry": 3, "husband": 1, "unknown": 5}
    reduced = reduce_query(query, index, top_terms=1)
    assert reduced == {"dowry": 3}  # qtf * idf: 3 * idf(df=2) beats 1 * idf(df=1); unknown terms dropped
    assert reduce_query(query, index, top_terms=0) == query


def test_top_k_excludes_query_itself():
    scores = np.array([0.5, 2.0, 1.0, 3.0])
    assert top_k(scores, 2) == [(3, 3.0), (1, 2.0)]
    assert top_k(scores, 2, exclude={3}) == [(1, 2.0), (2, 1.0)]


def test_split_passages_groups_until_enough_words():
    sentences = ["a " * 70, "b " * 70, "c " * 130, "d " * 10]
    assert split_passages(sentences, min_words=120) == [(0, 2), (2, 4)]


def test_passage_scores_max_and_top3():
    passage_index = InvertedIndex().build(["p0", "p1", "p2"], [["dowry"], ["dowry", "dowry"], ["bail"]])
    passage_doc = np.array([0, 0, 1])
    doc_max, passage_scores = passage_doc_scores(passage_index, passage_doc, 2, {"dowry": 1}, 1.2, 0.75, "max")
    assert doc_max[0] == passage_scores[:2].max() and doc_max[1] == 0
    doc_sum, _ = passage_doc_scores(passage_index, passage_doc, 2, {"dowry": 1}, 1.2, 0.75, "top3")
    assert abs(doc_sum[0] - passage_scores[:2].sum()) < 1e-12
