"""Long-judgment handling: passage-level BM25 (Lecture: Scoring and result assembly; LLD section 5).

A judgment is split into passages of consecutive paragraphs (at least PASSAGE_WORDS words each).
Passages are indexed like small documents; a judgment's score is the max (or the sum of the top 3)
of its passage scores, so one strongly matching passage is enough to surface a long judgment.
The same passages are used by the neural encoder later (chunks) and as "matched passages" in the demo.
"""
import numpy as np

from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import bm25_scores

PASSAGE_WORDS = 120


def split_passages(sentences: list, min_words: int = PASSAGE_WORDS) -> list:
    """Group consecutive text items into passages; returns [(first_item, last_item_exclusive)]."""
    passages = []
    start = 0
    words = 0
    for i, sentence in enumerate(sentences):
        words += len(sentence.split())
        if words >= min_words:
            passages.append((start, i + 1))
            start = i + 1
            words = 0
    if start < len(sentences):
        # A short tail joins the previous passage instead of forming a tiny one
        if passages and words < min_words // 2:
            passages[-1] = (passages[-1][0], len(sentences))
        else:
            passages.append((start, len(sentences)))
    return passages


def build_passage_index(doc_names: list, doc_sentences: list, doc_item_terms: list) -> tuple:
    """Index every passage. doc_item_terms[d][i] = index terms of text item i of document d.

    Returns (index, passage_doc array, passage spans [(first_item, last_item_exclusive)]).
    """
    names, terms, passage_doc, spans = [], [], [], []
    for doc_id, sentences in enumerate(doc_sentences):
        item_terms = doc_item_terms[doc_id]
        for start, end in split_passages(sentences):
            passage_terms = []
            for i in range(start, end):
                passage_terms.extend(item_terms[i])
            names.append(f"{doc_names[doc_id]}#{start}")
            terms.append(passage_terms)
            passage_doc.append(doc_id)
            spans.append((start, end))
    index = InvertedIndex().build(names, terms)
    return index, np.array(passage_doc, dtype=np.int32), spans


def passage_doc_scores(passage_index, passage_doc, num_docs: int, query: dict, k1: float, b: float,
                       aggregate: str = "max") -> tuple:
    """Judgment scores from passage scores. Returns (doc_scores, passage_scores)."""
    passage_scores = bm25_scores(passage_index, query, k1, b)
    doc_scores = np.zeros(num_docs, dtype=np.float64)
    if aggregate == "max":
        np.maximum.at(doc_scores, passage_doc, passage_scores)
        return doc_scores, passage_scores
    if aggregate != "top3":
        raise ValueError(f"aggregate must be 'max' or 'top3', got {aggregate!r}")
    # Passages of a document are contiguous (built in document order), so walk them doc by doc
    boundaries = np.flatnonzero(np.diff(passage_doc)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(passage_doc)]))
    for start, end in zip(starts, ends):
        best = np.sort(passage_scores[start:end])[-3:]
        doc_scores[passage_doc[start]] = best.sum()
    return doc_scores, passage_scores
