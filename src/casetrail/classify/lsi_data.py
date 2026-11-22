"""Shared LSI inputs for the statute classifiers: token cache, training index, label matrix, document vectors.

A document vector is a pair (term_ids, tfs) of numpy arrays over the training vocabulary.
The vocabulary is the top VOCAB_SIZE training terms by document frequency (LLD section 7).
"""
import pickle
from collections import Counter

import numpy as np

from casetrail import config
from casetrail.data import load_lsi, load_statutes
from casetrail.index.inverted_index import InvertedIndex
from casetrail.text.tokenizer import tokenize

VOCAB_SIZE = 50000


def lsi_tokens(split: str, stem: bool) -> list:
    """[(doc_id, tokens, statutes)] for an LSI split, tokenised once and cached in data/cache/."""
    path = config.DATA_DIR / "cache" / f"lsi_{split}_tokens_{'stem' if stem else 'nostem'}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    docs = load_lsi(split)
    print(f"  tokenising {len(docs)} LSI {split} documents (stem={stem}) ...")
    rows = [(doc["id"], tokenize(" ".join(doc["sentences"]), use_stemming=stem), doc["statutes"]) for doc in docs]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rows, f)
    return rows


def build_train_index(train_rows: list) -> InvertedIndex:
    df = Counter()
    for doc_id, tokens, statutes in train_rows:
        df.update(set(tokens))
    vocabulary = {term for term, count in df.most_common(VOCAB_SIZE)}
    names = [doc_id for doc_id, tokens, statutes in train_rows]
    term_lists = [[token for token in tokens if token in vocabulary] for doc_id, tokens, statutes in train_rows]
    return InvertedIndex().build(names, term_lists)


def statute_list() -> list:
    return list(load_statutes())


def label_matrix(rows: list, statutes: list) -> np.ndarray:
    """Y[d, c] = 1 if document d is labelled with statute c."""
    column = {statute: c for c, statute in enumerate(statutes)}
    labels = np.zeros((len(rows), len(statutes)), dtype=np.float64)
    for d, (doc_id, tokens, doc_statutes) in enumerate(rows):
        for statute in doc_statutes:
            labels[d, column[statute]] = 1.0
    return labels


def doc_vectors(rows: list, index: InvertedIndex) -> list:
    """(term_ids, tfs) per document over the training vocabulary; unknown terms are dropped."""
    vectors = []
    for doc_id, tokens, statutes in rows:
        counts = Counter(token for token in tokens if token in index.term_to_id)
        term_ids = np.array([index.term_to_id[term] for term in counts], dtype=np.int64)
        tfs = np.array(list(counts.values()), dtype=np.float64)
        vectors.append((term_ids, tfs))
    return vectors
