"""Inverted index: dictionary + postings with term frequencies (Lecture: Scoring and result assembly - inverted index).

Postings are stored in flat numpy arrays (LLD section 4):
  postings of term t  =  doc_ids[offsets[t]:offsets[t+1]]  with matching  tfs[...]
doc_ids are sorted inside each postings list. One index is built per candidate pool (train, dev, test).
"""
import json
from collections import Counter

import numpy as np


class InvertedIndex:
    def __init__(self):
        self.term_to_id = {}
        self.df = np.zeros(0, dtype=np.int32)
        self.offsets = np.zeros(1, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int32)
        self.tfs = np.zeros(0, dtype=np.int32)
        self.doc_len = np.zeros(0, dtype=np.int32)
        self.avg_len = 0.0
        self.doc_names = []

    def build(self, doc_names: list, doc_terms: list):
        """doc_terms[i] is the list of index terms of document doc_names[i]."""
        if len(doc_names) != len(doc_terms):
            raise ValueError("doc_names and doc_terms must have the same length")
        self.doc_names = list(doc_names)
        term_parts, doc_parts, tf_parts = [], [], []
        lengths = []
        for doc_id, terms in enumerate(doc_terms):
            counts = Counter(terms)
            ids = [self.term_to_id.setdefault(term, len(self.term_to_id)) for term in counts]
            term_parts.append(np.array(ids, dtype=np.int32))
            tf_parts.append(np.array(list(counts.values()), dtype=np.int32))
            doc_parts.append(np.full(len(ids), doc_id, dtype=np.int32))
            lengths.append(len(terms))
        term_col = np.concatenate(term_parts) if term_parts else np.zeros(0, dtype=np.int32)
        doc_col = np.concatenate(doc_parts) if doc_parts else np.zeros(0, dtype=np.int32)
        tf_col = np.concatenate(tf_parts) if tf_parts else np.zeros(0, dtype=np.int32)

        # Sort by (term, doc) so each term's postings are contiguous and doc-sorted
        order = np.lexsort((doc_col, term_col))
        self.doc_ids = doc_col[order]
        self.tfs = tf_col[order]
        vocab_size = len(self.term_to_id)
        self.df = np.bincount(term_col, minlength=vocab_size).astype(np.int32)
        self.offsets = np.zeros(vocab_size + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.df)
        self.doc_len = np.array(lengths, dtype=np.int32)
        self.avg_len = float(self.doc_len.mean()) if len(lengths) else 0.0
        return self

    def num_docs(self) -> int:
        return len(self.doc_names)

    def postings(self, term: str):
        """(doc_ids, tfs) of a term; empty arrays if the term is not in the dictionary."""
        term_id = self.term_to_id.get(term)
        if term_id is None:
            return self.doc_ids[:0], self.tfs[:0]
        start, end = self.offsets[term_id], self.offsets[term_id + 1]
        return self.doc_ids[start:end], self.tfs[start:end]

    def doc_freq(self, term: str) -> int:
        term_id = self.term_to_id.get(term)
        return 0 if term_id is None else int(self.df[term_id])

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["df", "offsets", "doc_ids", "tfs", "doc_len"]:
            np.save(folder / f"{name}.npy", getattr(self, name))
        terms = [""] * len(self.term_to_id)
        for term, term_id in self.term_to_id.items():
            terms[term_id] = term
        with open(folder / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"terms": terms, "doc_names": self.doc_names, "avg_len": self.avg_len}, f)

    def load(self, folder):
        if not (folder / "meta.json").exists():
            raise FileNotFoundError(f"No index in {folder}. Run `python scripts/build_index.py` first.")
        for name in ["df", "offsets", "doc_ids", "tfs", "doc_len"]:
            setattr(self, name, np.load(folder / f"{name}.npy"))
        with open(folder / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.term_to_id = {term: term_id for term_id, term in enumerate(meta["terms"])}
        self.doc_names = meta["doc_names"]
        self.avg_len = meta["avg_len"]
        return self
