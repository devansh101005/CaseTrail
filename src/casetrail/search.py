"""CaseTrailEngine: one entry point for the demo (HLD C9; LLD section 9).

engine = CaseTrailEngine(split="test")
engine.search(text, system="bm25_bigram", k=10)     -> precedents + per-signal breakdown + matched passages
engine.predict_statutes(text, method="nb", top=5)   -> IPC sections with scores and dev-tuned thresholds
Everything is loaded once from indexes/, models/ and results/ (built by scripts/); nothing is downloaded.
Only signals whose artifacts exist are offered (see engine.available()).
"""
import json
import math
import time

import numpy as np

from casetrail import config
from casetrail.data import load_pcr, load_statutes
from casetrail.engine_parts import (dense_resources, lsi_resources, opening_text, passage_resources,
                                    statute_scores, title_of)
from casetrail.fusion.combine import weighted_sum
from casetrail.index.inverted_index import InvertedIndex
from casetrail.neural.encoder import maxsim_scores
from casetrail.retrieval.bm25 import bm25_idf, bm25_scores, query_weights, reduce_query, top_k
from casetrail.retrieval.pcr_lexical import doc_terms, index_folder
from casetrail.text.tokenizer import tokenize


class CaseTrailEngine:
    def __init__(self, split: str = "test"):
        with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
            self.params = json.load(f)
        self.split = split
        self.lexical = self.params["pcr"]["systems"]["bm25_bigram"]
        self.index = InvertedIndex().load(index_folder(split, 2, self.lexical["stem"]))
        self.pool = load_pcr(split)
        self.statutes = load_statutes()
        self.passages = passage_resources(split, self.params["pcr"]["systems"]["bm25_passages"])
        self.dense = dense_resources(split, self.params)
        self.lsi = lsi_resources(self.params)
        self.citation_pagerank = self.load_graph_scores("pcr_citation_pagerank")
        self.similarity_pagerank = self.load_graph_scores(f"pcr_{split}_similarity_pagerank")
        self.position = {name: i for i, name in enumerate(self.index.doc_names)}
        # Opening text of every pool judgment, to recognise a pasted benchmark judgment (demo examples)
        self.opening = {opening_text(doc["sentences"]): doc_id for doc_id, doc in self.pool["candidates"].items()}

    def load_graph_scores(self, name: str) -> dict:
        path = config.INDEX_DIR / "graph" / f"{name}.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def available(self) -> dict:
        systems = ["bm25_bigram"] + (["neural", "fusion"] if self.dense else [])
        methods = [m for m in ["nb", "rocchio", "knn"] if self.lsi]
        return {"systems": systems, "methods": methods,
                "loaded": {"index": True, "passages": bool(self.passages), "dense": bool(self.dense),
                           "statute_classifiers": bool(self.lsi), "graph": bool(self.citation_pagerank)}}

    def query_terms(self, text: str) -> tuple:
        """Index terms of the query and the reduced BM25 query (term -> weight)."""
        items = [tokenize(sentence, use_stemming=self.lexical["stem"]) for sentence in text.split("\n")]
        terms = doc_terms(items, self.lexical["ngram"])
        query = query_weights(terms, binary=self.lexical["binary_query"])
        return terms, reduce_query(query, self.index, self.lexical["top_terms"])

    def search(self, text: str, system: str = "bm25_bigram", k: int = config.FINAL_K) -> dict:
        if system not in self.available()["systems"]:
            raise ValueError(f"system must be one of {self.available()['systems']}, got {system!r}")
        timings = {}
        start = time.perf_counter()
        terms, query = self.query_terms(text)
        timings["tokenise"] = (time.perf_counter() - start) * 1000
        known_id = self.opening.get(opening_text([text]))
        exclude = {self.position[known_id]} if known_id else set()
        cited = set(self.pool["queries"][known_id]["relevant"]) if known_id in self.pool["queries"] else None
        bm25 = bm25_scores(self.index, query, self.lexical["k1"], self.lexical["b"])
        bm25_run = top_k(bm25, config.CANDIDATES_PER_SIGNAL, exclude)
        timings["bm25"] = (time.perf_counter() - start) * 1000 - timings["tokenise"]
        neural_scores = self.neural_scores(text, timings) if system in ["neural", "fusion"] else None
        ranked, fused = self.final_ranking(system, bm25_run, neural_scores, exclude, k)
        results = []
        for rank, doc_id in enumerate(ranked, start=1):
            entry = self.result_entry(rank, doc_id, query, bm25, neural_scores, fused)
            entry["is_relevant_in_benchmark"] = None if cited is None else doc_id in cited
            results.append(entry)
        timings["total"] = (time.perf_counter() - start) * 1000
        return {"system": system, "known_query_id": known_id, "query_terms": self.explain_terms(query), "results": results,
                "num_query_terms": len(set(terms)), "timings_ms": {key: round(v, 1) for key, v in timings.items()}}

    def neural_scores(self, text: str, timings: dict) -> np.ndarray:
        start = time.perf_counter()
        query_vectors = self.dense["encode"](text.split("\n"))
        scores = maxsim_scores(query_vectors, self.dense["vectors"], self.dense["offsets"])
        timings["neural"] = (time.perf_counter() - start) * 1000
        return scores

    def final_ranking(self, system: str, bm25_run: list, neural_scores, exclude: set, k: int) -> tuple:
        names = self.index.doc_names
        if system == "bm25_bigram":
            return [names[i] for i, score in bm25_run[:k]], {}
        neural_run = top_k(neural_scores, config.CANDIDATES_PER_SIGNAL, exclude)
        if system == "neural":
            return [names[i] for i, score in neural_run[:k]], {}
        weight = self.dense["neural_weight"]
        fused = weighted_sum([{"q": [(names[i], s) for i, s in bm25_run]}, {"q": [(names[i], s) for i, s in neural_run]}],
                             [1 - weight, weight])["q"]
        return [doc_id for doc_id, score in fused[:k]], dict(fused)

    def explain_terms(self, query: dict, top: int = 12) -> list:
        num_docs = self.index.num_docs()
        rows = []
        for term in query:
            df = self.index.doc_freq(term)
            if df < 2:
                continue  # occurs only in the pasted judgment itself, so it cannot link to another judgment
            rows.append({"term": term.replace("__", " "), "df": df, "idf": round(bm25_idf(df, num_docs), 3),
                         "weight": query[term]})
        rows.sort(key=lambda row: -row["idf"] * row["weight"])
        return rows[:top]

    def result_entry(self, rank: int, doc_id: str, query: dict, bm25: np.ndarray, neural_scores, fused: dict) -> dict:
        position = self.position[doc_id]
        sentences = self.pool["candidates"][doc_id]["sentences"]
        breakdown = {"bm25": round(float(bm25[position]), 3),
                     "neural": None if neural_scores is None else round(float(neural_scores[position]), 4),
                     "fusion": round(fused[doc_id], 4) if doc_id in fused else None,
                     "pagerank_citation": self.citation_pagerank.get(doc_id),
                     "pagerank_similarity": self.similarity_pagerank.get(doc_id)}
        return {"rank": rank, "doc_id": doc_id, "title": title_of(sentences),
                "snippet": " ".join(sentences)[:320], "matched_passages": self.matched_passages(doc_id, query),
                "breakdown": breakdown}

    def matched_passages(self, doc_id: str, query: dict, top: int = 2) -> list:
        if not self.passages:
            return []
        unigram_query = {term: weight for term, weight in query.items() if "__" not in term}
        scores = bm25_scores(self.passages["index"], unigram_query, self.passages["k1"], self.passages["b"])
        doc_position = self.position[doc_id]
        ids = np.flatnonzero(self.passages["passage_doc"] == doc_position)
        best = ids[np.argsort(-scores[ids])[:top]]
        sentences = self.pool["candidates"][doc_id]["sentences"]
        found = []
        for passage_id in best:
            if scores[passage_id] > 0:
                start, end = self.passages["spans"][passage_id]
                found.append(" ".join(sentences[start:end])[:600])
        return found

    def predict_statutes(self, text: str, method: str = "nb", top: int = 5) -> dict:
        if method not in self.available()["methods"]:
            raise ValueError(f"method must be one of {self.available()['methods']}, got {method!r}")
        scores, thresholds = statute_scores(self.lsi, tokenize(text, use_stemming=self.lsi["stemming"]), method)
        order = np.argsort(-(scores - thresholds))[:top]
        rows = []
        for c in order:
            statute_id = self.lsi["statutes"][c]
            probability = 1 / (1 + math.exp(-scores[c])) if method == "nb" else None
            rows.append({"id": statute_id, "title": self.statutes[statute_id]["title"], "score": round(float(scores[c]), 4),
                         "threshold": round(float(thresholds[c]), 4), "probability": probability,
                         "above_threshold": bool(scores[c] >= thresholds[c])})
        return {"method": method, "statutes": rows}
