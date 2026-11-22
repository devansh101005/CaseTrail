"""Lexical PCR systems on IL-TUR pools: tokens cache, index building and ranked runs (LLD section 5).

A "system" is a plain dict of settings, e.g.
  {"model": "bm25", "ngram": 1, "stem": False, "top_terms": 50, "binary_query": False,
   "k1": 1.2, "b": 0.75, "passages": None}
passages = None scores whole judgments; "max" / "top3" scores passages and aggregates them per judgment.
"""
import json
import pickle
from collections import Counter

import numpy as np

from casetrail import config
from casetrail.data import load_pcr
from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import (bm25_scores, query_weights, reduce_query, tfidf_cosine_scores, tfidf_doc_norms,
                                      top_k)
from casetrail.retrieval.chunking import build_passage_index, passage_doc_scores
from casetrail.text.tokenizer import add_bigrams, tokenize

RUN_DEPTH = 100          # candidates kept per query in a run (enough for MAP@100 and every K we try)
BIGRAM_MIN_DF = 2        # a bigram in only one judgment can never link two judgments of the pool


def tokens_cache_path(split: str, stem: bool):
    return config.DATA_DIR / "cache" / f"pcr_{split}_tokens_{'stem' if stem else 'nostem'}.pkl"


def load_item_tokens(split: str, stem: bool, pool: dict = None) -> dict:
    """{doc_id: [token list per text item]} for every candidate of the split (queries are candidates too)."""
    path = tokens_cache_path(split, stem)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    if pool is None:
        pool = load_pcr(split)["candidates"]
    print(f"  tokenising {len(pool)} {split} judgments (stem={stem}) ...")
    item_tokens = {}
    for doc_id, doc in pool.items():
        item_tokens[doc_id] = [tokenize(sentence, use_stemming=stem) for sentence in doc["sentences"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(item_tokens, f)
    return item_tokens


def doc_terms(items: list, ngram: int) -> list:
    words = []
    for item in items:
        words.extend(item)
    return words if ngram == 1 else add_bigrams(words)


def prune_rare_bigrams(term_lists: list, min_df: int) -> list:
    df = Counter()
    for terms in term_lists:
        df.update({term for term in terms if "__" in term})
    return [[term for term in terms if "__" not in term or df[term] >= min_df] for terms in term_lists]


def index_folder(split: str, ngram: int, stem: bool, passages: bool = False):
    name = f"pcr_{split}_{'bi' if ngram == 2 else 'uni'}{'_stem' if stem else ''}{'_passages' if passages else ''}"
    return config.INDEX_DIR / name


def build_doc_index(split: str, ngram: int, stem: bool, item_tokens: dict) -> InvertedIndex:
    names = sorted(item_tokens)
    term_lists = [doc_terms(item_tokens[name], ngram) for name in names]
    if ngram == 2:
        term_lists = prune_rare_bigrams(term_lists, BIGRAM_MIN_DF)
    index = InvertedIndex().build(names, term_lists)
    index.save(index_folder(split, ngram, stem))
    return index


def build_passages(split: str, ngram: int, stem: bool, item_tokens: dict, pool: dict) -> InvertedIndex:
    names = sorted(item_tokens)
    sentences = [pool[name]["sentences"] for name in names]
    if ngram == 1:
        item_terms = [item_tokens[name] for name in names]
    else:
        item_terms = [[add_bigrams(item) for item in item_tokens[name]] for name in names]
    index, passage_doc, spans = build_passage_index(names, sentences, item_terms)
    folder = index_folder(split, ngram, stem, passages=True)
    index.save(folder)
    np.save(folder / "passage_doc.npy", passage_doc)
    with open(folder / "spans.json", "w", encoding="utf-8") as f:
        json.dump(spans, f)
    return index


def load_resources(split: str, system: dict, cache: dict) -> dict:
    """Load (once) the index files a system needs; `cache` is a plain dict kept by the caller."""
    passages = system.get("passages") is not None
    folder = index_folder(split, system["ngram"], system["stem"], passages=passages)
    if folder not in cache:
        resources = {"index": InvertedIndex().load(folder)}
        if passages:
            resources["passage_doc"] = np.load(folder / "passage_doc.npy")
            resources["doc_index"] = InvertedIndex().load(index_folder(split, system["ngram"], system["stem"]))
        cache[folder] = resources
    resources = cache[folder]
    if system["model"] == "tfidf" and "norms" not in resources:
        resources["norms"] = tfidf_doc_norms(resources["index"])
    return resources


def score_query(query_terms: list, system: dict, resources: dict) -> np.ndarray:
    """Scores of every judgment in the pool for one query (list of index terms)."""
    index = resources["index"]
    query = query_weights(query_terms, binary=system.get("binary_query", False))
    query = reduce_query(query, index, system.get("top_terms", 0))
    if system["model"] == "tfidf":
        return tfidf_cosine_scores(index, query, resources["norms"])
    if system.get("passages") is None:
        return bm25_scores(index, query, system["k1"], system["b"])
    num_docs = resources["doc_index"].num_docs()
    doc_scores, _ = passage_doc_scores(index, resources["passage_doc"], num_docs, query,
                                       system["k1"], system["b"], system["passages"])
    return doc_scores


def rank_split(split: str, system: dict, item_tokens: dict, queries, cache: dict, with_scores: bool = False,
               depth: int = RUN_DEPTH) -> dict:
    """Run a system on every query of a split: {query_id: [candidate ids, best first]} (the query itself skipped).

    with_scores=True gives [(candidate id, score)] instead of plain ids.
    """
    resources = load_resources(split, system, cache)
    doc_names = resources["doc_index"].doc_names if "doc_index" in resources else resources["index"].doc_names
    position = {name: i for i, name in enumerate(doc_names)}
    run = {}
    for query_id in queries:
        scores = score_query(doc_terms(item_tokens[query_id], system["ngram"]), system, resources)
        ranked = top_k(scores, depth, exclude={position[query_id]})
        if with_scores:
            run[query_id] = [(doc_names[doc_id], score) for doc_id, score in ranked]
        else:
            run[query_id] = [doc_names[doc_id] for doc_id, score in ranked]
    return run


def qrels_of(queries: dict) -> dict:
    return {query_id: set(query["relevant"]) for query_id, query in queries.items()}
