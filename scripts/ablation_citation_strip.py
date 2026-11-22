"""Leakage ablation: the best lexical PCR system with and WITHOUT stripping leftover citations.

Same tuned settings (results/tuned_params.json); only the tokenizer's citation stripping is switched off.
The difference measures how much the unmasked citation strings (P1 finding, docs/DATA.md) would inflate scores.
Rows "bm25_bigram(citations_kept)" -> results/pcr_metrics.csv. Nothing is tuned here.
"""
import argparse
import json

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import pcr_run_metrics
from casetrail.eval.results_io import metric_rows, write_rows
from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import query_weights, reduce_query, bm25_scores, top_k
from casetrail.retrieval.pcr_lexical import BIGRAM_MIN_DF, RUN_DEPTH, doc_terms, prune_rare_bigrams, qrels_of
from casetrail.text.tokenizer import tokenize

BASE = "bm25_bigram"


def run_without_stripping(split: str, system: dict) -> tuple:
    pcr = load_pcr(split)
    names = sorted(pcr["candidates"])
    print(f"  {split}: tokenising {len(names)} judgments with citations kept ...")
    item_tokens = {}
    for name in names:
        sentences = pcr["candidates"][name]["sentences"]
        item_tokens[name] = [tokenize(s, use_stemming=system["stem"], keep_citations=True) for s in sentences]
    term_lists = prune_rare_bigrams([doc_terms(item_tokens[name], system["ngram"]) for name in names], BIGRAM_MIN_DF)
    index = InvertedIndex().build(names, term_lists)
    position = {name: i for i, name in enumerate(names)}
    run = {}
    for query_id in pcr["queries"]:
        query = query_weights(doc_terms(item_tokens[query_id], system["ngram"]), system["binary_query"])
        query = reduce_query(query, index, system["top_terms"])
        scores = bm25_scores(index, query, system["k1"], system["b"])
        run[query_id] = [names[i] for i, score in top_k(scores, RUN_DEPTH, exclude={position[query_id]})]
    return run, qrels_of(pcr["queries"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        system = json.load(f)["pcr"]["systems"][BASE]
    rows = []
    for split in ["dev", "test"]:
        run, qrels = run_without_stripping(split, system)
        metrics = pcr_run_metrics(run, qrels, system["k"])
        print(f"{split:5s} {BASE} with citations kept: micro-F1@K={metrics['micro_f1@k']:.4f}")
        rows.extend(metric_rows("pcr", f"{BASE}(citations_kept)", split, metrics, len(qrels)))
    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)


if __name__ == "__main__":
    main()
