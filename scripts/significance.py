"""Paired randomisation tests for the headline comparisons on TEST -> results/significance.csv.

PCR: micro-F1@K (each system at its own dev-chosen K; 10,000 permutations) and MAP@100 (per-query AP).
LSI: macro-F1 (1,000 permutations, document-level swaps), thresholds = the dev-tuned ones.
Pairs whose run files are missing (e.g. Kaggle outputs not back yet) are skipped and listed.
"""
import argparse
import csv
import json

import numpy as np

from casetrail import config
from casetrail.classify.lsi_data import statute_list
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.data import load_lsi, load_pcr
from casetrail.eval.metrics import average_precision
from casetrail.eval.significance import (micro_f1_counts, randomisation_test_macro_f1, randomisation_test_mean,
                                         randomisation_test_micro_f1)
from casetrail.retrieval.pcr_lexical import qrels_of

RUNS_DIR = config.INDEX_DIR / "runs"
PCR_PAIRS = [
    ("bm25", "bm25_default"),              # does dev tuning of k1, b, stemming help?
    ("bm25", "tfidf"),                     # BM25 vs the vector space model
    ("bm25_bigram", "bm25"),               # do bigrams help?
    ("bm25_bigram", "bm25_passages"),
    ("bm25_bigram+citation_pagerank", "bm25_bigram"),
    ("fusion_weighted_minilm", "bm25_bigram"),     # does the (zero-shot) neural signal add to BM25?
    ("fusion_weighted_minilm_int8", "bm25_bigram"),
    ("dense_minilm", "dense_minilm_int8"),          # what does INT8 quantisation cost?
    ("fusion_weighted_student_distilled", "bm25_bigram"),
    ("bm25_bigram+teacher_rerank", "bm25_bigram"),
]
LSI_PAIRS = [("nb", "rocchio"), ("knn", "nb"), ("knn", "rocchio")]


def k_of(system: str, params: dict) -> int:
    """The dev-chosen K that each evaluation script recorded for its system."""
    return params["pcr"]["k_by_system"][system]


def load_run(system: str):
    path = RUNS_DIR / f"pcr_test_{system}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pcr_rows(params: dict, skipped: list) -> list:
    qrels = qrels_of(load_pcr("test")["queries"])
    rows = []
    for system_a, system_b in PCR_PAIRS:
        run_a, run_b = load_run(system_a), load_run(system_b)
        if run_a is None or run_b is None:
            skipped.append(f"pcr {system_a} vs {system_b}")
            continue
        micro = randomisation_test_micro_f1(micro_f1_counts(run_a, qrels, k_of(system_a, params)),
                                            micro_f1_counts(run_b, qrels, k_of(system_b, params)))
        ap_a = [average_precision(run_a.get(q, []), rel) for q, rel in qrels.items()]
        ap_b = [average_precision(run_b.get(q, []), rel) for q, rel in qrels.items()]
        mean_ap = randomisation_test_mean(ap_a, ap_b)
        rows.append(["pcr", system_a, system_b, "test", "micro_f1@k", micro["difference"], micro["p_value"], len(qrels)])
        rows.append(["pcr", system_a, system_b, "test", "map@100", mean_ap["difference"], mean_ap["p_value"], len(qrels)])
        print(f"{system_a} vs {system_b}: dF1 {micro['difference']:+.4f} (p={micro['p_value']:.4f}), "
              f"dMAP {mean_ap['difference']:+.4f} (p={mean_ap['p_value']:.4f})")
    return rows


def lsi_predictions(method: str, statutes: list, dev_labels: np.ndarray):
    dev_path, test_path = RUNS_DIR / f"lsi_{method}_dev.npy", RUNS_DIR / f"lsi_{method}_test.npy"
    if not dev_path.exists() or not test_path.exists():
        return None
    thresholds = tune_thresholds(np.load(dev_path), dev_labels)
    return predict_sets(np.load(test_path), thresholds, statutes)


def lsi_rows(skipped: list) -> list:
    statutes = statute_list()
    column = {statute: c for c, statute in enumerate(statutes)}
    dev_docs, test_docs = load_lsi("dev"), load_lsi("test")
    dev_labels = np.zeros((len(dev_docs), len(statutes)))
    for d, doc in enumerate(dev_docs):
        for statute in doc["statutes"]:
            dev_labels[d, column[statute]] = 1.0
    gold = [set(doc["statutes"]) for doc in test_docs]
    rows = []
    for method_a, method_b in LSI_PAIRS:
        pred_a, pred_b = lsi_predictions(method_a, statutes, dev_labels), lsi_predictions(method_b, statutes, dev_labels)
        if pred_a is None or pred_b is None:
            skipped.append(f"lsi {method_a} vs {method_b}")
            continue
        result = randomisation_test_macro_f1(pred_a, pred_b, gold, statutes)
        rows.append(["lsi", method_a, method_b, "test", "macro_f1", result["difference"], result["p_value"], len(gold)])
        print(f"{method_a} vs {method_b}: dMacroF1 {result['difference']:+.4f} (p={result['p_value']:.4f})")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        params = json.load(f)
    skipped = []
    rows = pcr_rows(params, skipped) + lsi_rows(skipped)
    with open(config.RESULTS_DIR / "significance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "system_a", "system_b", "split", "metric", "difference", "p_value", "n"])
        for row in rows:
            writer.writerow(row[:5] + [round(row[5], 4), round(row[6], 4), row[7]])
    print(f"Wrote results/significance.csv ({len(rows)} rows); skipped (runs missing): {skipped}")


if __name__ == "__main__":
    main()
