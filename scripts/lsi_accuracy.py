"""Accuracy measures for statute identification (multi-label) -> extra rows in results/lsi_metrics.csv.

The report format asks for accuracy next to precision / recall / F1. For a multi-label task "accuracy" has
three common forms, all reported here:
  subset accuracy  - share of documents whose predicted statute set is exactly right (very strict);
  sample accuracy  - mean over documents of |pred ∩ gold| / |pred ∪ gold| (Jaccard per document);
  hamming accuracy - share of all (document, statute) yes/no decisions that are right (dominated by easy "no"s).
Uses the saved score matrices and the per-statute thresholds tuned on dev; nothing is tuned here.
"""
import argparse

import numpy as np

from casetrail import config
from casetrail.classify.lsi_data import statute_list
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.data import load_lsi
from casetrail.eval.results_io import metric_rows, write_rows

RUNS_DIR = config.INDEX_DIR / "runs"


def accuracies(predicted: list, gold: list, num_labels: int) -> dict:
    exact = 0
    jaccard = 0.0
    wrong_decisions = 0
    for pred, true in zip(predicted, gold):
        exact += int(pred == true)
        union = pred | true
        jaccard += len(pred & true) / len(union) if union else 1.0
        wrong_decisions += len(pred ^ true)
    n = len(gold)
    return {"subset_accuracy": exact / n, "sample_accuracy": jaccard / n,
            "hamming_accuracy": 1.0 - wrong_decisions / (n * num_labels)}


def label_matrix(docs: list, statutes: list) -> np.ndarray:
    column = {statute: c for c, statute in enumerate(statutes)}
    labels = np.zeros((len(docs), len(statutes)))
    for d, doc in enumerate(docs):
        for statute in doc["statutes"]:
            labels[d, column[statute]] = 1.0
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    statutes = statute_list()
    docs = {split: load_lsi(split) for split in ["dev", "test"]}
    dev_labels = label_matrix(docs["dev"], statutes)
    rows = []
    for method in ["nb", "rocchio", "knn"]:
        dev_scores = np.load(RUNS_DIR / f"lsi_{method}_dev.npy")
        thresholds = tune_thresholds(dev_scores, dev_labels)
        for split in ["dev", "test"]:
            scores = dev_scores if split == "dev" else np.load(RUNS_DIR / f"lsi_{method}_test.npy")
            predicted = predict_sets(scores, thresholds, statutes)
            gold = [set(doc["statutes"]) for doc in docs[split]]
            result = accuracies(predicted, gold, len(statutes))
            print(f"{split:5s} {method:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in result.items()))
            rows.extend(metric_rows("lsi", method, split, result, len(gold)))
    write_rows(config.RESULTS_DIR / "lsi_metrics.csv", rows + keep_other_rows(rows))


def keep_other_rows(new_rows: list) -> list:
    """write_rows replaces whole (system, split) groups, so carry over the existing metrics of those groups."""
    from casetrail.eval.results_io import read_rows
    new_metrics = {(r["system"], r["split"], r["metric"]) for r in new_rows}
    groups = {(r["system"], r["split"]) for r in new_rows}
    return [r for r in read_rows(config.RESULTS_DIR / "lsi_metrics.csv")
            if (r["system"], r["split"]) in groups and (r["system"], r["split"], r["metric"]) not in new_metrics]


if __name__ == "__main__":
    main()
