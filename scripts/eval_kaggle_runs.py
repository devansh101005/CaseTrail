"""Evaluate the outputs that come back from the Kaggle notebook -> results/pcr_metrics.csv, results/lsi_metrics.csv.

PCR: indexes/runs/pcr_<split>_teacher_rerank.json (BM25 top-50 re-ranked by the cross-encoder teacher);
     K chosen on dev, applied to test.
LSI: indexes/runs/lsi_attention_seed<s>_<split>.npy (statute cross-attention probabilities per seed);
     per-statute thresholds tuned on dev scores, applied to test; mean and std over seeds.
Missing files are reported and skipped.
"""
import argparse
import json

import numpy as np

from casetrail import config
from casetrail.classify.lsi_data import statute_list
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.data import load_lsi, load_pcr
from casetrail.eval.metrics import best_k, multilabel_f1, pcr_run_metrics
from casetrail.eval.results_io import metric_rows, record_k, save_tuned, write_rows
from casetrail.retrieval.pcr_lexical import qrels_of

RUNS_DIR = config.INDEX_DIR / "runs"
LSI_METRICS = ["macro_f1", "macro_precision", "macro_recall", "micro_f1", "micro_precision", "micro_recall"]


def eval_teacher():
    paths = {split: RUNS_DIR / f"pcr_{split}_teacher_rerank.json" for split in ["dev", "test"]}
    if not all(path.exists() for path in paths.values()):
        print("Teacher re-ranking runs not found (run the Kaggle notebook first); skipped.")
        return
    runs, qrels = {}, {}
    for split, path in paths.items():
        with open(path, encoding="utf-8") as f:
            runs[split] = json.load(f)
        qrels[split] = qrels_of(load_pcr(split)["queries"])
    k, dev_f1 = best_k(runs["dev"], qrels["dev"], list(range(1, 21)))
    rows = []
    for split in ["dev", "test"]:
        metrics = pcr_run_metrics(runs[split], qrels[split], k)
        print(f"{split:5s} teacher_rerank: micro-F1@{k} = {metrics['micro_f1@k']:.4f}")
        rows.extend(metric_rows("pcr", "bm25_bigram+teacher_rerank", split, metrics, len(qrels[split])))
    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)
    save_tuned("pcr", "teacher_rerank", {"k": k, "dev_micro_f1": round(dev_f1, 4), "tuned_on": "dev"})
    record_k("bm25_bigram+teacher_rerank", k)
    for split in ["dev", "test"]:
        (RUNS_DIR / f"pcr_{split}_bm25_bigram+teacher_rerank.json").write_text(json.dumps(runs[split]), encoding="utf-8")


def lsi_gold(split: str, statutes: list) -> tuple:
    docs = load_lsi(split)
    column = {statute: c for c, statute in enumerate(statutes)}
    labels = np.zeros((len(docs), len(statutes)))
    for d, doc in enumerate(docs):
        for statute in doc["statutes"]:
            labels[d, column[statute]] = 1.0
    return labels, [set(doc["statutes"]) for doc in docs]


def eval_attention():
    seeds = sorted(int(p.stem.split("seed")[1].split("_")[0]) for p in RUNS_DIR.glob("lsi_attention_seed*_test.npy"))
    if not seeds:
        print("Statute attention scores not found (run the Kaggle notebook first); skipped.")
        return
    statutes = statute_list()
    dev_labels, dev_gold = lsi_gold("dev", statutes)
    test_labels, test_gold = lsi_gold("test", statutes)
    per_seed = {"dev": [], "test": []}
    for seed in seeds:
        dev_scores = np.load(RUNS_DIR / f"lsi_attention_seed{seed}_dev.npy")
        test_scores = np.load(RUNS_DIR / f"lsi_attention_seed{seed}_test.npy")
        thresholds = tune_thresholds(dev_scores, dev_labels)
        per_seed["dev"].append(multilabel_f1(predict_sets(dev_scores, thresholds, statutes), dev_gold, statutes))
        per_seed["test"].append(multilabel_f1(predict_sets(test_scores, thresholds, statutes), test_gold, statutes))
        print(f"seed {seed}: test macro-F1 {per_seed['test'][-1]['macro_f1']:.4f}")
    rows = []
    for split, results in per_seed.items():
        metrics = {}
        for name in LSI_METRICS:
            values = [result[name] for result in results]
            metrics[name] = float(np.mean(values))
            metrics[f"{name}_std"] = float(np.std(values))
        metrics["seeds"] = len(results)
        rows.extend(metric_rows("lsi", "neural_attention", split, metrics, len(dev_gold if split == "dev" else test_gold)))
    write_rows(config.RESULTS_DIR / "lsi_metrics.csv", rows)
    print("Wrote neural_attention rows to results/lsi_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    eval_teacher()
    eval_attention()


if __name__ == "__main__":
    main()
