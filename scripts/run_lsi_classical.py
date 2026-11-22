"""Classical statute identification (LSI): Naive Bayes, Rocchio and kNN -> results/lsi_metrics.csv.

Everything is fitted on LSI train. Hyper-parameters (NB alpha, kNN k) and the per-statute thresholds are chosen
on dev only; the test split is scored once at the end. Models are saved to models/ for the demo.
Text pipeline = the PCR one (light stemming on, chosen on PCR dev).
"""
import argparse
import json

import numpy as np

from casetrail import config
from casetrail.classify.knn import knn_scores, nearest_neighbours
from casetrail.classify.lsi_data import build_train_index, doc_vectors, label_matrix, lsi_tokens, statute_list
from casetrail.classify.naive_bayes import naive_bayes_scores, train_naive_bayes
from casetrail.classify.rocchio import rocchio_scores, tfidf_vectors, train_rocchio
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.eval.metrics import multilabel_f1
from casetrail.eval.results_io import metric_rows, write_rows
from casetrail.retrieval.bm25 import tfidf_doc_norms

NB_ALPHAS = [0.001, 0.003, 0.01, 0.1, 0.5, 1.0]
KNN_KS = [5, 10, 25, 50, 75, 100]
STEM = True


def evaluate(scores: np.ndarray, thresholds: np.ndarray, labels: np.ndarray, statutes: list) -> dict:
    gold = [{statutes[c] for c in np.flatnonzero(row)} for row in labels]
    return multilabel_f1(predict_sets(scores, thresholds, statutes), gold, statutes)


def pick_on_dev(name: str, dev_scores_by_setting: dict, dev_labels: np.ndarray, statutes: list) -> tuple:
    """Best setting by dev macro-F1 (thresholds tuned on dev for each setting)."""
    best = None
    for setting, scores in dev_scores_by_setting.items():
        thresholds = tune_thresholds(scores, dev_labels)
        macro = evaluate(scores, thresholds, dev_labels, statutes)["macro_f1"]
        print(f"  {name} setting={setting}: dev macro-F1 = {macro:.4f}")
        if best is None or macro > best[2]:
            best = (setting, thresholds, macro)
    return best


def knn_neighbours(index, rows: list) -> list:
    norms = tfidf_doc_norms(index)
    neighbours = []
    for doc_id, tokens, statutes in rows:
        query = {}
        for token in tokens:
            if token in index.term_to_id:
                query[token] = query.get(token, 0) + 1
        neighbours.append(nearest_neighbours(index, norms, query, max(KNN_KS)))
    return neighbours


def result_rows(method: str, split: str, result: dict, top_statutes: list, n: int) -> list:
    metrics = {key: result[key] for key in ["macro_f1", "macro_precision", "macro_recall", "micro_f1",
                                            "micro_precision", "micro_recall"]}
    for statute in top_statutes:
        metrics[f"f1[{statute}]"] = result["per_label"][statute]["f1"]
    return metric_rows("lsi", method, split, metrics, n)


def run_naive_bayes(data: dict) -> dict:
    print("Naive Bayes")
    models = {alpha: train_naive_bayes(data["vectors"]["train"], data["labels"]["train"], data["num_terms"], alpha)
              for alpha in NB_ALPHAS}
    dev_scores = {alpha: naive_bayes_scores(data["vectors"]["dev"], model) for alpha, model in models.items()}
    alpha, thresholds, dev_macro = pick_on_dev("nb", dev_scores, data["labels"]["dev"], data["statutes"])
    return {"dev": dev_scores[alpha], "test": naive_bayes_scores(data["vectors"]["test"], models[alpha]),
            "thresholds": thresholds, "setting": alpha, "model": models[alpha]}


def run_rocchio(data: dict) -> dict:
    print("Rocchio")
    index = data["index"]
    idf = np.log(index.num_docs() / np.maximum(index.df, 1))
    tfidf = {split: tfidf_vectors(data["vectors"][split], idf) for split in config.LSI_SPLITS}
    centroids = train_rocchio(tfidf["train"], data["labels"]["train"], data["num_terms"])
    dev_scores = rocchio_scores(tfidf["dev"], centroids)
    setting, thresholds, dev_macro = pick_on_dev("rocchio", {"centroids": dev_scores}, data["labels"]["dev"],
                                                 data["statutes"])
    return {"dev": dev_scores, "test": rocchio_scores(tfidf["test"], centroids), "thresholds": thresholds,
            "centroids": centroids, "idf": idf}


def run_knn(data: dict) -> dict:
    print("kNN (finding neighbours in the training index)")
    neighbours = {split: knn_neighbours(data["index"], data["rows"][split]) for split in ["dev", "test"]}
    train_labels = data["labels"]["train"]
    dev_scores = {k: knn_scores(neighbours["dev"], train_labels, k) for k in KNN_KS}
    k, thresholds, dev_macro = pick_on_dev("knn", dev_scores, data["labels"]["dev"], data["statutes"])
    return {"dev": dev_scores[k], "test": knn_scores(neighbours["test"], train_labels, k), "thresholds": thresholds,
            "setting": k}


def load_inputs() -> dict:
    statutes = statute_list()
    rows = {split: lsi_tokens(split, STEM) for split in config.LSI_SPLITS}
    index = build_train_index(rows["train"])
    index.save(config.INDEX_DIR / "lsi_train")
    labels = {split: label_matrix(rows[split], statutes) for split in config.LSI_SPLITS}
    vectors = {split: doc_vectors(rows[split], index) for split in config.LSI_SPLITS}
    return {"statutes": statutes, "rows": rows, "index": index, "labels": labels, "vectors": vectors,
            "num_terms": len(index.term_to_id)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data = load_inputs()
    frequency = data["labels"]["train"].sum(axis=0)
    top_statutes = [data["statutes"][c] for c in np.argsort(-frequency)[:10]]
    results = {"nb": run_naive_bayes(data), "rocchio": run_rocchio(data), "knn": run_knn(data)}
    report(results, data["labels"], data["statutes"], top_statutes)
    save_models(results, data["statutes"], {"nb_alpha": results["nb"]["setting"], "knn_k": results["knn"]["setting"],
                                            "stemming": STEM, "vocabulary": data["num_terms"]})


def report(results: dict, labels: dict, statutes: list, top_statutes: list):
    out_rows = []
    runs_dir = config.INDEX_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for method in ["nb", "rocchio", "knn"]:
        for split in ["dev", "test"]:
            np.save(runs_dir / f"lsi_{method}_{split}.npy", results[method][split])
            result = evaluate(results[method][split], results[method]["thresholds"], labels[split], statutes)
            print(f"{split:5s} {method:8s} macro-F1={result['macro_f1']:.4f}  micro-F1={result['micro_f1']:.4f}")
            out_rows.extend(result_rows(method, split, result, top_statutes, len(labels[split])))
    write_rows(config.RESULTS_DIR / "lsi_metrics.csv", out_rows)
    print("Wrote results/lsi_metrics.csv")


def save_models(results: dict, statutes: list, settings: dict):
    config.MODEL_DIR.mkdir(exist_ok=True)
    np.savez(config.MODEL_DIR / "lsi_classical.npz", nb_weights=results["nb"]["model"]["weights"],
             nb_prior=results["nb"]["model"]["prior"], rocchio_centroids=results["rocchio"]["centroids"],
             idf=results["rocchio"]["idf"], nb_thresholds=results["nb"]["thresholds"],
             rocchio_thresholds=results["rocchio"]["thresholds"], knn_thresholds=results["knn"]["thresholds"])
    params_path = config.RESULTS_DIR / "tuned_params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    params["lsi"] = dict(settings, tuned_on="dev", statutes=statutes, thresholds="per statute, models/lsi_classical.npz")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print("Saved models/lsi_classical.npz and the LSI entry of results/tuned_params.json")


if __name__ == "__main__":
    main()
