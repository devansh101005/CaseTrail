"""Error analysis on test (no tuning): best/worst PCR queries and most-confused statute pairs -> results/error_analysis.json.

PCR: per-query F1 at the system's dev-chosen K for the best lexical system, with query length and number of
precedents, so the report can discuss what easy and hard queries look like.
LSI: for the best classical method on dev, the (true statute, wrongly predicted statute) pairs that occur most often.
"""
import argparse
import json
from collections import Counter

import numpy as np

from casetrail import config
from casetrail.classify.lsi_data import statute_list
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.data import load_lsi, load_pcr
from casetrail.eval.metrics import f1
from casetrail.eval.results_io import read_rows
from casetrail.retrieval.pcr_lexical import qrels_of

SYSTEM = "bm25_bigram"


def pcr_queries(params: dict) -> dict:
    k = params["pcr"]["k_by_system"][SYSTEM]
    with open(config.INDEX_DIR / "runs" / f"pcr_test_{SYSTEM}.json", encoding="utf-8") as f:
        run = json.load(f)
    queries = load_pcr("test")["queries"]
    rows = []
    for query_id, relevant in qrels_of(queries).items():
        top = run[query_id][:k]
        hits = len(set(top) & relevant)
        precision, recall = hits / k, hits / len(relevant)
        words = len(" ".join(queries[query_id]["sentences"]).split())
        rows.append({"query_id": query_id, "f1": round(f1(precision, recall), 4), "hits": hits, "k": k,
                     "precedents": len(relevant), "query_words": words})
    rows.sort(key=lambda row: (-row["f1"], row["query_id"]))
    zero = [row for row in rows if row["hits"] == 0]
    return {"system": SYSTEM, "k": k, "best5": rows[:5], "worst5": rows[-5:],
            "queries_with_no_hit": len(zero), "queries": len(rows),
            "median_words_no_hit": int(np.median([r["query_words"] for r in zero])) if zero else None,
            "median_words_all": int(np.median([r["query_words"] for r in rows]))}


def lsi_confusions() -> dict:
    dev_macro = {row["system"]: float(row["value"]) for row in read_rows(config.RESULTS_DIR / "lsi_metrics.csv")
                 if row["split"] == "dev" and row["metric"] == "macro_f1" and row["system"] in ["nb", "rocchio", "knn"]}
    method = max(dev_macro, key=dev_macro.get)  # chosen on dev
    statutes = statute_list()
    column = {statute: c for c, statute in enumerate(statutes)}
    dev_docs = load_lsi("dev")
    dev_labels = np.zeros((len(dev_docs), len(statutes)))
    for d, doc in enumerate(dev_docs):
        for statute in doc["statutes"]:
            dev_labels[d, column[statute]] = 1.0
    runs = config.INDEX_DIR / "runs"
    thresholds = tune_thresholds(np.load(runs / f"lsi_{method}_dev.npy"), dev_labels)
    predicted = predict_sets(np.load(runs / f"lsi_{method}_test.npy"), thresholds, statutes)
    pairs = Counter()
    for doc, pred in zip(load_lsi("test"), predicted):
        gold = set(doc["statutes"])
        for missed in gold - pred:
            for wrong in pred - gold:
                pairs[(missed, wrong)] += 1
    return {"method": method, "top_confusions": [[a, b, n] for (a, b), n in pairs.most_common(10)],
            "note": "pair = (true statute that was missed, statute predicted instead in the same document)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        params = json.load(f)
    output = {"pcr": pcr_queries(params), "lsi": lsi_confusions()}
    with open(config.RESULTS_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    pcr = output["pcr"]
    print(f"PCR {pcr['system']}: {pcr['queries_with_no_hit']}/{pcr['queries']} test queries with no hit in the top {pcr['k']}; "
          f"median words {pcr['median_words_no_hit']} (no hit) vs {pcr['median_words_all']} (all)")
    print("LSI top confusions:", output["lsi"]["top_confusions"][:5])


if __name__ == "__main__":
    main()
