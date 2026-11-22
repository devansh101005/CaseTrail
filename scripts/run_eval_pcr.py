"""Evaluate the tuned lexical PCR systems -> results/pcr_metrics.csv (long format) + runs in indexes/runs/.

Parameters and K come from results/tuned_params.json (chosen on dev by tune_pcr_lexical.py); they are applied
unchanged to test. Metrics: micro-F1@K with the dev-chosen K (headline, IL-TUR protocol), micro P/R@K,
P@K and R@K for K = 5, 10, 20 (averaged over queries), MAP@100, nDCG@10, MRR.
"""
import argparse
import json

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import pcr_run_metrics
from casetrail.eval.results_io import metric_rows, record_k, write_rows
from casetrail.retrieval.pcr_lexical import load_item_tokens, qrels_of, rank_split

SYSTEM_ORDER = ["tfidf", "bm25_default", "bm25", "bm25_qr", "bm25_bigram", "bm25_passages"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = parser.parse_args()

    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        systems = json.load(f)["pcr"]["systems"]
    runs_dir = config.INDEX_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        queries = load_pcr(split)["queries"]
        qrels = qrels_of(queries)
        tokens = {False: load_item_tokens(split, False), True: load_item_tokens(split, True)}
        cache = {}
        for name in SYSTEM_ORDER:
            system = systems[name]
            run = rank_split(split, system, tokens[system["stem"]], queries, cache)
            with open(runs_dir / f"pcr_{split}_{name}.json", "w", encoding="utf-8") as f:
                json.dump(run, f)
            metrics = pcr_run_metrics(run, qrels, system["k"])
            print(f"{split:5s} {name:14s} K={system['k']:2d}  micro-F1@K={metrics['micro_f1@k']:.4f}  "
                  f"MAP={metrics['map@100']:.4f}  nDCG@10={metrics['ndcg@10']:.4f}")
            rows.extend(metric_rows("pcr", name, split, metrics, len(qrels)))
            record_k(name, system["k"])

    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)
    print("Wrote results/pcr_metrics.csv")


if __name__ == "__main__":
    main()
