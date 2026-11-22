"""Add link-analysis scores as a static prior to the best lexical PCR system (Lectures: PageRank, reputation systems).

final(d) = bm25(d) / max_bm25  +  lambda * z(log prior(d)),   re-ranking the top 100 BM25 candidates.
z() standardises log scores over the judgments of the pool. lambda and K are chosen on DEV (lambda = 0 is in
the grid, so a useless prior gets lambda = 0); the chosen values are then applied unchanged to test.
Writes rows "bm25_bigram+<prior>" to results/pcr_metrics.csv and the chosen values to results/tuned_params.json.
"""
import argparse
import json
import math

import numpy as np

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import best_k, pcr_run_metrics
from casetrail.eval.results_io import metric_rows, record_k, save_run, save_tuned, write_rows
from casetrail.retrieval.pcr_lexical import load_item_tokens, qrels_of, rank_split

BASE = "bm25_bigram"
LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
K_VALUES = list(range(1, 21))
GRAPH_DIR = config.INDEX_DIR / "graph"


def prior_file(prior: str, split: str) -> str:
    if prior == "citation_pagerank":
        return "pcr_citation_pagerank"
    return f"pcr_{split}_{prior}"


def prior_z(prior: str, split: str, pool_ids: list) -> dict:
    """z-score of log(prior) over the pool; judgments missing from the graph get the smallest graph score."""
    path = GRAPH_DIR / f"{prior_file(prior, split)}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/build_graph.py` first.")
    with open(path, encoding="utf-8") as f:
        scores = json.load(f)
    smallest = min(value for value in scores.values() if value > 0)
    logs = np.array([math.log(max(scores.get(doc_id, smallest), smallest)) for doc_id in pool_ids])
    spread = logs.std() if logs.std() > 0 else 1.0
    return {doc_id: (value - logs.mean()) / spread for doc_id, value in zip(pool_ids, logs)}


def rerank(base_run: dict, z: dict, weight: float) -> dict:
    run = {}
    for query_id, ranked in base_run.items():
        top_score = max(ranked[0][1], 1e-12)
        rescored = [(score / top_score + weight * z[doc_id], doc_id) for doc_id, score in ranked]
        rescored.sort(reverse=True)
        run[query_id] = [doc_id for score, doc_id in rescored]
    return run


def base_run(split: str, system: dict) -> tuple:
    queries = load_pcr(split)["queries"]
    tokens = load_item_tokens(split, system["stem"])
    run = rank_split(split, system, tokens, queries, {}, with_scores=True)
    return run, qrels_of(queries), sorted(tokens)


def tune_on_dev(prior: str, system: dict) -> dict:
    run, qrels, pool_ids = base_run("dev", system)
    z = prior_z(prior, "dev", pool_ids)
    best = {"lambda": 0.0, "k": system["k"], "dev_micro_f1": -1.0}
    for weight in LAMBDAS:
        k, score = best_k(rerank(run, z, weight), qrels, K_VALUES)
        print(f"  {prior} lambda={weight:<5} K={k:2d} dev micro-F1@K={score:.4f}")
        if score > best["dev_micro_f1"]:
            best = {"lambda": weight, "k": k, "dev_micro_f1": round(score, 4)}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", nargs="+", default=["citation_pagerank", "similarity_pagerank", "reference_hub"])
    args = parser.parse_args()

    params_path = config.RESULTS_DIR / "tuned_params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    system = params["pcr"]["systems"][BASE]
    chosen = {}
    for prior in args.priors:
        chosen[prior] = tune_on_dev(prior, system)
        print(f"{prior}: chosen on dev {chosen[prior]}")

    rows = []
    for split in ["dev", "test"]:
        run, qrels, pool_ids = base_run(split, system)
        for prior in args.priors:
            z = prior_z(prior, split, pool_ids)
            reranked = rerank(run, z, chosen[prior]["lambda"])
            save_run(split, f"{BASE}+{prior}", reranked)
            record_k(f"{BASE}+{prior}", chosen[prior]["k"])
            metrics = pcr_run_metrics(reranked, qrels, chosen[prior]["k"])
            print(f"{split:5s} {BASE}+{prior}: micro-F1@K={metrics['micro_f1@k']:.4f}")
            rows.extend(metric_rows("pcr", f"{BASE}+{prior}", split, metrics, len(qrels)))
    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)
    save_tuned("pcr", "graph_priors", {"base": BASE, "tuned_on": "dev", "priors": chosen})
    print("Wrote results/pcr_metrics.csv and results/tuned_params.json")


if __name__ == "__main__":
    main()
