"""Dense (neural) PCR retrieval and its fusion with BM25 -> results/pcr_metrics.csv + results/tuned_params.json.

Needs scripts/encode_corpus.py for dev and test first. Choices made on DEV only: chunk aggregation
(MaxSim vs mean vector), fusion method parameters (RRF k, weighted-sum weight) and K; test is scored once.
Rows: dense_<model>, fusion_rrf_<model>, fusion_weighted_<model> (fused with the tuned bigram BM25).
"""
import argparse
import json

import numpy as np

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import best_k, pcr_run_metrics
from casetrail.eval.results_io import metric_rows, record_k, save_run, save_tuned, write_rows
from casetrail.fusion.combine import ids_only, rrf, weighted_sum
from casetrail.neural.encoder import maxsim_scores, mean_vector_scores
from casetrail.retrieval.bm25 import top_k
from casetrail.retrieval.pcr_lexical import RUN_DEPTH, load_item_tokens, qrels_of, rank_split

LEXICAL = "bm25_bigram"
K_VALUES = list(range(1, 21))
RRF_KS = [10, 30, 60, 100]
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]   # weight of the neural ranker; BM25 gets 1 - weight
AGGREGATE = {"maxsim": maxsim_scores, "mean": mean_vector_scores}


def dense_run(model: str, split: str, queries: dict, aggregate: str) -> dict:
    folder = config.INDEX_DIR / "dense" / f"{model}_{split}"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found. Run `python scripts/encode_corpus.py --model {model}` first.")
    vectors = np.load(folder / "vectors.npy")
    offsets = np.load(folder / "offsets.npy")
    with open(folder / "doc_names.json", encoding="utf-8") as f:
        names = json.load(f)["doc_names"]
    position = {name: i for i, name in enumerate(names)}
    run = {}
    for query_id in queries:
        i = position[query_id]
        scores = AGGREGATE[aggregate](vectors[offsets[i]:offsets[i + 1]], vectors, offsets)
        run[query_id] = [(names[j], score) for j, score in top_k(scores, RUN_DEPTH, exclude={i})]
    return run


def lexical_run(split: str, queries: dict, system: dict) -> dict:
    tokens = load_item_tokens(split, system["stem"])
    return rank_split(split, system, tokens, queries, {}, with_scores=True)


def tune_on_dev(model: str, lexical_system: dict) -> dict:
    queries = load_pcr("dev")["queries"]
    qrels = qrels_of(queries)
    chosen = {}
    best = (-1.0, None, None)
    for aggregate in AGGREGATE:
        run = dense_run(model, "dev", queries, aggregate)
        k, score = best_k(ids_only(run), qrels, K_VALUES)
        print(f"  dense aggregate={aggregate}: K={k} dev micro-F1@K={score:.4f}")
        if score > best[0]:
            best = (score, aggregate, k)
    chosen["dense"] = {"aggregate": best[1], "k": best[2], "dev_micro_f1": round(best[0], 4)}
    dense = dense_run(model, "dev", queries, best[1])
    lexical = lexical_run("dev", queries, lexical_system)
    chosen["rrf"] = pick(qrels, [(rrf_k, rrf([lexical, dense], rrf_k)) for rrf_k in RRF_KS], "rrf_k")
    chosen["weighted"] = pick(qrels, [(w, weighted_sum([lexical, dense], [1 - w, w])) for w in WEIGHTS],
                              "neural_weight")
    return chosen


def pick(qrels: dict, candidates: list, name: str) -> dict:
    best = None
    for value, run in candidates:
        k, score = best_k(ids_only(run), qrels, K_VALUES)
        print(f"  {name}={value}: K={k} dev micro-F1@K={score:.4f}")
        if best is None or score > best["dev_micro_f1"]:
            best = {name: value, "k": k, "dev_micro_f1": round(score, 4)}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="folder name under indexes/dense/ without the split")
    parser.add_argument("--tag", required=True, help="short name used in result rows, e.g. minilm")
    args = parser.parse_args()
    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        lexical_system = json.load(f)["pcr"]["systems"][LEXICAL]
    chosen = tune_on_dev(args.model, lexical_system)
    print(f"chosen on dev: {chosen}")
    rows = []
    for split in ["dev", "test"]:
        queries = load_pcr(split)["queries"]
        qrels = qrels_of(queries)
        dense = dense_run(args.model, split, queries, chosen["dense"]["aggregate"])
        lexical = lexical_run(split, queries, lexical_system)
        runs = {f"dense_{args.tag}": (dense, chosen["dense"]["k"]),
                f"fusion_rrf_{args.tag}": (rrf([lexical, dense], chosen["rrf"]["rrf_k"]), chosen["rrf"]["k"]),
                f"fusion_weighted_{args.tag}": (weighted_sum([lexical, dense], [1 - chosen["weighted"]["neural_weight"],
                                                                             chosen["weighted"]["neural_weight"]]),
                                                chosen["weighted"]["k"])}
        for name, (run, k) in runs.items():
            save_run(split, name, ids_only(run))
            record_k(name, k)
            metrics = pcr_run_metrics(ids_only(run), qrels, k)
            print(f"{split:5s} {name}: micro-F1@K={metrics['micro_f1@k']:.4f}")
            rows.extend(metric_rows("pcr", name, split, metrics, len(qrels)))
    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)
    save_tuned("pcr", f"dense_{args.tag}", dict(chosen, model=args.model, lexical=LEXICAL, tuned_on="dev"))


if __name__ == "__main__":
    main()
