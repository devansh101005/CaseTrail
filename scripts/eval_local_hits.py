"""Query-dependent HITS re-ranking for PCR (Lecture: HITS - root set / base set, Kleinberg 1999).

For each query: root set = top ROOT BM25 results; edges = BM25 similarity links (each judgment -> its top-10
neighbours in the pool) that stay inside the root set; HITS on this small graph gives each result an authority
score ("many good results point to it"). final = bm25 / max_bm25 + lambda * authority / max_authority.
lambda and K are chosen on DEV; test is scored once. Rows "bm25_bigram+local_hits" -> results/pcr_metrics.csv.
"""
import argparse
import json

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import best_k, pcr_run_metrics
from casetrail.eval.results_io import metric_rows, record_k, save_run, save_tuned, write_rows
from casetrail.graph.hits import hits
from casetrail.retrieval.pcr_lexical import load_item_tokens, qrels_of, rank_split

BASE = "bm25_bigram"
ROOT = 50
NEIGHBOURS = 10
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
K_VALUES = list(range(1, 21))


def local_authority(ranked: list, neighbours: dict) -> dict:
    """HITS authority of every root-set judgment, scaled so the best one is 1."""
    root = [doc_id for doc_id, score in ranked[:ROOT]]
    position = {doc_id: i for i, doc_id in enumerate(root)}
    edges = []
    for doc_id in root:
        for neighbour in neighbours.get(doc_id, []):
            if neighbour in position and neighbour != doc_id:
                edges.append((position[doc_id], position[neighbour], 1.0))
    if not edges:
        return {doc_id: 0.0 for doc_id in root}
    hub, authority, iterations = hits(len(root), len(root), edges, config.HITS_TOL, config.HITS_MAX_ITER)
    top = max(authority.max(), 1e-12)
    return {doc_id: float(authority[position[doc_id]] / top) for doc_id in root}


def rerank(base_run: dict, authorities: dict, weight: float) -> dict:
    run = {}
    for query_id, ranked in base_run.items():
        top_score = max(ranked[0][1], 1e-12)
        rescored = []
        for doc_id, score in ranked:
            rescored.append((score / top_score + weight * authorities[query_id].get(doc_id, 0.0), doc_id))
        rescored.sort(reverse=True)
        run[query_id] = [doc_id for score, doc_id in rescored]
    return run


def prepare(split: str, system: dict) -> tuple:
    """Base run with scores, qrels, and per-query local authorities."""
    queries = load_pcr(split)["queries"]
    tokens = load_item_tokens(split, system["stem"])
    cache = {}
    base_run = rank_split(split, system, tokens, queries, cache, with_scores=True)
    print(f"  {split}: finding the {NEIGHBOURS} BM25 neighbours of every judgment in the pool ...")
    neighbour_run = rank_split(split, system, tokens, sorted(tokens), cache, depth=NEIGHBOURS)
    authorities = {query_id: local_authority(ranked, neighbour_run) for query_id, ranked in base_run.items()}
    return base_run, qrels_of(queries), authorities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    params_path = config.RESULTS_DIR / "tuned_params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    system = params["pcr"]["systems"][BASE]

    run, qrels, authorities = prepare("dev", system)
    chosen = {"lambda": 0.0, "k": system["k"], "dev_micro_f1": -1.0}
    for weight in LAMBDAS:
        k, score = best_k(rerank(run, authorities, weight), qrels, K_VALUES)
        print(f"  lambda={weight:<5} K={k:2d} dev micro-F1@K={score:.4f}")
        if score > chosen["dev_micro_f1"]:
            chosen = {"lambda": weight, "k": k, "dev_micro_f1": round(score, 4)}
    print(f"chosen on dev: {chosen}")

    rows = []
    for split in ["dev", "test"]:
        if split == "test":
            run, qrels, authorities = prepare("test", system)
        reranked = rerank(run, authorities, chosen["lambda"])
        save_run(split, f"{BASE}+local_hits", reranked)
        record_k(f"{BASE}+local_hits", chosen["k"])
        metrics = pcr_run_metrics(reranked, qrels, chosen["k"])
        print(f"{split:5s} {BASE}+local_hits: micro-F1@K={metrics['micro_f1@k']:.4f}")
        rows.extend(metric_rows("pcr", f"{BASE}+local_hits", split, metrics, len(qrels)))
    write_rows(config.RESULTS_DIR / "pcr_metrics.csv", rows)
    save_tuned("pcr", "local_hits", dict(chosen, base=BASE, root=ROOT, neighbours=NEIGHBOURS, tuned_on="dev"))


if __name__ == "__main__":
    main()
