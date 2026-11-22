"""Tune the lexical PCR systems on the DEV split only -> results/tuned_params.json + results/pcr_dev_tuning.csv.

Coordinate-wise grid search (cheaper than the full grid, and easy to explain):
  stage A: query representation (full text vs top-N terms, raw vs binary qtf), stemming, unigram vs +bigram,
           at the textbook defaults k1 = 1.2, b = 0.75;
  stage B: k1 and b for each system family at its best stage-A setting;
  stage C: passage-level BM25 (max vs top-3 aggregation).
For every system the cut-off K of micro-F1@K is also chosen on dev (K = 1..20), as in U-CREAT.
The test split is never touched here.
"""
import argparse
import csv
import json
from pathlib import Path

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.metrics import best_k
from casetrail.retrieval.pcr_lexical import load_item_tokens, qrels_of, rank_split

K_VALUES = list(range(1, 21))
TOP_TERMS = [0, 25, 50, 100, 200, 400, 800, 1600]
K1_VALUES = [0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 4.5, 6.0, 10.0]
B_VALUES = [0.3, 0.5, 0.75, 0.9, 1.0]


def describe(system: dict) -> str:
    keys = ["model", "ngram", "stem", "top_terms", "binary_query", "k1", "b", "passages"]
    return "|".join(f"{key}={system.get(key)}" for key in keys)


def evaluate(system: dict, state: dict) -> float:
    """Dev micro-F1 at the best K; every evaluated setting is logged in state['rows']."""
    tokens = state["tokens"][system["stem"]]
    run = rank_split("dev", system, tokens, state["queries"], state["cache"])
    k, score = best_k(run, state["qrels"], K_VALUES)
    state["rows"].append({"system": describe(system), "k": k, "f1": score})
    print(f"  {describe(system)}  K={k}  dev micro-F1@K={score:.4f}")
    return score


def best_of(systems: list, state: dict) -> tuple:
    best_system, best_score = None, -1.0
    for system in systems:
        score = evaluate(system, state)
        if score > best_score:
            best_system, best_score = system, score
    return best_system, best_score


def tune_k1_b(system: dict, state: dict) -> dict:
    grid = []
    for k1 in K1_VALUES:
        for b in B_VALUES:
            grid.append(dict(system, k1=k1, b=b))
    best_system, best_score = best_of(grid, state)
    return best_system


def stage_a(state: dict) -> dict:
    """Best query setting per (ngram, stem), at k1 = 1.2, b = 0.75."""
    found = {}
    for ngram in [1, 2]:
        for stem in [False, True]:
            for top_terms in TOP_TERMS:
                for binary in [False, True]:
                    system = {"model": "bm25", "ngram": ngram, "stem": stem, "top_terms": top_terms,
                              "binary_query": binary, "k1": 1.2, "b": 0.75, "passages": None}
                    found[(ngram, stem, top_terms, binary)] = evaluate(system, state)
    return found


def best_setting(found: dict, ngram: int, stem: bool, full_text: bool) -> tuple:
    """(top_terms, binary) with the best stage-A score; full_text=True means top_terms = 0 only, False excludes 0."""
    options = []
    for (g, s, top_terms, binary), score in found.items():
        if g == ngram and s == stem and (top_terms == 0) == full_text:
            options.append((score, top_terms, binary))
    score, top_terms, binary = max(options)
    return top_terms, binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "tuned_params.json"))
    args = parser.parse_args()

    dev = load_pcr("dev")
    state = {"queries": dev["queries"], "qrels": qrels_of(dev["queries"]), "cache": {}, "rows": [],
             "tokens": {False: load_item_tokens("dev", False), True: load_item_tokens("dev", True)}}

    print("Stage A: query representation, stemming, n-grams")
    found = stage_a(state)
    uni_best = {False: -1.0, True: -1.0}
    for (ngram, stem_used, top_terms, binary), score in found.items():
        if ngram == 1:
            uni_best[stem_used] = max(uni_best[stem_used], score)
    stem = uni_best[True] > uni_best[False]
    print(f"Stemming {'ON' if stem else 'OFF'} (best unigram dev F1: off {uni_best[False]:.4f}, on {uni_best[True]:.4f})")

    print("Stage B: k1 and b per system")
    systems = {}
    top_terms, binary = best_setting(found, 1, stem, full_text=True)
    systems["bm25"] = tune_k1_b({"model": "bm25", "ngram": 1, "stem": stem, "top_terms": 0,
                                 "binary_query": binary, "passages": None}, state)
    top_terms, binary = best_setting(found, 1, stem, full_text=False)
    systems["bm25_qr"] = tune_k1_b({"model": "bm25", "ngram": 1, "stem": stem, "top_terms": top_terms,
                                    "binary_query": binary, "passages": None}, state)
    bigram_options = [best_setting(found, 2, stem, full_text=True), best_setting(found, 2, stem, full_text=False)]
    bigram_scores = [found[(2, stem, n, q)] for n, q in bigram_options]
    top_terms, binary = bigram_options[0] if bigram_scores[0] >= bigram_scores[1] else bigram_options[1]
    systems["bm25_bigram"] = tune_k1_b({"model": "bm25", "ngram": 2, "stem": stem, "top_terms": top_terms,
                                        "binary_query": binary, "passages": None}, state)

    print("Stage C: passage-level BM25")
    passage_grid = []
    for aggregate in ["max", "top3"]:
        for k1 in [0.9, 1.2, 2.0, 3.0, 6.0]:
            for b in [0.5, 0.75, 1.0]:
                passage_grid.append(dict(systems["bm25_qr"], k1=k1, b=b, passages=aggregate))
    systems["bm25_passages"], score = best_of(passage_grid, state)

    # Untuned references: textbook BM25 and tf-idf cosine on the full query text
    systems["bm25_default"] = {"model": "bm25", "ngram": 1, "stem": False, "top_terms": 0, "binary_query": False,
                               "k1": config.BM25_K1, "b": config.BM25_B, "passages": None}
    systems["tfidf"] = {"model": "tfidf", "ngram": 1, "stem": stem, "top_terms": 0, "binary_query": False,
                        "passages": None}

    print("Final K per system (dev)")
    tuned = {}
    for name, system in systems.items():
        run = rank_split("dev", system, state["tokens"][system["stem"]], state["queries"], state["cache"])
        k, score = best_k(run, state["qrels"], K_VALUES)
        tuned[name] = dict(system, k=k, dev_micro_f1=round(score, 4))
        print(f"  {name}: K={k} dev micro-F1@K={score:.4f}")
    save(args.out, tuned, state["rows"], stem, len(state["qrels"]))


def save(out_path: str, tuned: dict, rows: list, stem: bool, num_queries: int):
    # Keep other tasks' entries (e.g. LSI thresholds) when the file already exists
    params = {}
    if Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            params = json.load(f)
    params["pcr"] = {"tuned_on": "dev", "stemming": stem, "systems": tuned}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    with open(config.RESULTS_DIR / "pcr_dev_tuning.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "system", "split", "metric", "value", "n"])
        for row in rows:
            writer.writerow(["pcr", row["system"], "dev", f"micro_f1@{row['k']}", round(row["f1"], 4), num_queries])
    print(f"Wrote {out_path} and results/pcr_dev_tuning.csv")


if __name__ == "__main__":
    main()
