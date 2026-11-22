"""Prepare neural training data -> data/training/ (run on Kaggle or locally after build_index.py).

- train_items.json: every TRAIN query with its relevant candidates and BM25 hard negatives;
- dev_queries.json: dev query ids + relevant ids (for early stopping / model selection on dev);
- chunks_<split>.json: chunk texts of every judgment of the train, dev and test pools;
- bm25_<split>_top100.json: the tuned bigram BM25 ranking for dev and test queries (for teacher re-ranking).
Test relevance labels are never written here.
Hard negatives (DECISIONS B5): non-relevant candidates in the BM25 top 50 that rank below the first relevant one,
so the few strongest unlabelled matches above every precedent (possible unlabelled precedents) are skipped.
"""
import argparse
import json

from transformers import AutoTokenizer

from casetrail import config
from casetrail.data import load_pcr
from casetrail.neural.encoder import chunk_judgment
from casetrail.retrieval.pcr_lexical import load_item_tokens, rank_split

OUT_DIR = config.DATA_DIR / "training"
MINE_DEPTH = 50
MAX_NEGATIVES = 10


def hard_negatives(ranked: list, relevant: set) -> list:
    start = 10  # no precedent in the top 50: skip the first 10 as possible unlabelled precedents
    for i, doc_id in enumerate(ranked):
        if doc_id in relevant:
            start = i + 1
            break
    return [doc_id for doc_id in ranked[start:] if doc_id not in relevant][:MAX_NEGATIVES]


def write_json(name: str, value):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(value, f)
    print(f"  wrote data/training/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=str(config.MODEL_DIR / "hf" / "sentence-transformers__all-MiniLM-L6-v2"),
                        help="tokenizer used to measure chunk lengths (MiniLM and e5-small share the BERT vocabulary)")
    args = parser.parse_args()
    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        system = json.load(f)["pcr"]["systems"]["bm25_bigram"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for split in config.PCR_SPLITS:
        pcr = load_pcr(split)
        tokens = load_item_tokens(split, system["stem"])
        run = rank_split(split, system, tokens, pcr["queries"], {}, depth=100)
        if split == "train":
            items = []
            for query_id, query in pcr["queries"].items():
                relevant = set(query["relevant"])
                positives = [doc_id for doc_id in query["relevant"] if doc_id in pcr["candidates"]]
                items.append({"query_id": query_id, "positives": positives,
                              "hard_negatives": hard_negatives(run[query_id][:MINE_DEPTH], relevant)})
            write_json("train_items.json", items)
        else:
            write_json(f"bm25_{split}_top100.json", run)
        if split == "dev":
            write_json("dev_queries.json", {query_id: query["relevant"] for query_id, query in pcr["queries"].items()})
        print(f"  chunking {len(pcr['candidates'])} {split} judgments ...")
        chunks = {doc_id: chunk_judgment(doc["sentences"], tokenizer) for doc_id, doc in pcr["candidates"].items()}
        write_json(f"chunks_{split}.json", chunks)


if __name__ == "__main__":
    main()
