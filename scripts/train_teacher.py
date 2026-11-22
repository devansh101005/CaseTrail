"""Train the cross-encoder teacher, score the training pairs for distillation, and re-rank BM25 for dev/test.

Run on a GPU (Kaggle) after scripts/prepare_training_data.py. With --tag sel, every output name gets "_sel":
  models/teacher[_sel]/                         fine-tuned cross-encoder
  data/training/teacher_scores[_sel].json       teacher score of every (train query, positive / hard negative)
  indexes/runs/pcr_<split>_teacher_rerank[_sel].json   BM25 top-50 re-ranked by the teacher (rankings only;
                                                metrics are computed locally by scripts/eval_kaggle_runs.py)
  results/training_teacher[_sel].json           settings, loss, and the dev pair accuracy check
--select-with <encoder folder>: the teacher sees the most similar chunk pairs (by that encoder) instead of random
ones; run 1 with random pairs never got above chance (see src/casetrail/neural/distill.py).
"""
import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from casetrail import config
from casetrail.neural.distill import dev_pair_accuracy, rerank_run, score_pairs, teacher_examples, train_teacher_epoch
from casetrail.neural.train_retriever import chunk_vectors

TRAIN_DIR = config.DATA_DIR / "training"


def load_json(name: str):
    with open(TRAIN_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def smoke_subset(items: list, chunks: dict, split_chunks: dict, bm25: dict, dev_qrels: dict, depth: int) -> tuple:
    """For --max-examples smoke runs on a CPU: 4 train queries, 5 dev and 2 test queries, and only their judgments."""
    items = items[:4]
    keep = set()
    for item in items:
        keep.add(item["query_id"])
        keep.update(item["positives"] + item["hard_negatives"])
    chunks = {d: chunks[d] for d in keep if d in chunks}
    bm25 = {split: dict(list(sorted(bm25[split].items()))[:2]) for split in ["dev", "test"]}
    dev_qrels = {q: dev_qrels[q] for q in sorted(dev_qrels)[:5]}
    small = {}
    for split in ["dev", "test"]:
        needed = set()
        for query_id, ranked in bm25[split].items():
            needed.add(query_id)
            needed.update(ranked[:depth])
        if split == "dev":
            for query_id, relevant in dev_qrels.items():
                needed.add(query_id)
                needed.update(relevant)
                needed.update(bm25_dev_negatives(query_id, relevant))
        small[split] = {d: split_chunks[split][d] for d in needed if d in split_chunks[split]}
    return items, chunks, small, bm25, dev_qrels


def bm25_dev_negatives(query_id: str, relevant: set) -> list:
    ranked = load_json("bm25_dev_top100.json")[query_id]
    return [d for d in ranked if d not in relevant][:3]


def teacher_scores(model, tokenizer, items: list, chunks: dict, device, vectors: dict) -> dict:
    pairs = []
    for item in items:
        for doc_id in item["positives"] + item["hard_negatives"]:
            pairs.append((item["query_id"], doc_id))
    scores = score_pairs(model, tokenizer, pairs, chunks, device, vectors=vectors)
    result = {}
    for (query_id, doc_id), score in zip(pairs, scores):
        result.setdefault(query_id, {})[doc_id] = score
    return result


def train(model, tokenizer, items: list, chunks: dict, args, device, vectors: dict, log: dict):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        examples = teacher_examples(items, random.Random(config.SEED + epoch))
        if args.max_examples:
            examples = examples[:args.max_examples]
        steps = (len(examples) + args.batch_size - 1) // args.batch_size
        settings = {"seed": config.SEED + epoch, "batch_size": args.batch_size, "lr": args.lr,
                    "warmup_steps": int(args.warmup * steps) if epoch == 1 else 0}
        loss = train_teacher_epoch(model, tokenizer, optimizer, examples, chunks, settings, device, vectors)
        print(f"epoch {epoch}: train loss {loss:.4f}")
        log["epochs"].append({"epoch": epoch, "train_loss": loss})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-base", help="HF id or local folder")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1, help="share of the first epoch used for lr warm-up")
    parser.add_argument("--select-with", default="", help="encoder folder used to pick the most similar chunk pairs")
    parser.add_argument("--tag", default="", help="suffix for every output name, e.g. sel")
    parser.add_argument("--max-examples", type=int, default=0, help="for smoke tests only (0 = all)")
    parser.add_argument("--rerank-depth", type=int, default=50)
    args = parser.parse_args()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, chunks = load_json("train_items.json"), load_json("chunks_train.json")
    split_chunks = {split: load_json(f"chunks_{split}.json") for split in ["dev", "test"]}
    bm25 = {split: load_json(f"bm25_{split}_top100.json") for split in ["dev", "test"]}
    dev_qrels = {q: set(r) for q, r in load_json("dev_queries.json").items()}
    if args.max_examples:
        items, chunks, split_chunks, bm25, dev_qrels = smoke_subset(items, chunks, split_chunks, bm25, dev_qrels,
                                                                   args.rerank_depth)
    vectors = None
    if args.select_with:
        vectors = chunk_vectors(args.select_with, [chunks, split_chunks["dev"], split_chunks["test"]], device)
    # Smoke runs (--max-examples) never touch the real outputs
    prefix = "smoke_" if args.max_examples else ""
    suffix = f"_{args.tag}" if args.tag else ""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    log = {"settings": vars(args), "device": str(device), "epochs": []}
    train(model, tokenizer, items, chunks, args, device, vectors, log)

    accuracy = dev_pair_accuracy(model, tokenizer, dev_qrels, load_json("bm25_dev_top100.json"), split_chunks["dev"],
                                 device, vectors)
    log["dev_pair_accuracy"] = accuracy
    print(f"dev pair accuracy (cited case above a BM25 negative; 0.5 = chance): {accuracy:.3f}")
    out_dir = config.MODEL_DIR / f"{prefix}teacher{suffix}"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(TRAIN_DIR / f"{prefix}teacher_scores{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(teacher_scores(model, tokenizer, items, chunks, device, vectors), f)
    runs_dir = config.INDEX_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for split in ["dev", "test"]:
        reranked = rerank_run(model, tokenizer, bm25[split], split_chunks[split], device, args.rerank_depth, vectors)
        with open(runs_dir / f"{prefix}pcr_{split}_teacher_rerank{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(reranked, f)
        print(f"re-ranked {len(reranked)} {split} queries")
    log_path = out_dir / "training_log.json" if args.max_examples else config.RESULTS_DIR / f"training_teacher{suffix}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
