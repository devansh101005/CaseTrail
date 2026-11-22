"""Train the cross-encoder teacher, score the training pairs for distillation, and re-rank BM25 for dev/test.

Run on a GPU (Kaggle) after scripts/prepare_training_data.py. Outputs:
  models/teacher/                         fine-tuned cross-encoder
  data/training/teacher_scores.json       teacher score of every (train query, positive / hard negative)
  indexes/runs/pcr_<split>_teacher_rerank.json   BM25 top-50 re-ranked by the teacher (rankings only;
                                          metrics are computed locally by scripts/eval_kaggle_runs.py)
  results/training_teacher.json           training loss per epoch
"""
import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from casetrail import config
from casetrail.neural.distill import rerank_run, score_pairs, teacher_examples, train_teacher_epoch

TRAIN_DIR = config.DATA_DIR / "training"


def load_json(name: str):
    with open(TRAIN_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def teacher_scores(model, tokenizer, items: list, chunks: dict, device) -> dict:
    pairs = []
    for item in items:
        for doc_id in item["positives"] + item["hard_negatives"]:
            pairs.append((item["query_id"], doc_id))
    scores = score_pairs(model, tokenizer, pairs, chunks, device)
    result = {}
    for (query_id, doc_id), score in zip(pairs, scores):
        result.setdefault(query_id, {})[doc_id] = score
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-base", help="HF id or local folder")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-examples", type=int, default=0, help="for smoke tests only (0 = all)")
    parser.add_argument("--rerank-depth", type=int, default=50)
    args = parser.parse_args()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, chunks = load_json("train_items.json"), load_json("chunks_train.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log = {"settings": vars(args), "device": str(device), "epochs": []}
    for epoch in range(1, args.epochs + 1):
        examples = teacher_examples(items, random.Random(config.SEED + epoch))
        if args.max_examples:
            examples = examples[:args.max_examples]
        settings = {"seed": config.SEED + epoch, "batch_size": args.batch_size}
        loss = train_teacher_epoch(model, tokenizer, optimizer, examples, chunks, settings, device)
        print(f"epoch {epoch}: train loss {loss:.4f}")
        log["epochs"].append({"epoch": epoch, "train_loss": loss})
    # Smoke runs (--max-examples) never touch the real outputs
    prefix = "smoke_" if args.max_examples else ""
    out_dir = config.MODEL_DIR / f"{prefix}teacher"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    score_items = items[:4] if args.max_examples else items
    with open(TRAIN_DIR / f"{prefix}teacher_scores.json", "w", encoding="utf-8") as f:
        json.dump(teacher_scores(model, tokenizer, score_items, chunks, device), f)
    runs_dir = config.INDEX_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for split in ["dev", "test"]:
        base = load_json(f"bm25_{split}_top100.json")
        if args.max_examples:
            base = dict(list(base.items())[:2])
        split_chunks = load_json(f"chunks_{split}.json")
        reranked = rerank_run(model, tokenizer, base, split_chunks, device, args.rerank_depth)
        with open(runs_dir / f"{prefix}pcr_{split}_teacher_rerank.json", "w", encoding="utf-8") as f:
            json.dump(reranked, f)
        print(f"re-ranked {len(reranked)} {split} queries")
    log_path = out_dir / "training_log.json" if args.max_examples else config.RESULTS_DIR / "training_teacher.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
