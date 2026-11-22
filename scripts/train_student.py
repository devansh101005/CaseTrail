"""Train the bi-encoder student on PCR train (contrastive + BM25 hard negatives, optional distillation).

Run on a GPU (Kaggle T4, see notebooks/train_retriever.ipynb) after scripts/prepare_training_data.py
(and scripts/train_teacher.py for --distill). Model selection / early stopping uses dev only: after every epoch
the dev pool is ranked and the epoch with the best dev micro-F1@K is kept.
Writes models/student_<variant>/ (Hugging Face format) and results/training_student_<variant>.json.
"""
import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from casetrail import config
from casetrail.eval.metrics import best_k
from casetrail.neural.train_retriever import chunk_vectors, dense_dev_run, make_examples, train_epoch

TRAIN_DIR = config.DATA_DIR / "training"


def load_json(name: str):
    path = TRAIN_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/prepare_training_data.py` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_inputs(distill: bool, smoke: bool = False, teacher_file: str = "teacher_scores.json") -> dict:
    items = load_json("train_items.json")
    chunks = load_json("chunks_train.json")
    relevant = {item["query_id"]: set(item["positives"]) for item in items}
    dev_queries = load_json("dev_queries.json")
    teacher = load_json(("smoke_" if smoke else "") + teacher_file) if distill else None
    if teacher is not None:
        items = [item for item in items if item["query_id"] in teacher]  # distil only what the teacher scored
    return {"items": items, "chunks": chunks, "relevant": relevant, "teacher": teacher,
            "dev_chunks": load_json("chunks_dev.json"), "dev_qrels": {q: set(r) for q, r in dev_queries.items()}}


def smoke_subset(data: dict, docs: int = 60) -> dict:
    """For --max-examples smoke runs on a CPU: validate on a few dev queries and a small dev pool (2 chunks each)."""
    queries = sorted(data["dev_qrels"])[:5]
    keep = set(queries)
    for query_id in queries:
        keep |= data["dev_qrels"][query_id] & set(data["dev_chunks"])
    keep |= set(sorted(data["dev_chunks"])[:docs])
    data["dev_chunks"] = {doc_id: data["dev_chunks"][doc_id][:2] for doc_id in keep}
    data["dev_qrels"] = {query_id: data["dev_qrels"][query_id] for query_id in queries}
    # 4 training queries and only their judgments, so a selection encoder has little to embed
    data["items"] = data["items"][:4]
    needed = set()
    for item in data["items"]:
        needed.add(item["query_id"])
        needed.update(item["positives"] + item["hard_negatives"])
    data["chunks"] = {doc_id: data["chunks"][doc_id] for doc_id in needed if doc_id in data["chunks"]}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HF id or local folder")
    parser.add_argument("--variant", default="plain", help="name of the output, e.g. plain or distilled")
    parser.add_argument("--distill", action="store_true", help="add Margin-MSE to teacher scores")
    parser.add_argument("--teacher-scores", default="teacher_scores.json",
                        help="file in data/training/ written by train_teacher.py (e.g. teacher_scores_sel.json)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--select-with", default="", help="fixed encoder folder used to pick candidate chunks that "
                        "match the query chunks (instead of random chunks)")
    parser.add_argument("--max-examples", type=int, default=0, help="for smoke tests only (0 = all)")
    args = parser.parse_args()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_inputs(args.distill, smoke=bool(args.max_examples), teacher_file=args.teacher_scores)
    if args.max_examples:
        data = smoke_subset(data)
    if args.select_with:
        data["selection"] = chunk_vectors(args.select_with, [data["chunks"]], device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Smoke runs (--max-examples) never touch the real outputs
    prefix = "smoke_" if args.max_examples else ""
    out_dir = config.MODEL_DIR / f"{prefix}student_{args.variant}"
    log = {"settings": vars(args), "device": str(device), "epochs": []}
    # Epoch 0 = the pretrained model before any training (zero-shot dev score, also used to pick the student)
    run = dense_dev_run(model, tokenizer, data["dev_chunks"], data["dev_qrels"], device)
    k, best_f1 = best_k(run, data["dev_qrels"], list(range(1, 21)))
    print(f"epoch 0 (pretrained, zero-shot): dev micro-F1@{k} = {best_f1:.4f}")
    log["epochs"].append({"epoch": 0, "train_loss": None, "dev_micro_f1": best_f1, "dev_k": k})
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log["best_epoch"] = 0
    for epoch in range(1, args.epochs + 1):
        examples = make_examples(data["items"], args.hard_negatives, random.Random(config.SEED + epoch))
        if args.max_examples:
            examples = examples[:args.max_examples]
        settings = {"seed": config.SEED + epoch, "batch_size": args.batch_size, "distill_weight": args.distill_weight}
        loss = train_epoch(model, tokenizer, optimizer, examples, data, settings, device)
        run = dense_dev_run(model, tokenizer, data["dev_chunks"], data["dev_qrels"], device)
        k, dev_f1 = best_k(run, data["dev_qrels"], list(range(1, 21)))
        print(f"epoch {epoch}: train loss {loss:.4f}, dev micro-F1@{k} = {dev_f1:.4f}")
        log["epochs"].append({"epoch": epoch, "train_loss": loss, "dev_micro_f1": dev_f1, "dev_k": k})
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            log["best_epoch"] = epoch
    log_path = out_dir / "training_log.json" if args.max_examples else config.RESULTS_DIR / f"training_student_{args.variant}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Best epoch {log['best_epoch']} saved to {out_dir}")


if __name__ == "__main__":
    main()
