"""Train the statute cross-attention classifier (LSI) on frozen chunk vectors, for several seeds.

Needs scripts/encode_lsi.py first. For every seed: train on LSI train, keep the epoch with the best dev macro-F1
(per-statute thresholds tuned on dev), then save the dev and test probability matrices to
indexes/runs/lsi_attention_seed<seed>_<split>.npy. Metrics are computed by scripts/eval_kaggle_runs.py
(thresholds from dev, applied unchanged to test). Training curves -> results/training_statute_attention.json.
"""
import argparse
import json

import numpy as np
import torch

from casetrail import config
from casetrail.classify.thresholds import predict_sets, tune_thresholds
from casetrail.data import load_lsi, load_statutes
from casetrail.eval.metrics import multilabel_f1
from casetrail.neural.statute_attention import StatuteAttention, positive_weights, predict_scores, train_epoch

TRAIN_DIR = config.DATA_DIR / "training"


def load_split(split: str, statute_ids: list, prefix: str = "") -> tuple:
    vectors = np.load(TRAIN_DIR / f"{prefix}lsi_{split}_vectors.npy").astype(np.float32)
    offsets = np.load(TRAIN_DIR / f"{prefix}lsi_{split}_offsets.npy")
    doc_chunks = [vectors[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    column = {statute: c for c, statute in enumerate(statute_ids)}
    docs = load_lsi(split)[:len(doc_chunks)]  # same order as encode_lsi.py (smoke runs keep the first documents)
    labels = np.zeros((len(docs), len(statute_ids)), dtype=np.float32)
    for d, doc in enumerate(docs):
        for statute in doc["statutes"]:
            labels[d, column[statute]] = 1.0
    return doc_chunks, labels


def dev_macro_f1(scores: np.ndarray, labels: np.ndarray, statute_ids: list) -> float:
    thresholds = tune_thresholds(scores, labels)
    gold = [{statute_ids[c] for c in np.flatnonzero(row)} for row in labels]
    return multilabel_f1(predict_sets(scores, thresholds, statute_ids), gold, statute_ids)["macro_f1"]


def train_seed(seed: int, data: dict, args, device, prefix: str = "") -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StatuteAttention(data["statute_vectors"], heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_weight = positive_weights(data["train"][1])
    best = {"dev_macro_f1": -1.0}
    curve = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, data["train"][0], data["train"][1], pos_weight, args.batch_size,
                           device, rng)
        dev_scores = predict_scores(model, data["dev"][0], 256, device)
        macro = dev_macro_f1(dev_scores, data["dev"][1], data["statute_ids"])
        print(f"  seed {seed} epoch {epoch}: loss {loss:.4f} dev macro-F1 {macro:.4f}")
        curve.append({"epoch": epoch, "train_loss": loss, "dev_macro_f1": macro})
        if macro > best["dev_macro_f1"]:
            best = {"dev_macro_f1": macro, "epoch": epoch, "dev_scores": dev_scores,
                    "test_scores": predict_scores(model, data["test"][0], 256, device)}
    runs_dir = config.INDEX_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    np.save(runs_dir / f"{prefix}lsi_attention_seed{seed}_dev.npy", best["dev_scores"])
    np.save(runs_dir / f"{prefix}lsi_attention_seed{seed}_test.npy", best["test_scores"])
    return {"seed": seed, "best_epoch": best["epoch"], "curve": curve}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[config.SEED, config.SEED + 1, config.SEED + 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true", help="use the smoke_ files of encode_lsi.py --max-docs")
    args = parser.parse_args()
    prefix = "smoke_" if args.smoke else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    statute_ids = list(load_statutes())
    data = {"statute_ids": statute_ids, "statute_vectors": np.load(TRAIN_DIR / f"{prefix}lsi_statute_vectors.npy")}
    for split in config.LSI_SPLITS:
        data[split] = load_split(split, statute_ids, prefix)
    log = {"settings": vars(args), "device": str(device), "seeds": []}
    for seed in args.seeds:
        log["seeds"].append(train_seed(seed, data, args, device, prefix))
    log_path = TRAIN_DIR / "smoke_training_log.json" if args.smoke else config.RESULTS_DIR / "training_statute_attention.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
