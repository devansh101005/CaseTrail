"""Encode LSI judgments (chunk vectors) and the 100 statute descriptions with a frozen encoder (GPU recommended).

Writes data/training/lsi_<split>_vectors.npy (float16), lsi_<split>_offsets.npy, lsi_<split>_ids.json and
lsi_statute_vectors.npy, for scripts/train_statute_attention.py. On a laptop CPU this takes over an hour
(about 400k chunks), so it runs in the Kaggle notebook.
"""
import argparse
import json

import numpy as np
import torch

from casetrail import config
from casetrail.data import load_lsi, load_statutes
from casetrail.neural.encoder import chunk_judgment, load_encoder
from casetrail.neural.train_retriever import encode_all

OUT_DIR = config.DATA_DIR / "training"
LSI_MAX_CHUNKS = 16   # LSI documents are shorter (median ~500 words); 16 chunks cover almost all of them


def encode_split(split: str, tokenizer, model, device, max_docs: int = 0):
    docs = load_lsi(split)
    prefix = ""
    if max_docs:  # smoke run on a CPU: first documents only, written under smoke_ names
        docs, prefix = docs[:max_docs], "smoke_"
    texts, offsets = [], [0]
    for doc in docs:
        texts.extend(chunk_judgment(doc["sentences"], tokenizer, max_chunks=LSI_MAX_CHUNKS))
        offsets.append(len(texts))
    print(f"LSI {split}: {len(docs)} documents -> {len(texts)} chunks")
    vectors = encode_all(model, tokenizer, texts, device, batch_size=128)
    np.save(OUT_DIR / f"{prefix}lsi_{split}_vectors.npy", vectors.astype(np.float16))
    np.save(OUT_DIR / f"{prefix}lsi_{split}_offsets.npy", np.array(offsets, dtype=np.int64))
    with open(OUT_DIR / f"{prefix}lsi_{split}_ids.json", "w", encoding="utf-8") as f:
        json.dump([doc["id"] for doc in docs], f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="encoder folder (e.g. models/student_distilled)")
    parser.add_argument("--max-docs", type=int, default=0, help="smoke test only: first N documents per split")
    args = parser.parse_args()
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_encoder(args.model_dir)
    model.to(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statutes = load_statutes()
    statute_texts = [f"{s['title']}. {s['text']}" for s in statutes.values()]
    prefix = "smoke_" if args.max_docs else ""
    np.save(OUT_DIR / f"{prefix}lsi_statute_vectors.npy", encode_all(model, tokenizer, statute_texts, device))
    for split in config.LSI_SPLITS:
        encode_split(split, tokenizer, model, device, args.max_docs)


if __name__ == "__main__":
    main()
