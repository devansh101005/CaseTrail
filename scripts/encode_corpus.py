"""Encode every judgment of a PCR pool into chunk vectors -> indexes/dense/<model>_<split>/ (HLD 5.1 step 5).

Saves vectors.npy (unit-length float32, one row per chunk), offsets.npy (doc i owns rows offsets[i]:offsets[i+1]),
doc_names.json, and the encoding time (used for the efficiency table).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from casetrail import config
from casetrail.data import load_pcr
from casetrail.neural.encoder import chunk_judgment, encode_texts, load_encoder
from casetrail.neural.export_onnx import onnx_encode, onnx_session

# e5 models expect a "query: " prefix for symmetric (document-to-document) similarity
PREFIX = {"intfloat__e5-small-v2": "query: "}


def dense_folder(model_name: str, split: str):
    return config.INDEX_DIR / "dense" / f"{model_name}_{split}"


def encode_pool(model_dir, model_name: str, split: str, batch_size: int, onnx_path=None, prefix_key: str = ""):
    tokenizer, model = load_encoder(str(model_dir))
    session = onnx_session(onnx_path) if onnx_path else None
    prefix = PREFIX.get(prefix_key or model_dir.name, "")
    pool = load_pcr(split)["candidates"]
    names = sorted(pool)
    chunks, offsets = [], [0]
    for name in names:
        chunks.extend(chunk_judgment(pool[name]["sentences"], tokenizer))
        offsets.append(len(chunks))
    print(f"{model_name} {split}: {len(names)} judgments -> {len(chunks)} chunks; encoding ...")
    start = time.time()
    if session is None:
        vectors = encode_texts(tokenizer, model, chunks, batch_size, prefix)
    else:
        vectors = onnx_encode(session, tokenizer, chunks, batch_size, prefix)
    seconds = time.time() - start
    folder = dense_folder(model_name, split)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", vectors)
    np.save(folder / "offsets.npy", np.array(offsets, dtype=np.int64))
    with open(folder / "doc_names.json", "w", encoding="utf-8") as f:
        # model_dir / onnx: the engine must encode queries with exactly the encoder that built this index
        json.dump({"doc_names": names, "model_dir": model_dir.as_posix(), "onnx": str(onnx_path) if onnx_path else None,
                   "prefix": prefix, "chunks": len(chunks), "encode_seconds": round(seconds, 1),
                   "chunks_per_second": round(len(chunks) / seconds, 1), "threads": torch.get_num_threads()}, f)
    print(f"  done in {seconds:.0f} s ({len(chunks) / seconds:.1f} chunks/s on CPU)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Hugging Face model folder, e.g. models/hf/intfloat__e5-small-v2 "
                        "or models/student_distilled")
    parser.add_argument("--name", default="", help="name of the output (default: the folder name)")
    parser.add_argument("--onnx", default="", help="encode with this ONNX file instead of PyTorch (e.g. the INT8 model)")
    parser.add_argument("--prefix-key", default="", help="base model folder name, to apply the e5 'query: ' prefix")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(config.SEED)
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"{model_dir} not found. Run `python scripts/download_models.py` (or the Kaggle training) first.")
    name = args.name or model_dir.name
    for split in args.splits:
        encode_pool(model_dir, name, split, args.batch_size, args.onnx or None, args.prefix_key)


if __name__ == "__main__":
    main()
