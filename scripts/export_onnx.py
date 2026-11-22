"""Export an encoder to ONNX fp32 + INT8 and measure size, CPU latency and INT8 fidelity -> results/efficiency.csv.

Latency = time to encode the chunks of one query judgment (the per-query neural cost in the demo),
for PyTorch fp32, ONNX fp32 and ONNX INT8, on this laptop's CPU. Run it when the machine is otherwise idle.
"""
import argparse
import time

import numpy as np
import torch

from casetrail import config
from casetrail.data import load_pcr
from casetrail.eval.results_io import metric_rows, write_rows
from casetrail.neural.encoder import chunk_judgment, encode_texts, load_encoder
from casetrail.neural.export_onnx import export_fp32, quantise_int8
from casetrail.neural.onnx_runtime import onnx_encode, onnx_session

PREFIX = {"intfloat__e5-small-v2": "query: "}


def dev_chunks(tokenizer, limit: int) -> tuple:
    """Chunks of dev judgments for the fidelity check, and the chunks of one median-length judgment for latency."""
    pool = load_pcr("dev")["candidates"]
    names = sorted(pool)
    chunks = []
    per_doc = []
    for name in names[:60]:
        doc_chunks = chunk_judgment(pool[name]["sentences"], tokenizer)
        per_doc.append(doc_chunks)
        chunks.extend(doc_chunks)
    per_doc.sort(key=len)
    return chunks[:limit], per_doc[len(per_doc) // 2]


def encode_variant(variant: str, texts: list, tools: dict) -> np.ndarray:
    if variant == "torch_fp32":
        return encode_texts(tools["tokenizer"], tools["model"], texts, prefix=tools["prefix"])
    return onnx_encode(tools[variant], tools["tokenizer"], texts, prefix=tools["prefix"])


def time_encoding(variant: str, texts: list, tools: dict, repeats: int = 20) -> dict:
    """Median and 95th-percentile wall time (ms) of encoding `texts` once, after one warm-up call."""
    encode_variant(variant, texts, tools)
    times = []
    for repeat in range(repeats):
        start = time.perf_counter()
        encode_variant(variant, texts, tools)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Hugging Face model folder (pretrained or our trained student)")
    parser.add_argument("--tag", required=True, help="short name for files and result rows, e.g. minilm")
    parser.add_argument("--prefix-key", default="", help="folder name used to look up the e5 'query: ' prefix")
    args = parser.parse_args()
    torch.manual_seed(config.SEED)
    out_dir = config.MODEL_DIR / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = out_dir / f"{args.tag}_fp32.onnx", out_dir / f"{args.tag}_int8.onnx"
    tokenizer = export_fp32(args.model_dir, fp32_path)
    quantise_int8(fp32_path, int8_path)
    prefix = PREFIX.get(args.prefix_key, "")

    tokenizer, model = load_encoder(args.model_dir)
    check_chunks, query_chunks = dev_chunks(tokenizer, 200)
    tools = {"tokenizer": tokenizer, "model": model, "prefix": prefix,
             "onnx_fp32": onnx_session(fp32_path), "onnx_int8": onnx_session(int8_path)}
    reference = encode_variant("torch_fp32", check_chunks, tools)
    int8_vectors = encode_variant("onnx_int8", check_chunks, tools)
    cosines = (reference * int8_vectors).sum(axis=1)
    print(f"INT8 vs PyTorch fp32 cosine on {len(cosines)} dev chunks: mean {cosines.mean():.4f}, min {cosines.min():.4f}")
    sizes = {"torch_fp32": fp32_path.stat().st_size / 1e6, "onnx_fp32": fp32_path.stat().st_size / 1e6,
             "onnx_int8": int8_path.stat().st_size / 1e6}
    rows = []
    for variant in ["torch_fp32", "onnx_fp32", "onnx_int8"]:
        timing = time_encoding(variant, query_chunks, tools)
        metrics = {"size_mb": sizes[variant], "query_encode_median_ms": timing["median_ms"],
                   "query_encode_p95_ms": timing["p95_ms"], "query_chunks": len(query_chunks)}
        if variant == "onnx_int8":
            metrics["cosine_vs_fp32_mean"] = float(cosines.mean())
            metrics["cosine_vs_fp32_min"] = float(cosines.min())
        print(f"{variant}: {metrics}")
        rows.extend(metric_rows("efficiency", f"{args.tag}_{variant}", "dev", metrics, len(query_chunks)))
    write_rows(config.RESULTS_DIR / "efficiency.csv", rows)
    print("Wrote results/efficiency.csv")


if __name__ == "__main__":
    main()
