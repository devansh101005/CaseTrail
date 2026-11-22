"""Loading helpers for CaseTrailEngine (search.py): passages, dense index + encoder, statute classifiers, titles.

Each loader returns None when its artifacts have not been built, so the demo still runs with fewer signals.
"""
import json
import re

import numpy as np

from casetrail import config
from casetrail.classify.lsi_data import statute_list
from casetrail.index.inverted_index import InvertedIndex
from casetrail.retrieval.bm25 import tfidf_cosine_scores, tfidf_doc_norms
from casetrail.retrieval.pcr_lexical import index_folder

MASK_TAG = re.compile(r"<([A-Z_]+)>")
PREFIX = {"intfloat__e5-small-v2": "query: "}


def opening_text(sentences: list) -> str:
    """First ~300 non-space characters of a judgment, used to recognise a pasted pool judgment."""
    return re.sub(r"\s+", "", " ".join(sentences))[:300]


def readable_tag(match) -> str:
    return f"[{match.group(1).lower()}]"


def title_of(sentences: list) -> str:
    """'PETITIONER: A Vs. RESPONDENT: B ...' -> 'A v. B'; otherwise the first words of the judgment."""
    head = MASK_TAG.sub(readable_tag, " ".join(sentences[:2]))
    match = re.search(r"PETITIONER:\s*(.*?)\s+Vs\.?\s+RESPONDENT:\s*(.*?)\s+DATE OF JUDGMENT", head)
    if match:
        return f"{match.group(1).strip()} v. {match.group(2).strip()}"[:120]
    return head.strip()[:90]


def passage_resources(split: str, passage_system: dict):
    folder = index_folder(split, 1, passage_system["stem"], passages=True)
    if not (folder / "meta.json").exists():
        return None
    with open(folder / "spans.json", encoding="utf-8") as f:
        spans = json.load(f)
    return {"index": InvertedIndex().load(folder), "passage_doc": np.load(folder / "passage_doc.npy"), "spans": spans,
            "k1": passage_system["k1"], "b": passage_system["b"]}


def best_dense_entry(params: dict, split: str):
    """The usable dense model with the best dev micro-F1 of its weighted fusion (chosen on dev)."""
    entries = [(value["weighted"]["dev_micro_f1"], key) for key, value in params["pcr"].items()
               if key.startswith("dense_") and isinstance(value, dict) and "weighted" in value
               and dense_info(value["model"], split) is not None]
    return max(entries)[1] if entries else None


def make_encoder(info: dict):
    """The encoder recorded in the dense index (ONNX file if it was built with one, else PyTorch);
    returns (function text items -> chunk vectors, runtime name)."""
    # Imported here so the demo starts without loading torch when no neural model has been built
    from casetrail.neural.encoder import chunk_judgment, encode_texts, load_encoder
    from casetrail.neural.export_onnx import onnx_encode, onnx_session
    model_dir = config.PROJECT_ROOT / info["model_dir"]
    tokenizer, torch_model = load_encoder(str(model_dir))
    prefix = info.get("prefix") or PREFIX.get(model_dir.name, "")
    session = onnx_session(config.PROJECT_ROOT / info["onnx"]) if info.get("onnx") else None

    def encode(text_items: list) -> np.ndarray:
        chunks = chunk_judgment(text_items, tokenizer)
        if session is not None:
            return onnx_encode(session, tokenizer, chunks, prefix=prefix)
        return encode_texts(tokenizer, torch_model, chunks, prefix=prefix)

    return encode, ("onnx_int8" if session is not None else "torch_fp32")


def dense_info(model: str, split: str):
    """What the dense index records about its encoder; older indexes without it fall back to models/hf/<model>."""
    path = config.INDEX_DIR / "dense" / f"{model}_{split}" / "doc_names.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    if "model_dir" not in info:
        info["model_dir"] = f"models/hf/{model}"
        info["onnx"] = None
    return info if (config.PROJECT_ROOT / info["model_dir"]).exists() else None


def dense_resources(split: str, params: dict):
    key = best_dense_entry(params, split)
    if key is None:
        return None
    entry = params["pcr"][key]
    folder = config.INDEX_DIR / "dense" / f"{entry['model']}_{split}"
    encode, runtime = make_encoder(dense_info(entry["model"], split))
    return {"vectors": np.load(folder / "vectors.npy"), "offsets": np.load(folder / "offsets.npy"), "encode": encode,
            "neural_weight": entry["weighted"]["neural_weight"], "model": entry["model"], "runtime": runtime}


def lsi_resources(params: dict):
    path = config.MODEL_DIR / "lsi_classical.npz"
    if "lsi" not in params or not path.exists():
        return None
    models = dict(np.load(path))
    index = InvertedIndex().load(config.INDEX_DIR / "lsi_train")
    return {"models": models, "index": index, "statutes": statute_list(), "stemming": params["lsi"]["stemming"],
            "knn_k": params["lsi"]["knn_k"], "knn_labels": None, "knn_norms": None}


def knn_inputs(lsi: dict):
    """Training label matrix and tf-idf norms for kNN, loaded on first use (reading LSI train takes a while)."""
    if lsi["knn_labels"] is None:
        from casetrail.data import load_lsi
        column = {statute: c for c, statute in enumerate(lsi["statutes"])}
        docs = load_lsi("train")
        labels = np.zeros((len(docs), len(lsi["statutes"])))
        for d, doc in enumerate(docs):
            for statute in doc["statutes"]:
                labels[d, column[statute]] = 1.0
        lsi["knn_labels"] = labels
        lsi["knn_norms"] = tfidf_doc_norms(lsi["index"])
    return lsi["knn_labels"], lsi["knn_norms"]


def statute_scores(lsi: dict, tokens: list, method: str) -> tuple:
    """(score per statute, dev-tuned threshold per statute) for one document."""
    index, models = lsi["index"], lsi["models"]
    counts = {}
    for token in tokens:
        if token in index.term_to_id:
            counts[token] = counts.get(token, 0) + 1
    term_ids = np.array([index.term_to_id[t] for t in counts], dtype=np.int64)
    tfs = np.array(list(counts.values()), dtype=np.float64)
    if method == "nb":
        return models["nb_prior"] + tfs @ models["nb_weights"][term_ids], models["nb_thresholds"]
    if method == "rocchio":
        weights = (1.0 + np.log(tfs)) * models["idf"][term_ids] if len(tfs) else tfs
        weights = weights / max(np.linalg.norm(weights), 1e-12)
        return weights @ models["rocchio_centroids"][term_ids], models["rocchio_thresholds"]
    labels, norms = knn_inputs(lsi)
    similarity = tfidf_cosine_scores(index, counts, norms)
    neighbours = np.argsort(-similarity)[:lsi["knn_k"]]
    return similarity[neighbours] @ labels[neighbours], models["knn_thresholds"]
