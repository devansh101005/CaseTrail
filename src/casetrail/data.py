"""Load the IL-TUR `pcr` and `lsi` tasks from data/raw/ into plain dicts (LLD section 2).

pcr[split] = {"queries":    {doc_id: {"id", "sentences", "relevant"}},
              "candidates": {cand_id: {"id", "sentences"}}}
lsi[split] = [{"id", "sentences", "statutes": [statute_id]}]
statutes   = {statute_id: {"id", "title", "text"}}
Note: every PCR query also appears in its own split's candidate pool; retrieval must skip the query itself.
"""
import json

import pyarrow.parquet as pq

from casetrail import config


def parquet_path(task: str, name: str):
    path = config.RAW_DIR / task / f"{name}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return path


def read_rows(task: str, name: str) -> list:
    return pq.read_table(parquet_path(task, name)).to_pylist()


def load_pcr(split: str) -> dict:
    if split not in config.PCR_SPLITS:
        raise ValueError(f"PCR split must be one of {config.PCR_SPLITS}, got {split!r}")
    queries = {}
    for row in read_rows("pcr", f"{split}_queries"):
        queries[row["id"]] = {"id": row["id"], "sentences": row["text"], "relevant": row["relevant_candidates"]}
    candidates = {}
    for row in read_rows("pcr", f"{split}_candidates"):
        candidates[row["id"]] = {"id": row["id"], "sentences": row["text"]}
    return {"queries": queries, "candidates": candidates}


def lsi_label_names() -> list:
    """Statute name for each integer label, from the Hugging Face ClassLabel metadata in the parquet file."""
    schema = pq.read_schema(parquet_path("lsi", "train"))
    features = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    return features["labels"][0]["names"]


def load_lsi(split: str) -> list:
    if split not in config.LSI_SPLITS:
        raise ValueError(f"LSI split must be one of {config.LSI_SPLITS}, got {split!r}")
    names = lsi_label_names()
    docs = []
    for row in read_rows("lsi", split):
        statutes = [names[label] for label in row["labels"]]
        docs.append({"id": row["id"], "sentences": row["text"], "statutes": statutes})
    return docs


def load_statutes() -> dict:
    """The 100 IPC sections: first list item is the title (e.g. 'Punishment of offences ...:'), the rest the text."""
    statutes = {}
    for row in read_rows("lsi", "statutes"):
        title = row["text"][0].strip().rstrip(":")
        text = " ".join(row["text"][1:])
        statutes[row["id"]] = {"id": row["id"], "title": title, "text": text}
    return statutes


def doc_text(doc: dict) -> str:
    return " ".join(doc["sentences"])
