"""Read/write the long-format results CSVs (columns: task, system, split, metric, value, n; LLD section 10)."""
import csv
import json

from casetrail import config

HEADER = ["task", "system", "split", "metric", "value", "n"]


def read_rows(path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def write_rows(path, new_rows: list):
    """Replace the rows of the (system, split) pairs present in new_rows, keep all other rows."""
    replaced = {(row["system"], row["split"]) for row in new_rows}
    kept = [row for row in read_rows(path) if (row["system"], row["split"]) not in replaced]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(kept + new_rows)


def metric_rows(task: str, system: str, split: str, metrics: dict, n: int) -> list:
    return [{"task": task, "system": system, "split": split, "metric": metric, "value": round(value, 4), "n": n}
            for metric, value in metrics.items()]


def save_tuned(task: str, key: str, value):
    """Set tuned_params[task][key] = value, re-reading the file first so other scripts' entries are kept."""
    path = config.RESULTS_DIR / "tuned_params.json"
    params = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
    params.setdefault(task, {})[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def save_run(split: str, system: str, run: dict):
    """Ranked candidate ids per query -> indexes/runs/pcr_<split>_<system>.json (used by scripts/significance.py)."""
    folder = config.INDEX_DIR / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"pcr_{split}_{system}.json", "w", encoding="utf-8") as f:
        json.dump(run, f)


def record_k(system: str, k: int):
    """Remember the dev-chosen K of a PCR system (read by scripts/significance.py)."""
    path = config.RESULTS_DIR / "tuned_params.json"
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    params.setdefault("pcr", {}).setdefault("k_by_system", {})[system] = k
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
