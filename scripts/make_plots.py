"""Figures for the report and the evidence page -> results/fig_*.png (default matplotlib style, as the report asks).

Every value is read from results/ (or computed from saved dev runs); a figure whose inputs are missing is skipped.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
import numpy as np  # noqa: E402

from casetrail import config  # noqa: E402
from casetrail.data import load_pcr  # noqa: E402
from casetrail.eval.metrics import micro_prf_at_k  # noqa: E402
from casetrail.eval.results_io import read_rows  # noqa: E402
from casetrail.retrieval.pcr_lexical import qrels_of  # noqa: E402

RESULTS = config.RESULTS_DIR


def value_table(path, split: str, metric: str) -> dict:
    return {row["system"]: float(row["value"]) for row in read_rows(path)
            if row["split"] == split and row["metric"] == metric}


def published(task: str) -> list:
    path = RESULTS / "published_baselines.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [row for row in json.load(f)["baselines"] if row["task"] == task]


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=150)
    plt.close(fig)
    print(f"  wrote results/{name}")


def plot_pcr_systems():
    test = value_table(RESULTS / "pcr_metrics.csv", "test", "micro_f1@k")
    if not test:
        return print("  skip fig_pcr_systems (no PCR results)")
    # The leakage ablation is an analysis, not a system
    systems = sorted([s for s in test if "citations_kept" not in s], key=lambda s: test[s])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(systems) + 2))
    ax.barh(systems, [100 * test[s] for s in systems], height=0.6)
    ax.set_ylim(-1.6, len(systems) - 0.4)
    labels = {"BM25 (word level)": "published BM25", "Event-filtered docs + BM25 (quad-gram) [SOTA]": "published SOTA"}
    for row in published("pcr"):
        if row["system"] in labels:
            ax.axvline(row["value_percent"], color="0.3", linestyle="--", linewidth=1)
            ax.text(row["value_percent"], -1.1, f"{labels[row['system']]} {row['value_percent']}", fontsize=7,
                    ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xlabel("test micro-F1@K (%) at the dev-chosen K")
    ax.set_title("PCR: our systems vs published baselines (dashed)")
    save(fig, "fig_pcr_systems.png")


def plot_f1_vs_k():
    queries = load_pcr("dev")["queries"]
    qrels = qrels_of(queries)
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = 0
    for system in ["bm25_default", "bm25", "tfidf", "bm25_bigram"]:
        path = config.INDEX_DIR / "runs" / f"pcr_dev_{system}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        ks = list(range(1, 21))
        ax.plot(ks, [100 * micro_prf_at_k(run, qrels, k)["f1"] for k in ks], marker="o", markersize=3, label=system)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return print("  skip fig_pcr_f1_vs_k (no dev runs)")
    ax.set_xlabel("K (candidates returned per query)")
    ax.set_ylabel("dev micro-F1@K (%)")
    ax.set_title("PCR dev: F1 rises then falls with K (precision/recall trade-off)")
    ax.legend()
    save(fig, "fig_pcr_f1_vs_k.png")


def plot_bm25_grid():
    rows = [row for row in read_rows(RESULTS / "pcr_dev_tuning.csv")
            if "ngram=2" in row["system"] and "passages=None" in row["system"] and "top_terms=400" in row["system"]
            and "binary_query=True" in row["system"]]
    cells = {}
    for row in rows:
        parts = dict(part.split("=") for part in row["system"].split("|"))
        cells[(float(parts["k1"]), float(parts["b"]))] = 100 * float(row["value"])
    if len(cells) < 4:
        return print("  skip fig_bm25_grid (tuning rows missing)")
    k1s, bs = sorted({k for k, b in cells}), sorted({b for k, b in cells})
    grid = np.array([[cells.get((k1, b), np.nan) for b in bs] for k1 in k1s])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(bs)), [str(b) for b in bs])
    ax.set_yticks(range(len(k1s)), [str(k) for k in k1s])
    for i in range(len(k1s)):
        for j in range(len(bs)):
            if not np.isnan(grid[i, j]):
                # White text on the dark half of the colour scale, black on the light half
                dark = grid[i, j] > np.nanmin(grid) + 0.6 * (np.nanmax(grid) - np.nanmin(grid))
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
    ax.set_xlabel("b (length normalisation)")
    ax.set_ylabel("k1 (tf saturation)")
    ax.set_title("Bigram BM25: dev micro-F1@K (%) over k1 and b")
    fig.colorbar(image, ax=ax)
    save(fig, "fig_bm25_grid.png")


def plot_lsi_methods():
    macro = value_table(RESULTS / "lsi_metrics.csv", "test", "macro_f1")
    micro = value_table(RESULTS / "lsi_metrics.csv", "test", "micro_f1")
    methods = [m for m in ["nb", "rocchio", "knn", "neural_attention"] if m in macro]
    if not methods:
        return print("  skip fig_lsi_methods (no LSI results)")
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, [100 * macro[m] for m in methods], width=0.38, label="macro-F1")
    ax.bar(x + 0.2, [100 * micro[m] for m in methods], width=0.38, label="micro-F1")
    for row in published("lsi"):
        ax.axhline(row["value_percent"], color="0.3", linestyle="--", linewidth=1)
        ax.text(-0.45, row["value_percent"] + 0.4, f"LeSICiN macro-F1 {row['value_percent']}", fontsize=7,
                ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xticks(x, methods)
    ax.set_ylabel("test F1 (%)")
    ax.set_title("LSI (100 IPC statutes): classical vs neural")
    ax.legend()
    save(fig, "fig_lsi_methods.png")


def plot_lsi_per_statute():
    rows = read_rows(RESULTS / "lsi_metrics.csv")
    dev_macro = {row["system"]: float(row["value"]) for row in rows if row["split"] == "dev" and row["metric"] == "macro_f1"}
    if not dev_macro:
        return print("  skip fig_lsi_per_statute (no LSI results)")
    best = max(dev_macro, key=dev_macro.get)  # chosen on dev, not test
    per = [(row["metric"][3:-1], 100 * float(row["value"])) for row in rows
           if row["system"] == best and row["split"] == "test" and row["metric"].startswith("f1[")]
    if not per:
        return print("  skip fig_lsi_per_statute (no per-statute rows)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([name for name, value in per][::-1], [value for name, value in per][::-1], height=0.6)
    ax.set_xlabel("test F1 (%)")
    ax.set_title(f"LSI: F1 of the 10 most frequent statutes ({best}, best on dev)")
    save(fig, "fig_lsi_per_statute.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    plot_pcr_systems()
    plot_f1_vs_k()
    plot_bm25_grid()
    plot_lsi_methods()
    plot_lsi_per_statute()


if __name__ == "__main__":
    main()
