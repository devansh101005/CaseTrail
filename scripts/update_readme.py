"""Regenerate the README results table from results/ (no hand-typed numbers).

Replaces the text between <!-- results:start --> and <!-- results:end --> in README.md.
"""
import argparse
import json

from casetrail import config
from casetrail.eval.results_io import read_rows

START, END = "<!-- results:start -->", "<!-- results:end -->"
PCR_ORDER = ["bm25_default", "tfidf", "bm25", "bm25_qr", "bm25_passages", "bm25_bigram",
             "bm25_bigram+citation_pagerank", "bm25_bigram+local_hits"]


def lookup(rows: list, system: str, split: str, metric: str):
    for row in rows:
        if row["system"] == system and row["split"] == split and row["metric"] == metric:
            return float(row["value"])
    return None


def count_test_items(rows: list) -> str:
    for row in rows:
        if row["split"] == "test":
            return f"{int(row['n']):,}"
    return "not run yet"


def cell(value, scale: float = 100.0) -> str:
    return "not run yet" if value is None else f"{scale * value:.2f}"


def pcr_table(rows: list) -> list:
    systems = [s for s in PCR_ORDER if lookup(rows, s, "test", "micro_f1@k") is not None]
    systems += sorted({r["system"] for r in rows if r["system"] not in PCR_ORDER and r["metric"] == "micro_f1@k"
                       and r["split"] == "test" and "citations_kept" not in r["system"]})
    lines = [f"**Prior case retrieval (IL-TUR `pcr`, {count_test_items(rows)} test queries)**", "",
             "| System | K (dev) | dev µF1@K | test µF1@K | test MAP@100 | test nDCG@10 |", "|---|---|---|---|---|---|"]
    for system in systems:
        k = lookup(rows, system, "test", "k")
        lines.append(f"| {system} | {int(k) if k else '-'} | {cell(lookup(rows, system, 'dev', 'micro_f1@k'))} | "
                     f"{cell(lookup(rows, system, 'test', 'micro_f1@k'))} | {cell(lookup(rows, system, 'test', 'map@100'))} | "
                     f"{cell(lookup(rows, system, 'test', 'ndcg@10'))} |")
    return lines


def lsi_table(rows: list) -> list:
    systems = sorted({r["system"] for r in rows if r["metric"] == "macro_f1" and r["split"] == "test"})
    lines = [f"**Statute identification (IL-TUR `lsi`, {count_test_items(rows)} test documents)**", "",
             "| Method | dev macro-F1 | test macro-F1 | test micro-F1 |", "|---|---|---|---|"]
    for system in systems:
        lines.append(f"| {system} | {cell(lookup(rows, system, 'dev', 'macro_f1'))} | "
                     f"{cell(lookup(rows, system, 'test', 'macro_f1'))} | {cell(lookup(rows, system, 'test', 'micro_f1'))} |")
    return lines


def published_lines() -> list:
    path = config.RESULTS_DIR / "published_baselines.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        baselines = json.load(f)["baselines"]
    lines = ["**Published baselines on the same splits (from the papers, not ours)**", ""]
    for row in baselines:
        lines.append(f"- {row['task'].upper()} · {row['system']}: {row['value_percent']} {row['metric']} ({row['source']})")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    body = pcr_table(read_rows(config.RESULTS_DIR / "pcr_metrics.csv")) + [""]
    body += lsi_table(read_rows(config.RESULTS_DIR / "lsi_metrics.csv")) + [""] + published_lines()
    body += ["", "All values in %. Every setting (k1, b, n-grams, λ, K, thresholds) was chosen on dev; test was scored once."]
    readme_path = config.PROJECT_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        raise SystemExit("README.md has no results markers")
    before, rest = readme.split(START, 1)
    after = rest.split(END, 1)[1]
    readme_path.write_text(before + START + "\n" + "\n".join(body) + "\n" + END + after, encoding="utf-8")
    print("Updated the results table in README.md")


if __name__ == "__main__":
    main()
