"""Write the PUBLISHED baseline numbers we compare against to results/published_baselines.json.

These are not our results. They are copied from the papers (table and row given for each), so that the README,
report and frontend read every number from results/ (no hand-typed numbers).
IL-TUR's PCR split (827/118/237 queries = 1182) is the IL-PCR corpus of Joshi et al. (2023), Table 13 of IL-TUR.
"""
import argparse
import json

from casetrail import config

IL_TUR = ("Joshi, A., Paul, S., Sharma, A., Goyal, P., Ghosh, S. and Modi, A., 2024. IL-TUR: Benchmark for Indian "
          "Legal Text Understanding and Reasoning. In Proceedings of the 62nd Annual Meeting of the Association for "
          "Computational Linguistics (Volume 1: Long Papers). (pages: verify)")
U_CREAT = ("Joshi, A., Sharma, A., Tanikella, S.K. and Modi, A., 2023, July. U-CREAT: Unsupervised Case Retrieval using "
           "Events extrAcTion. In Proceedings of the 61st Annual Meeting of the Association for Computational "
           "Linguistics (Volume 1: Long Papers) (pp. 13899-13915).")

PCR_METRIC = ("micro-averaged F1 over all queries of the top-K retrieved candidates; "
              "K chosen on the validation set, F1 reported on test at that K (U-CREAT Sec. 5 and App. A)")

BASELINES = [
    # task, system, metric, value (%), K, precision, recall, source
    ["pcr", "BM25 (word level)", "micro_f1@k", 13.85, 5, 17.11, 11.64, "IL-TUR Table 23; U-CREAT Table 2/5"],
    ["pcr", "BM25 (bigram)", "micro_f1@k", 28.59, 7, 29.30, 27.91, "IL-TUR Table 23; U-CREAT Table 2/5"],
    ["pcr", "Atomic events + BM25", "micro_f1@k", 36.77, 7, 37.69, 35.90, "IL-TUR Table 23"],
    ["pcr", "Event-filtered docs + BM25 (quad-gram) [SOTA]", "micro_f1@k", 39.15, 7, 40.12, 38.22, "IL-TUR Table 23 and Table 2"],
    ["pcr", "DistilBERT (full document)", "micro_f1@k", 16.61, 7, 17.02, 16.21, "IL-TUR Table 23"],
    ["lsi", "LeSICiN (graph-based) [SOTA]", "macro_f1", 28.08, None, None, None, "IL-TUR Table 2 (mF1 = macro-F1)"],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "published_baselines.json"))
    args = parser.parse_args()

    rows = []
    for task, system, metric, value, k, precision, recall, source in BASELINES:
        rows.append({"task": task, "system": system, "metric": metric, "value_percent": value, "k": k,
                     "precision_percent": precision, "recall_percent": recall, "source": source})
    output = {"note": "Published numbers, not ours.", "pcr_metric_definition": PCR_METRIC,
              "references": {"IL-TUR": IL_TUR, "U-CREAT": U_CREAT}, "baselines": rows}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(rows)} published baselines to {args.out}")


if __name__ == "__main__":
    main()
