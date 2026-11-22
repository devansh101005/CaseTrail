# CaseTrail — Roadmap

**Submission:** ZIP named `CSD358_Devansh_Anamika_Abhinav` with report PDF + GitHub link, source code, dataset (see `docs/SUBMISSION.md`).

Each phase ends in a **working, tested state**. After every phase Claude Code: runs `pytest -q`, summarises what changed in plain English, updates `docs/IR_CONCEPTS.md`, and **suggests** a commit message. Devansh commits. Claude never commits.

Mark items `[x]` when done.

---

## P0 — Setup
- [x] `.venv`, `requirements.txt` (CPU torch install note), `src/casetrail/config.py` (paths, `SEED = 13`)
- [x] `pytest` runs (one smoke test)
- [x] `README.md` skeleton: setup + run commands
**Done when:** `pytest -q` passes on the laptop.

## P1 — Data
- [x] `scripts/download_data.py`: IL-TUR `pcr` and `lsi` (gated; uses `HF_TOKEN` / `huggingface-cli login`) → `data/raw/`
- [x] `src/casetrail/data.py`: load queries, candidates, relevance lists, LSI docs + labels, statute descriptions
- [x] `scripts/data_stats.py` → `results/data_stats.json` (counts, avg length, labels per doc, label frequency)
- [x] Inspect how case citations appear in the text (masked or not) — write the finding into `docs/DATA.md`
**Done when:** counts match the IL-TUR card (PCR 827/118/237 queries; LSI 43k/10k/13k), stats saved.

## P2 — Classical retrieval (PCR)
- [x] Tokeniser + normaliser (keeps section references), stop words
- [x] Inverted index (dictionary + postings + doc lengths), save/load
- [x] BM25 and tf-idf cosine; heap top-K
- [x] Long-query handling: full text vs top-N term query reduction vs paragraph-level max/sum
- [x] `eval/metrics.py`: P@K, R@K, F1@K, **micro-F1@K (IL-TUR definition)**, MAP, nDCG@10 + unit tests
- [x] Tune k1, b, N on **dev** only → `results/tuned_params.json`
**Done when:** test-set µF1@K for BM25 variants in `results/pcr_metrics.csv`.

## P3 — Link analysis
- [x] Build graphs: judgment–statute bipartite; citation graph only if citations are usable (P1 finding); fallback: kNN similarity graph
- [x] PageRank (power iteration, damping, dangling nodes, convergence check) + tests vs a tiny hand-computed graph
- [x] HITS (hubs/authorities, normalisation, convergence) + tests
- [x] Add PageRank prior / HITS features to PCR ranking; measure the change
**Done when:** ablation row "+ PageRank" in `results/pcr_metrics.csv`; top authorities listed in `results/graph_stats.json`.

## P4 — Statute identification baselines (LSI)
- [x] Multinomial Naive Bayes (one-vs-rest), Rocchio, kNN on tf-idf — written by us
- [x] Decision thresholds tuned on **dev**
- [x] Macro-F1, micro-F1, P, R per method → `results/lsi_metrics.csv`; per-statute F1 for the most frequent statutes
**Done when:** three classical methods compared with the LeSICiN baseline (28.08 macro-F1).

## P5 — Neural retriever (train on Kaggle, infer on laptop)
- [ ] `notebooks/train_retriever.ipynb`: bi-encoder, contrastive loss, **BM25 hard negatives**, chunking for long judgments — *code + CPU smoke tests done; needs the Kaggle run*
- [ ] Teacher cross-encoder trained/fine-tuned on PCR train pairs; **distil** into the student — *code + CPU smoke tests done; needs the Kaggle run*
- [x] Export student → ONNX → INT8; measure size, CPU latency, quality drop — *done on the zero-shot MiniLM; re-run `export_onnx.py` on the trained student*
- [x] Fusion with BM25 (+ PageRank): RRF and a weighted sum tuned on dev
**Done when:** ablation table BM25 → + neural → + fusion → + reranker, with size/latency, in `results/`.

## P6 — Neural statute classifier
- [ ] Statute-description cross-attention model (statutes attend over case chunks), multi-label BCE loss — *code + CPU tests done; needs the Kaggle run*
- [ ] Compare with NB / Rocchio / kNN and LeSICiN; 3 seeds if time allows
**Done when:** row added to `results/lsi_metrics.csv` with mean ± std.

## P7 — Evaluation wrap-up
- [x] Significance tests for headline comparisons → `results/significance.csv` — *re-run after the Kaggle outputs to add the trained neural rows*
- [x] Error analysis: 5 good and 5 bad PCR queries, confused statute pairs (`results/error_analysis.json`)
- [x] `scripts/make_plots.py` → `results/fig_*.png` (default matplotlib style)
**Done when:** every number planned for the report exists in `results/`.

## P8 — Backend + frontend demo
- [x] `api/main.py` (FastAPI): `/search` (precedents + per-signal scores + matched passages), `/statutes`, `/results` (reads `results/`), `/health`
- [x] `frontend/` Next.js + TypeScript + Tailwind following `docs/FRONTEND_GUIDE.md`
- [x] Pages: landing, search/trace, evidence (results), how-it-works, team
- [x] Runs offline on the laptop; first query < ~2 s after warm-up — *latency met (BM25 ≈ 0.1 s, fusion ≈ 1.3 s on a full test judgment); engine verified with `HF_HUB_OFFLINE=1`, fonts self-hosted at build; do one rehearsal with Wi-Fi off*
**Done when:** full demo flow works end to end on real test judgments.

## P9 — Report, README, submission
- [ ] Report in research-paper format (`docs/REPORT_FORMAT.md`) → `docs/report/CaseTrail_report.pdf`
- [ ] README: setup, run, data, what works / planned, results table generated from `results/`
- [ ] `docs/AI_USE.md` final, `docs/DATA.md` credits final
- [ ] Clean-folder reproduction test from a fresh clone
- [ ] ZIP per `docs/SUBMISSION.md` — *decided 2026-11-22: the IL-TUR files are NOT put in the ZIP; include `scripts/download_data.py`, the split ids and a note on how to get the data*
- [ ] Demo + viva rehearsal — *NOT DONE YET (2026-11-22): Devansh will rehearse later, including one run with Wi-Fi off*
**Done when:** ZIP built and checked; demo rehearsed.

---

**If short on time, cut in this order:** LLM explanation → HITS features → distillation (keep the plain fine-tuned student) → cross-attention classifier. **Never cut:** evaluation, README, demo.
