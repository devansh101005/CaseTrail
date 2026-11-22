# CaseTrail — Decisions, Stack and Trade-offs

Each decision: **what we chose**, the alternatives, **why**, and the **trade-off we accept**. These are the "why did you choose X?" viva answers. Add a new entry whenever a decision changes (date it).

---

## A. Stack

| # | Area | Chosen | Alternatives considered | Why | Trade-off accepted |
|---|---|---|---|---|---|
| A1 | Core IR language | **Python + numpy** | Java (Lucene), Elasticsearch, Pyserini | Course rule: we implement the IR ourselves; numpy postings are fast enough for pools of a few thousand docs; everyone can read Python | Slower than Lucene at scale; fine for ~8k PCR docs and 66k LSI docs |
| A2 | Index storage | **numpy arrays + JSON on disk** | SQLite, Postgres FTS, a vector DB | Transparent, matches the lecture's dictionary/postings model, zero services to run | No incremental updates; rebuilt by a script (acceptable for a static benchmark) |
| A3 | Dense search | **brute-force numpy dot product** | FAISS, HNSW, pgvector | Candidate pools are small (≤ ~5k docs × ≤ 32 chunks), so exact search is milliseconds; no extra dependency; exact results | Would not scale to millions of documents (mention as future work: cluster pruning / ANN) |
| A4 | Deep learning | **PyTorch + Hugging Face Transformers** | TensorFlow, JAX | Devansh's experience (LightDep, BioX-DTI); largest model ecosystem | Heavier install; CPU-only torch on the laptop |
| A5 | Inference | **ONNX Runtime, INT8 dynamic quantisation** | plain PyTorch CPU, OpenVINO, TensorRT | 2–4× smaller and typically faster on CPU; same recipe Devansh used in LightDep; portable | Small accuracy drop (measured and reported); export step to maintain |
| A6 | Training compute | **Kaggle free T4 GPUs** (+ laptop GTX 1650 4 GB for small runs) | Colab, paid cloud | Free, ~30 GPU-hours/week, enough for small encoders | Session limits, data upload step; keep notebooks restartable |
| A7 | Backend | **FastAPI + Uvicorn** | Flask, Django, Streamlit | Simple request/response models, automatic docs at `/docs`, plain `def` routes are enough; Devansh used it in VidhiVault | One more process to run in the demo |
| A8 | Frontend | **Next.js (App Router) + TypeScript + Tailwind** | Streamlit, plain HTML/JS, Gradio | Needed for the Waypoint design language and a polished demo (Demo = 10 points); fonts self-hosted for offline use; we already know React/Next.js from Waypoint | Second tech stack (Node + Python); more setup — start early (P2) |
| A9 | Charts | **hand-written SVG components** | Plotly, Chart.js | Full control over the Waypoint look; tiny bundle | More code per chart |
| A10 | Persistence of results | **CSV/JSON/PNG in `results/`** | a database, MLflow | Simple, diffable, committed to git, read directly by README/report/frontend | No experiment-tracking UI |
| A11 | Classical classifiers | **our own NB / Rocchio / kNN** | scikit-learn | Course rule and viva; sklearn used only as a reference check in tests | More code to test |

## B. Design choices

| # | Decision | Alternatives | Why | Trade-off |
|---|---|---|---|---|
| B1 | **BM25 as the lexical baseline** (tf-idf kept for comparison) | tf-idf only, query likelihood | BM25's tf saturation and length normalisation suit long legal documents; standard strong baseline | Two parameters to tune on dev |
| B2 | **Query reduction** for whole-judgment queries | use the full text; LLM-written summaries | Full judgments make huge, noisy queries; top tf-idf terms keep the discriminative ones and are explainable | May drop useful rare context — compared on dev |
| B3 | **Chunking (≤256 tokens) + MaxSim-style aggregation** | truncate to first 512 tokens; long-context models | Small encoders have short windows; facts are spread through judgments | More vectors per doc; more compute at encode time |
| B4 | **Bi-encoder student + distillation from a cross-encoder teacher** | cross-encoder only; bi-encoder only; large off-the-shelf embedding model | Cross-encoder is accurate but too slow for whole pools; distillation keeps much of its quality at bi-encoder speed; strong viva story (Devansh's LightDep expertise) | Extra training stage; cuttable if time is short |
| B5 | **BM25-mined hard negatives** | random / in-batch negatives only | Teaches the model the hard cases keyword search gets wrong; ties classical IR to training | Risk of false negatives (unlabelled relevant cases) — keep only negatives ranked below the relevant ones |
| B6 | **Fusion: RRF first, then weighted sum tuned on dev** | learning-to-rank model | RRF is robust and parameter-light; weighted sum is explainable for the "why" panel | LTR could be stronger; possible future work |
| B7 | **PageRank as a static prior; HITS on the judgment–statute graph** | no graph; GNNs | Directly uses the post-mid-term lectures; cheap; explainable authority scores | Depends on usable links (P1 finding); may add little quality — still reported honestly |
| B8 | **Statute cross-attention classifier** | flat BCE head on [CLS]; LeSICiN-style graph model | Uses statute descriptions as information; same mechanism Devansh built in BioX-DTI; explainable per-statute attention | More parameters than a flat head; needs class weighting for rare statutes |
| B9 | **One-vs-rest NB/Rocchio/kNN with dev-tuned thresholds** | top-k fixed predictions | Multi-label task; thresholds optimise macro-F1 fairly | Threshold tuning adds a step |
| B10 | **Metrics: µF1@K (PCR) and macro-F1 (LSI)** as headline | only MAP/nDCG | Required for comparison with the IL-TUR published baselines; P/R/F1 also required by the report format | µF1@K depends on K — follow the paper's protocol exactly |
| B11 | **No LLM in the core pipeline or evaluation** | LLM reranking / judging | Reproducible, free, offline, and the IR is ours; an LLM explanation is an optional demo layer only | Less "flashy" — compensated by the trained models |
| B12 | **Benchmark data instead of crawling** | crawl Indian Kanoon / SCI website | Judged relevance labels are essential for evaluation; crawling legal sites raises terms-of-use issues | The crawler lecture is not used in the system |

## C. Quality vs speed vs size (what we will measure)
| System | Expected quality | Expected latency | Size |
|---|---|---|---|
| BM25 (+ query reduction) | baseline | fastest | index only |
| + PageRank prior | ≈ baseline (small change) | ~same | + one array |
| Student bi-encoder (INT8) | higher on "same meaning, different words" | moderate | ~20–30 MB |
| Fusion + distilled reranker | best | slowest of ours | + reranker |
| Teacher cross-encoder (reference only) | upper bound | too slow for full pools | hundreds of MB |
The final table in the report shows the actual numbers from `results/efficiency.csv` and `results/pcr_metrics.csv`.

## D. Open decisions (resolve during the phases)
- [x] Student model choice → **e5-small-v2** (zero-shot dev micro-F1@K 0.3526 vs MiniLM-L6 0.3357, Kaggle epoch 0; change log 2026-11-22).
- [ ] Teacher model choice — decide in P5 on Kaggle GPU memory.
- [x] Citation graph available? (P1) → **No full citation graph.** Citations are masked as `<CITATION>`. Only partial leftovers survive (reporter strings, SCR headers), and they are exactly the relevance labels, so we strip them (see change log). P3 builds **G2** (judgment ↔ section-mention bipartite graph; PCR texts keep section references) and **G3** (BM25 kNN similarity graph). A "partial G1" made from leftover citations is **not** used for ranking: queries are part of every candidate pool, so its edges would include query → relevant-candidate links.
- [x] Stemming on/off (P2, dev) → **on** (our light S-stemmer): best unigram BM25 dev µF1@K 0.3255 with stemming vs 0.3214 without (`results/pcr_dev_tuning.csv`). A small but consistent gain, so it is used everywhere (PCR and LSI).
- [x] Dataset files allowed in the ZIP? → **No** (decided by Devansh, 2026-11-22). The ZIP gets the download script, the exact split ids and instructions; IL-TUR is gated and CC BY-NC-SA, so redistributing it is avoided.

## Change log
| Date | Change | Reason |
|---|---|---|
| 2026-11-21 | Laptop uses **CPU-only torch**; all GPU training on Kaggle (the GTX 1650 is not used) | The CUDA wheel is ~2.5 GB and the 4 GB card is too small to matter; keeps setup simple and the demo CPU-only (A5, A6) |
| 2026-11-21 | Data downloaded with `huggingface_hub.snapshot_download` (parquet files of `pcr/` and `lsi/` only) and read with pandas/pyarrow, **not** `load_dataset(..., revision="script")` | `datasets` ≥ 4.0 no longer runs dataset loading scripts; the Hub's main branch already holds plain parquet files (~392 MB for the two tasks); no remote code is executed; one dependency fewer |
| 2026-11-21 | Minimal `pyproject.toml` added so `pip install -e .` makes `casetrail` importable | Scripts, tests and the API all import the same package without `sys.path` hacks |
| 2026-11-21 | PCR: the cut-off **K for µF1@K is chosen on dev**, then applied unchanged to test | If the published protocol reports the best K on test, copying that would tune on test (CLAUDE.md rule 3); the difference is stated in the report |
| 2026-11-21 | Published baseline numbers (39.15, 28.08) are written to `results/published_baselines.json` by a script, with the citation | Rule 2: every number shown in README/report/frontend comes from `results/` |
| 2026-11-21 | Porter stemmer (if used) is our own implementation | Rule 4 spirit + no extra dependency |
| 2026-11-21 | **Strip leftover citations before indexing** (SCR `CITATION:` / `CITATOR INFO` header fields and unmasked reporter citations → `<CITATION>`), in `text/citations.py` | P1 found that these strings alone recover 14–17% of the relevant PCR pairs. Keeping them would let BM25 match citation numbers, i.e. read the labels. Trade-off: our numbers may be slightly lower than systems that did not clean this, and the report says so |
| 2026-11-21 | No-leakage rule for PCR changed from "no test id in train/dev" (LLD) to "**query sets disjoint and no test/dev (query, relevant) pair in train**" | The benchmark's candidate pools share old judgments by design (809 test-pool ids also occur in train/dev pools, identical text); what must not leak is test queries and their labels. Enforced in `tests/test_data.py` |
| 2026-11-21 | Keep the benchmark splits unchanged despite 233 near-duplicate LSI test docs (1.8%) in LSI train | Comparability with the published baselines; reported as a limitation |
| 2026-11-21 | Light stemmer = our own Harman-style S-stemmer (+ -ing/-ed/-ly), not Porter | Short and explainable in the viva; dev decides whether stemming helps at all (it does, slightly) |
| 2026-11-21 | **Word bigrams added as index terms** (bigrams in only one judgment of the pool are pruned) | U-CREAT/IL-TUR report BM25 jumping from 13.85 to 28.59 with bigrams; on our dev the bigram system is the best lexical one (µF1@K 0.4525 vs 0.3803 unigram). Pruned bigrams can never link two pool judgments; trade-off: ~4× larger index |
| 2026-11-21 | BM25 grid widened to k1 ≤ 10 after the first search chose the edge (k1 = 3.0) | A best value on the grid edge means the grid is too small. Unigram BM25 then chose k1 = 10, b = 1.0, i.e. almost no tf saturation and full length normalisation; we stop there, since k1 → ∞ just makes the tf part linear |
| 2026-11-21 | Coordinate-wise grid search (query setting → k1/b → passages) instead of the full grid | Same dev-only rule, a fraction of the runs, and each stage is easy to explain |
| 2026-11-21 | Citation graph built from **train relevance labels only** (train query → cited candidate) and used as a PageRank prior for dev/test candidates | The candidate pools share old judgments, so "how often was this judgment cited in training" is a legitimate authority signal. It uses no dev/test labels |
| 2026-11-21 | Link-analysis priors kept in the system and reported, although dev chose λ = 0 (or ≈ 0) for all four | Honest negative result. Static PageRank (citation-from-train, similarity graph), the HITS hub prior and query-dependent HITS do not improve tuned bigram BM25 on dev, and the tiny dev gain of the citation prior (+0.003) turns into −0.007 on test |
| 2026-11-21 | HITS statute authority is **not** added as a prior to the LSI classifiers | Thresholds are tuned per statute, and any per-statute constant (like log authority) is absorbed exactly by its threshold, so it cannot change a decision. It is reported in `graph_stats.json` and used in the demo explanation only |
| 2026-11-21 | LSI classical grids widened (NB α down to 0.001, kNN k up to 100) after the first search chose the edges (α = 0.01, k = 50) | Same rule as for BM25: a best value on the grid edge means the grid is too small; still dev only |
| 2026-11-21 | Neural training split into **tested scripts + a minimal Kaggle notebook** that downloads IL-TUR itself (HF_TOKEN Kaggle secret) | Only a small code zip is uploaded; the notebook writes rankings/score matrices only, and every metric is computed locally (`scripts/eval_kaggle_runs.py`). No test labels go to Kaggle |
| 2026-11-21 | Teacher = `BAAI/bge-reranker-base` (fine-tuned), with `cross-encoder/ms-marco-MiniLM-L-12-v2` as the smaller fallback | Already a strong re-ranker, fits a T4 at 512 tokens, MIT licence; about 1.1 GB, so it is downloaded on Kaggle, never on the laptop (rule 7) |
| 2026-11-22 | Student candidates (MiniLM-L6, e5-small) are compared **zero-shot on dev as "epoch 0" of `train_student.py` on Kaggle**; the better one is trained. Locally only MiniLM is encoded (the demo model) | Laptop CPU encoding measured 15.7 chunks/s (`indexes/dense/.../doc_names.json`, with other jobs running): about 25 min per pool for MiniLM, so e5-small (twice as deep) is not worth laptop time. Epoch 0 also gives every training curve a measured zero-shot starting point. Note: on Kaggle e5 is scored without its "query: " prefix (same text treatment for both candidates) |
| 2026-11-21 | Chunks are packed without overlap (`CHUNK_STRIDE` unused) | Simpler; MaxSim over ≤ 32 evenly spaced chunks already covers long judgments |
| 2026-11-21 | Heavy jobs run **one at a time** on the laptop | 16 GB RAM is shared with other apps; running encoding, LSI kNN and training together ran out of memory (P5 first attempt) |
| 2026-11-21 | Frontend: Next.js 16 + React 19 + Tailwind 4, TypeScript pinned to 5.9 | npm's default TypeScript 7 (native compiler) may not be supported by Next's type check; 5.x is the safe choice |
| 2026-11-21 | Demo shows "cited in the real judgment" when the pasted text is an IL-TUR test query | Display only (benchmark labels are never used for ranking); lets the audience see hits and misses live |
| 2026-11-22 | INT8 export uses **per-channel** dynamic quantisation | Per-tensor INT8 failed the LLD fidelity check (mean cosine vs fp32 0.946 on 200 dev chunks); per-channel gave 0.993 (min 0.985) at the same 23 MB. Measured with a probe of five settings; chosen on dev chunks only |
| 2026-11-22 | Query terms that occur only in the query judgment itself (df = 1, because benchmark queries are also pool members) are **kept** in ranking, but hidden in the demo's explanation panel | Found while checking the demo. A dev-only probe (bigram BM25 at the tuned k1, b; top 200–1600 terms; with/without dropping df < 2) moved dev micro-F1@K by at most 0.4 points (0.4525 → 0.4549), within noise on 118 queries, so the tuned pipeline stays. The same probe shows 800 terms at the final k1, b scoring 0.4569, a limit of the coordinate-wise search (top_terms was chosen at default k1, b) |
| 2026-11-22 | **Student = intfloat/e5-small-v2**, chosen on dev by the Kaggle epoch-0 run: zero-shot dev micro-F1@K 0.3526 vs 0.3357 for MiniLM-L6 | Resolves the open "student model choice" decision on dev. MiniLM scored 0.3357 on both the laptop and Kaggle, so the pipeline reproduces across machines. Logs: `results/training_student_zeroshot_*.json` (come back with the Kaggle outputs). e5 was scored without its "query: " prefix, the same text treatment as MiniLM |
| 2026-11-22 | Kaggle batch sizes: teacher 2 judgment pairs (18 chunk pairs × 512 tokens), students 8 queries | The teacher at batch 8 (72 sequences of 512 tokens + AdamW states for 278M weights) ran out of the T4's memory (CUDA OOM at 14.48 / 14.56 GiB). Smaller batches see the same data and epochs in more steps |
| 2026-11-22 | Running the ONNX encoder moved to `neural/onnx_runtime.py` (imports only onnxruntime); export/quantisation stays in `neural/export_onnx.py` | Windows Smart App Control started blocking the `ml_dtypes` DLL (pulled in by `onnx` via `onnxruntime.quantization`), and the demo backend failed to start. The demo never needs the export tools, so it no longer imports them |
| 2026-11-22 | **Teacher run 2: selected chunk pairs** (`--select-with` e5-small, the 4 most similar chunk pairs) + 10% learning-rate warm-up + a dev pair-accuracy check; outputs tagged `_sel` (`notebooks/train_teacher_selected.ipynb`) | Run 1's teacher (3 x 3 random chunk pairs) stayed at loss ln 2 = 0.693 for 3,600+ steps: random parts of two long judgments rarely contain the point that links them, so the labels looked like noise. Run 1 is kept as is (its plain student and statute attention do not use the teacher) and reported as the random-pair ablation |
| 2026-11-22 | Run 2 students (`plain_sel`, `distilled_sel`) train on **candidate chunks that match the query's chunks** (`--select-with` e5-small), not 4 random chunks | Run 1's plain student fell below zero-shot after epoch 1 (dev micro-F1@K 0.3526 → 0.3391): training on random candidate chunks does not match evaluation, where MaxSim lets each query chunk find its best chunk among all of them. Query chunks stay random so every part of the query is trained |
| 2026-11-22 | Clean-clone reproduction (partial) passed: setup, data stats, indexes, dev tuning and PCR evaluation regenerate byte-identical results | Checks rule 2 end to end on a fresh clone; LSI, graphs and neural steps still to repeat before submission |
