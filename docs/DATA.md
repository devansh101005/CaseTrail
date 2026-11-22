# Data, Models and Credits

## IL-TUR benchmark (main dataset)
- **Source:** https://huggingface.co/datasets/Exploration-Lab/IL-TUR
- **Paper:** Joshi, A., Paul, S., Sharma, A., Goyal, P., Ghosh, S. and Modi, A., 2024, August. IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). (verify pages before citing)
- **Licence:** CC BY-NC-SA 4.0. **Gated:** accept the conditions on the dataset page and log in (`huggingface-cli login` or `HF_TOKEN`) before downloading. Never commit a token.
- **How we load it:** not with the card's `load_dataset(..., revision="script")` (recent `datasets` no longer runs loading scripts). `scripts/download_data.py` downloads the parquet files of the main branch (`pcr/*.parquet`, `lsi/*.parquet`, 391.6 MB) into `data/raw/`, and `src/casetrail/data.py` reads them with pyarrow.
- **Files:** `pcr/{train,dev,test}_{queries,candidates}`, fields `id`, `text` (list of strings), `relevant_candidates` (list of ids; None for candidates). `lsi/{train,dev,test,statutes}`, fields `id`, `text`, `labels` (list of int class indices; the index → statute name map is the ClassLabel in the parquet metadata and equals the order of the `statutes` split, e.g. `34` → `Section 302`).

### Task `pcr` — Prior Case Retrieval (from the card)
- Indian Supreme Court judgments, ~8k documents in total.
- train / dev / test queries: 827 / 118 / 237; candidates: 4.3k / 1k / 1.7k.
- Fields: `id`, `text` (list of sentences), `relevant_candidates` (list of ids; None for candidate splits).
- Published baseline on the card: 39.15% µF1@K (event-based). Original task paper: Joshi et al., ACL 2023 (2023.acl-long.777).

### Task `lsi` — Legal Statute Identification (from the card)
- 66k Supreme Court and High Court documents, each citing statutes from a target set of 100 IPC sections.
- train / dev / test: 43k / 10k / 13k, plus a `statutes` split (100 statute descriptions).
- Published baseline on the card: 28.08% macro-F1 (LeSICiN). Original task paper: Paul et al., AAAI 2022.

### Findings from P1 (all numbers from `scripts/data_stats.py` → `results/data_stats.json`)
- [x] **Counts match the card.** PCR queries 827 / 118 / 237, candidates 4320 / 1023 / 1727. LSI 42750 / 10181 / 13019 docs, 100 statutes, every doc has ≥ 1 label.
- [x] **Citations in PCR are masked, but not completely.** Case citations are replaced by `<CITATION>` (33,171 tags in train candidates); names/courts by `<NAME>`/`<ORG>`. Two kinds of leftovers remain:
  - unmasked reporter citations in the body (e.g. `AIR 1974 SC 753`, `[1962] 3 SCR 618`);
  - the SCR header of about a third of the judgments (`CITATION: 1966 AIR 1053 …`, `CITATOR INFO : F 1967 SC 819 …`, i.e. the judgment's own citation and the later cases that cite it).
  Matching these strings recovers **14.1% / 15.6% / 16.5%** of the relevant (query, candidate) pairs in train / dev / test without any retrieval. This is label leakage, so the text pipeline strips header citation fields and leftover reporter citations before indexing (`text/citations.py`, DECISIONS change log). A full citation graph is therefore **not available** (see DECISIONS D).
- [x] **Queries sit inside their own candidate pool.** All 827 / 118 / 237 queries are also candidates of the same split; no query lists itself as relevant → retrieval must skip the query itself. 10 (train) and 3 (test) relevant ids are not in the pool (they count as misses; recall cannot reach 1).
- [x] **Lengths.** PCR documents average about 6.4k–8.7k words (median ~4k), with outliers up to 415k words, so chunking/query reduction is required. LSI documents are shorter (mean ~1.2k words, median ~450–560). The LSI `text` items are paragraphs rather than sentences, and some spaces are missing at old line breaks (`theprosecution`).
- [x] **LSI statutes are masked** in the text as `<SECTION>` / `<ACT>`, and parties as `<ENTITY>`. So statute numbers cannot be read off the text (no label leakage), and the `sec_*` protected tokens only matter for PCR, whose texts keep section references.
- [x] **LSI labels are skewed.** Mean 3.7 labels per train doc (max 31). The most frequent are Section 34 (11,731 train docs), 302 (10,380), 506, 323, 307; the rarest are Section 450 (272), 200 (287), 190 (304).
- [x] **Overlap between splits.**
  - PCR query sets are disjoint across splits, and no labelled (query, relevant) pair of test appears in train/dev. The candidate pools do share old judgments by design: 809 test-pool ids also occur in train/dev pools, with identical text. 170 test queries appear as *candidates* in the train pool.
  - LSI ids are disjoint; 233 test docs (1.8%) are near-duplicates of LSI train docs (≥ 50% shared text shingles). We keep the benchmark splits unchanged for comparability and mention this.
  - **PCR and LSI do not overlap:** 0 PCR test-pool docs are near-duplicates of LSI train docs, and only 1 PCR train-pool doc is a near-duplicate of an LSI test doc.

## Models
| Model | Use | Size | Licence | Source |
|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych, 2019; MiniLM: Wang et al., 2020) | student candidate, zero-shot dense baseline | 91.6 MB | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| intfloat/e5-small-v2 (Wang et al., 2022, "Text Embeddings by Weakly-Supervised Contrastive Pre-training") | student candidate, zero-shot dense baseline | 134.4 MB | MIT | https://huggingface.co/intfloat/e5-small-v2 |
| BAAI/bge-reranker-base (Xiao et al., 2023, C-Pack / BGE) | cross-encoder teacher (fine-tuned on Kaggle) | about 1.1 GB, on Kaggle only | MIT | https://huggingface.co/BAAI/bge-reranker-base |

Sizes are those printed by `scripts/download_models.py`; the teacher size is from its model page (verify on download). Verify every licence on the model card before the final submission.

## Libraries
Listed in `requirements.txt` and `frontend/package.json`.
- Pipeline: numpy, pandas, pyarrow (parquet reading), huggingface_hub (download), PyTorch, Hugging Face Transformers, ONNX + ONNX Runtime, FastAPI + Uvicorn, matplotlib.
- Reference-only (used in `tests/` only, never in the pipeline): rank_bm25, scikit-learn, networkx. pytest for tests.

## What goes into the submission ZIP
The professor asks for the dataset in the ZIP. Because IL-TUR is gated and NC-SA licensed, include: the **download script**, the **exact split ids/files we used** if the licence allows redistribution for this non-commercial course use, and a note explaining how to get the full data. Decide this in P9 after re-reading the licence and the dataset conditions.
