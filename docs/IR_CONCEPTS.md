# IR Concepts → Code (keep updated after every phase)

Lectures in `E:\001IR\Lectures`. The viva will ask "where is X in your code and why".

| Lecture | Concept | Where in CaseTrail | Status |
|---|---|---|---|
| 01 Introduction | IR system, documents, queries, relevance | collection, queries and relevance judgments loaded in `src/casetrail/data.py`; corpus statistics in `scripts/data_stats.py`; engine `src/casetrail/search.py` (planned) | data: done (P1) |
| 02 Vocabulary | normalisation of noisy text (citation strings → one `<CITATION>` symbol) | `src/casetrail/text/citations.py` | done (P1) |
| 02 Vocabulary | tokenisation, special tokens (`sec_498a`, `art_21`), stop words (English + legal boilerplate from train df), light stemming | `text/tokenizer.py`, `text/legal_refs.py`, `text/stopwords.py` (+ `scripts/top_df_terms.py`), `text/stemmer.py` | done (P2) |
| 08 Evaluation | test collections, relevance judgments, no train/test leakage | `tests/test_data.py`, leakage + overlap findings in `docs/DATA.md`, `scripts/ablation_citation_strip.py` | done (P1/P2) |
| 06 tf-idf | tf, idf, (1 + log tf)·log(N/df), cosine similarity in the vector space model | `retrieval/bm25.py` (`tfidf_doc_norms`, `tfidf_cosine_scores`), `classify/rocchio.py`, `classify/knn.py` | done (P2) |
| Scoring & result assembly | inverted index (dictionary + postings), term-at-a-time BM25 accumulators, heap top-K, query reduction, phrase information via bigrams, passage scoring | `index/inverted_index.py`, `retrieval/bm25.py`, `retrieval/chunking.py`, `retrieval/pcr_lexical.py`, `scripts/build_index.py`, `scripts/tune_pcr_lexical.py` | done (P2) |
| Web crawler | crawling, politeness (not used: data is a benchmark; mention why in report) | — | n/a |
| 09 Reputation systems | authority / trust signals: "cited often in training" as a reputation prior | `scripts/eval_graph_prior.py` (citation-graph PageRank prior) | done (P3) |
| 10 PageRank | random surfer, damping, power iteration, dangling nodes, weighted links | `graph/pagerank.py` (+ tests vs hand computation and networkx), graphs in `graph/build_graph.py`, `scripts/build_graph.py` | done (P3) |
| HITS | hubs and authorities; global on the bipartite judgment–reference / judgment–statute graphs, and query-dependent on each query's root set (Kleinberg) | `graph/hits.py`, `scripts/build_graph.py`, `scripts/eval_local_hits.py` | done (P3) |
| 08 Evaluation | precision, recall, F1, micro-F1@K (IL-TUR), P@K, R@K, MAP, nDCG, MRR, macro/micro F1 | `eval/metrics.py` (+ `tests/test_metrics.py`) | done (P2) |
| 08 Evaluation | statistical significance: paired randomisation tests (per-query and pooled metrics) | `eval/significance.py`, `scripts/significance.py` | code done (P7) |
| 13 Naive Bayes | multinomial NB, add-α smoothing, log-odds, one-vs-rest multi-label, per-class thresholds | `classify/naive_bayes.py`, `classify/thresholds.py`, `scripts/run_lsi_classical.py` (+ test vs sklearn) | done (P4) |
| 14 Vector space classification | Rocchio (unit-length tf-idf centroids, cosine), kNN (neighbours via the inverted index, similarity-weighted votes) | `classify/rocchio.py`, `classify/knn.py` | done (P4) |
| Beyond syllabus | dense retrieval with chunked MaxSim, InfoNCE with BM25 hard negatives, cross-encoder teacher + Margin-MSE distillation, INT8 ONNX quantisation, statute cross-attention | `neural/encoder.py`, `neural/train_retriever.py`, `neural/distill.py`, `neural/export_onnx.py`, `neural/statute_attention.py`, `notebooks/train_retriever.ipynb` | code + CPU smoke tests done; training waits for Kaggle |
| Beyond syllabus | rank fusion: reciprocal rank fusion, weighted sum of normalised scores | `fusion/combine.py`, `scripts/eval_dense_fusion.py` | done (code) |
| 01 Introduction | the IR system end to end: query → ranked results with explanations | `search.py` (CaseTrailEngine), `api/main.py`, `frontend/` | built (P8) |
