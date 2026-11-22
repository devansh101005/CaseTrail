# Evaluation Protocol

## Splits (no leakage)
- **train:** model training, hard-negative mining, classifier fitting.
- **dev:** tuning BM25 k1/b, query-reduction N, fusion weights, classification thresholds, early stopping, model selection.
- **test:** final numbers only. Run once per final system. Never tune on test.

## PCR metrics (prior case retrieval)
- **µF1@K** — the IL-TUR / U-CREAT protocol (micro-averaged F1 of the top-K retrieved candidates against the relevant set, over all queries). Implement exactly as in the paper (check whether K is fixed or the best K is reported) so the number is comparable with 39.15%.
- Also report **Precision@K, Recall@K** (K = 5, 10, 20), **MAP**, **nDCG@10**, **MRR**.
- Retrieve only from the candidate pool of the same split; exclude the query itself.

## LSI metrics (statute identification, multi-label)
- **Macro-F1** (main; comparable with LeSICiN 28.08%), **micro-F1**, macro precision and recall.
- Thresholds per method chosen on dev. Report per-statute F1 for the 10 most frequent statutes.

## Ablations
- PCR: BM25 (full text) → BM25 + query reduction → + PageRank prior → neural student → fusion → + distilled reranker.
- LSI: NB vs Rocchio vs kNN vs neural cross-attention (± HITS features).
- Efficiency: index size, model size (MB: fp32 vs INT8), CPU latency (median, p95), quality drop after quantisation.

## Statistics
- Paired randomisation test (10,000 permutations) or bootstrap on per-query scores for headline comparisons; report p-values.
- Neural models: mean ± std over 3 seeds if time allows.

## Comparison with published work (required)
Table: our best systems vs the IL-TUR published baselines (PCR 39.15% µF1@K; LSI 28.08% macro-F1), with a note that our setup follows the benchmark splits. Verify the numbers from the papers.

## Output files (all small, committed)
`results/pcr_metrics.csv`, `results/lsi_metrics.csv`, `results/efficiency.csv`, `results/significance.csv`, `results/graph_stats.json`, `results/data_stats.json`, `results/tuned_params.json`, `results/fig_*.png`.
Every CSV has the columns: `task, system, split, metric, value, n`.
