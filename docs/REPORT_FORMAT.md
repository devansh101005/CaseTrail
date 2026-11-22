# Report Format (from Prof. Sonia Khetarpaul's announcement)

The report must follow the **format of a research paper** with these sections, in this order:

1. **Title of Your Project**
2. **Abstract**
3. **Introduction** — background, importance of the topic, motivation (why the topic is useful), and **our contributions**.
4. **Literature Review / Related Work** — briefly discuss the papers we read. **Cite at least 5–7 research papers** and explain the novelty of our approach.
5. **Objective**
6. **Proposed Model** — explain the process with the help of a **diagram**.
7. **Methodology** — our approach, the algorithms applied, and the relevant **formulas**.
8. **Experimentation and Results** — datasets used; results with **graphs, tables, and evaluation metrics such as precision, recall, accuracy, and F1-score**; a discussion of the results; and a **comparison with at least one existing research paper**.
9. **Conclusions and Limitations**
10. **References** — in this format (example from the announcement):
    > Zheng, Y., Zhang, L., Xie, X. and Ma, W.Y., 2009, April. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of the 18th International Conference on World Wide Web (pp. 791–800).

## Grading weight
Report is 6 of 25 points (Demo 10, Viva 5, Novelty 4).

## How we fill it (CaseTrail)
| Section | Content |
|---|---|
| Introduction | legal search problem in India, two tasks (PCR, LSI), contributions as a numbered list (distilled INT8 retriever, link analysis in legal retrieval, statute cross-attention, explainable demo) |
| Related work (≥7 papers) | IL-TUR (Joshi et al., ACL 2024); U-CREAT event-based PCR (Joshi et al., ACL 2023); LeSICiN (Paul et al., AAAI 2022); BM25 (Robertson & Zaragoza, 2009); PageRank (Page et al., 1999 / Brin & Page, 1998); HITS (Kleinberg, 1999); DPR (Karpukhin et al., EMNLP 2020); distillation for retrieval (e.g. Hofstätter et al., 2020 margin-MSE); Sentence-BERT (Reimers & Gurevych, 2019); course textbook (Manning, Raghavan & Schütze, 2008). **Verify every citation's authors, venue, year and pages before using it.** |
| Objective | 3–4 measurable objectives (beat/approach baselines, latency/size targets, explainability) |
| Proposed model | pipeline diagram (offline + online), one paragraph per box |
| Methodology | formulas: BM25, tf-idf, PageRank (with damping), HITS update rules, Naive Bayes with smoothing, Rocchio centroids, kNN, contrastive loss, distillation loss, cross-attention, metrics (P, R, F1, micro-F1@K, macro-F1) |
| Experiments | data table, setup, main results table, ablations, NB vs Rocchio vs kNN vs neural, quality vs size/latency plot, significance, **comparison with IL-TUR published baselines**, error analysis |
| Conclusions & limitations | IPC → BNS change, English-only judgments, Supreme Court only, gated dataset, not legal advice |

## Style
- Research-paper look: title block with authors + roll numbers + affiliation, abstract, numbered sections, numbered tables (caption above) and figures (caption below).
- Times New Roman (or similar serif) body, Arial headings, plain grid tables, default matplotlib figures.
- Easy language with enough technical detail; every number comes from `results/`.
- Include the **work division** and the **AI-use declaration** (short, truthful) at the end, before or after references.
