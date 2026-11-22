# CaseTrail — Project Plan

**Course:** CSD358 Information Retrieval (Prof. Sonia Khetarpaul) · course project, 25 points
**Team:** Devansh Pandey (2310110461) · Anamika Pal (2310110037) · Abhinav Bachchas (2310110383)
**Title:** CaseTrail: Precedent and Statute Retrieval for Indian Court Judgments using Link Analysis and Distilled Neural Models
**Not an extension** of the mid-term project (declared "No" on the form).

---

## 1. Problem and users
In Indian courts, lawyers and law students must (a) find earlier judgments similar to a new case (**precedents**) and (b) identify which **sections of law** apply. Today this is keyword search (Indian Kanoon, SCC Online) plus manual reading. Two problems:
- **Same facts, different words.** "harassment for dowry" vs "demand of money from the bride's family" — keyword search misses it.
- **Judgments are very long.** Strong neural models that compare two full judgments are too slow to run against thousands of candidates.

| User | Need | CaseTrail gives |
|---|---|---|
| Lawyer | arguments for a new client | similar earlier judgments + sections they may have missed |
| Law student / researcher | follow a legal topic | "judgments like this one" |
| Legal-aid NGO / junior advocate | triage many complaints | statute suggestions with confidence |
| Court researcher | consistency checks | earlier rulings on the same facts |

It is a research/search aid, **not legal advice**.

## 2. Tasks and data (IL-TUR, Joshi et al., ACL 2024)
| Task | What | Size | Published baseline (from the IL-TUR card) |
|---|---|---|---|
| **PCR** prior case retrieval | query judgment → rank candidate judgments | 827 / 118 / 237 train/dev/test queries; 4.3k / 1k / 1.7k candidates | **39.15% µF1@K** ("Event-based", Joshi et al. ACL 2023, U-CREAT) |
| **LSI** legal statute identification | judgment → subset of 100 IPC statutes (multi-label) | 43k / 10k / 13k train/dev/test + 100 statute descriptions | **28.08% macro-F1** (LeSICiN, Paul et al. AAAI 2022) |

Verify these numbers against the papers themselves before quoting them in the report.

## 3. System design
```
OFFLINE
  judgments ─► normalise + tokenise ─► inverted index ─► BM25 / tf-idf            (lectures: index, tf-idf, scoring)
           ─► graph: judgment ↔ statute (bipartite) [+ citations if present]
                 ─► PageRank (authority of a judgment)                             (lecture: PageRank)
                 ─► HITS: judgments = hubs, statutes = authorities                 (lecture: HITS)
           ─► neural student encoder (INT8 ONNX) ─► chunk vectors                  (beyond syllabus)

ONLINE (one query judgment)
  query ─► chunk / reduce ─┬─► BM25 top-200 ──┐
                           ├─► neural top-200 ─┼─► fusion (+ PageRank prior) ─► distilled reranker ─► top-K precedents
                           └─► statute classifier ─► IPC sections + confidence
  "why" panel: BM25 score · neural score · graph authority · matched passages
```

## 4. Components
### 4.1 Classical IR (written by us)
- **Text processing:** lower-casing, punctuation, legal stop words, keep section references (e.g. "Section 304B", "302 IPC") as tokens.
- **Inverted index + BM25** (k1, b tuned on train/dev) and **tf-idf cosine** as the VSM comparison.
- **Long-query handling:** a full judgment as a query is long and noisy. Compare: whole document, top-N tf-idf terms (query reduction), and paragraph-level BM25 with max/sum aggregation.
- **PageRank** on the judgment graph (power iteration, damping 0.85, dangling nodes handled) → static authority prior g(d).
- **HITS** on the bipartite judgment–statute graph → statute authority, judgment hub scores; used as features.
- **Statute classifiers:** multinomial **Naive Bayes**, **Rocchio**, **kNN** (one-vs-rest, threshold tuned on dev).

### 4.2 Neural (Devansh's strength — trained by us)
1. **Bi-encoder retriever (student):** small encoder (e.g. MiniLM / e5-small / a legal BERT), trained contrastively on PCR train pairs with **BM25-mined hard negatives**.
2. **Cross-encoder teacher → student distillation:** teacher scores (query, candidate) pairs; the student learns the teacher's scores (margin-MSE or KL). Export student to **INT8 ONNX**; report size and CPU latency.
3. **Statute cross-attention classifier:** each of the 100 statute descriptions attends over the case's chunk embeddings (multi-head cross-attention, masked pooling) → multi-label logits. Compared with NB / Rocchio / kNN and LeSICiN.

### 4.3 Modern layer for the demo (optional, only if time allows)
- **Grounded explanation:** an LLM writes a 3-line "why these cases" summary **only from retrieved passages**, and each sentence is checked against its cited passage. Must be clearly labelled and never used in evaluation numbers.

## 5. Evaluation (details in `docs/EVALUATION.md`)
- **PCR:** µF1@K (IL-TUR protocol), plus Precision@K, Recall@K, MAP, nDCG@10 for analysis.
- **LSI:** macro-F1 (main, comparable to LeSICiN), micro-F1, precision, recall.
- **Ablations:** BM25 → + query reduction → + PageRank → + neural → + reranker; NB vs Rocchio vs kNN vs neural.
- **Efficiency:** index size, model size (MB), CPU latency (median / p95).
- **Significance:** paired randomisation (or bootstrap) tests on the headline comparisons.
- **Comparison with published work:** IL-TUR / U-CREAT / LeSICiN numbers (required by the report format).

## 6. Novelty (4 points)
1. **Distilled, CPU-friendly legal retriever** (teacher → INT8 student) with a measured quality–size–latency trade-off.
2. **Link analysis inside legal retrieval:** PageRank authority + HITS on the judgment–statute graph as ranking and classification signals.
3. **Statute-description cross-attention** classifier compared head-to-head with the classical NB / Rocchio / kNN family.
4. An **explainable demo** showing every signal's contribution per result.
Write "to our knowledge"; never claim "first".

## 7. Work division
Required in the report. Filled in at the end of the project with what each team member actually did.

## 8. Rubric mapping
| Criterion | Plan |
|---|---|
| Demo (10) | Live web app: paste judgment → precedents + statutes + "why" panel, in ~1 s on a laptop CPU |
| Viva (5) | Every component is ours and explainable; `docs/IR_CONCEPTS.md` maps lectures to code |
| Report (6) | Research-paper format (`docs/REPORT_FORMAT.md`), ≥7 papers, comparison with IL-TUR baselines, graphs + tables |
| Novelty (4) | §6 |

## 9. Risks and cuts
| Risk | Fix |
|---|---|
| Citations masked in PCR text → no citation graph | Use the judgment–statute bipartite graph (from LSI-style section mentions) and similarity-kNN graph for PageRank |
| Neural training too slow | Smaller encoder, shorter max length, fewer epochs, train on Kaggle T4 |
| Long documents blow up memory | Chunk to ≤256 tokens, cap chunks per document, mean/max pooling |
| Time | Cut order: LLM explanation → HITS features in classifier → distillation (keep plain fine-tuned student) → cross-attention (keep NB/Rocchio/kNN). **Never cut** evaluation, the README, or the demo. |
