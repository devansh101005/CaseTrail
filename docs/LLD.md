# CaseTrail — Low-Level Design (LLD)

Concrete formats, algorithms, defaults and interfaces. Defaults marked *(tune on dev)* are starting values, not final.
Code style follows our coding rules (plain functions/dicts, simple classes, no dataclasses).

---

## 1. Configuration (`src/casetrail/config.py`)
```
PROJECT_ROOT, DATA_DIR = data/, RAW_DIR = data/raw/, INDEX_DIR = indexes/, MODEL_DIR = models/, RESULTS_DIR = results/
SEED = 13
BM25_K1 = 1.2, BM25_B = 0.75                       (tune on dev)
QUERY_TOP_TERMS = 50                               (tune on dev; 0 = full text)
CANDIDATES_PER_SIGNAL = 200, RERANK_TOP = 50, FINAL_K = 10
CHUNK_TOKENS = 256, CHUNK_STRIDE = 192, MAX_CHUNKS_PER_DOC = 32
PAGERANK_DAMPING = 0.85, PAGERANK_TOL = 1e-8, PAGERANK_MAX_ITER = 100
HITS_MAX_ITER = 100, HITS_TOL = 1e-8
NB_ALPHA = 1.0, KNN_K = 25                         (tune on dev)
RRF_K = 60
API_HOST = 127.0.0.1, API_PORT = 8000
```

## 2. Data model (in memory)
Plain dicts, loaded by `data.py`:
```
pcr[split] = {
  "queries":    {doc_id: {"id": str, "sentences": [str], "relevant": [cand_id]}},
  "candidates": {cand_id: {"id": str, "sentences": [str]}}
}
lsi[split] = [ {"id": str, "sentences": [str], "statutes": [statute_id]} ]
statutes   = {statute_id: {"id": str, "title": str, "text": str}}
```
`text = " ".join(sentences)`; sentence boundaries are kept for chunking and passage highlighting.
`statute_id` is the statute name (`"Section 302"`); LSI integer labels are mapped through the parquet ClassLabel names.
Every PCR query is also in its own split's candidate pool, so retrieval skips the query itself.
No-leakage tests (`tests/test_data.py`) assert that query sets are disjoint across splits and that no test/dev (query, relevant) pair appears in train. Candidate pools are allowed to share judgments, because the benchmark does this by design (P1 finding, DECISIONS change log).

## 3. Text pipeline (`text/`)
0. **Strip leftover citations** (`text/citations.py`, before anything else): remove SCR header `CITATION:` / `CITATOR INFO` fields and unmasked reporter citations (they leak PCR labels; P1 finding). Mask tags (`<CITATION>`, `<NAME>`, `<ORG>`, `<ENTITY>`, `<SECTION>`, `<ACT>`) are dropped as tokens.
1. Unicode NFC, lower-case.
2. Protect legal references before tokenising: regex for `section 302`, `s. 302`, `302 ipc`, `sec. 498-a`, `article 21` → canonical tokens `sec_302`, `sec_498a`, `art_21`.
3. Tokenise with `[a-z0-9_]+`; drop tokens of length 1 except digits inside protected tokens.
4. Stop words: English list + legal boilerplate (`hon'ble`, `learned`, `counsel`, `appellant`, `respondent`, `petitioner`, `court`, `judgment`) — the boilerplate list is chosen from the highest-df terms in train and documented.
5. Optional light stemming (Porter, our own implementation or documented library) — compare on dev.

## 4. Inverted index (`index/inverted_index.py`)
```
class InvertedIndex:
    term_to_id: dict[str, int]
    df: numpy int32 [V]
    offsets: numpy int64 [V+1]           # postings of term t = slice offsets[t]:offsets[t+1]
    doc_ids: numpy int32 [P]             # sorted within each term
    tfs: numpy int32 [P]
    doc_len: numpy int32 [N]; avg_len: float
    doc_names: list[str]                 # internal id -> IL-TUR id
    save(folder) / load(folder)          # numpy.save + json
```
Build: count terms per document → sort by (term, doc) → fill arrays. One index per candidate pool (train, dev, test).

## 5. Lexical retrieval (`retrieval/`)
**BM25** (term-at-a-time accumulation into a numpy score array, then heap top-K):
```
idf(t)   = ln(1 + (N − df_t + 0.5) / (df_t + 0.5))
score(d) = Σ_t  qtf_t · idf(t) · tf_{t,d}·(k1+1) / (tf_{t,d} + k1·(1 − b + b·|d|/avgdl))
```
**tf-idf cosine** (VSM comparison): `w = (1+log tf)·log(N/df)`, cosine with length-normalised vectors.
**Query reduction** for long query judgments: keep the top-`QUERY_TOP_TERMS` terms by `tf·idf` in the query (0 = all terms).
**Bigrams** (`ngram = 2`): word bigrams `a__b` are added after the unigrams; bigrams occurring in only one judgment of a pool are pruned. **Binary query weights** (`qtf = 1`) are an option for very long queries.
**Tuning** (`scripts/tune_pcr_lexical.py`, dev only): coordinate-wise grid (query setting and stemming → k1 ∈ [0.6, 10], b ∈ [0.3, 1] → passages); K of µF1@K chosen on dev in 1..20. Chosen values live in `results/tuned_params.json`; `config.py` keeps only the textbook defaults.
**Paragraph mode** (`retrieval/chunking.py`): passages = consecutive text items with ≥ 120 words; score the query against passages; doc score = max (or sum of top-3) passage scores. Compared on dev.

## 6. Link analysis (`graph/`)
**Graphs** (adjacency as `dict[int, list[int]]` + weights):
- G1 citation graph: **not built.** P1 found citations masked; the leftovers equal the labels (see DATA.md).
- G1′ citation graph from **train labels**: train query → each relevant candidate; PageRank = how often/where a judgment was cited in training. Used as a prior for dev/test candidates that also occur in it (pools share judgments).
- G2 bipartite judgment–reference graph: edge if a PCR judgment mentions a protected `sec_*`/`art_*` token (weight = count; references in < 2 judgments dropped). For LSI: judgment → statute from LSI train labels.
- G3 fallback similarity graph — each judgment → its top-10 BM25 neighbours (directed, weights = normalised score).

**PageRank** (power iteration):
```
r = 1/N
repeat:
    r_new[v] = (1−d)/N + d · ( Σ_{u→v} r[u]·w(u,v)/W(u)  +  Σ_{dangling u} r[u]/N )
until  ||r_new − r||_1 < tol  or  max_iter
```
**HITS** on G2 (judgments as hubs, statutes as authorities):
```
a = 1, h = 1
repeat: a = Aᵀ h ; h = A a ; L2-normalise a and h ; until change < tol
```
Outputs: `pagerank[doc]`, `hub[doc]`, `authority[statute]` → `indexes/graph/*.npy` + top lists in `results/graph_stats.json`.
Use in ranking (`scripts/eval_graph_prior.py`): re-rank the top 100 by `bm25/max_bm25 + λ·z(log prior)` (z = standardised over the pool), λ and K tuned on dev. Query-dependent HITS (`scripts/eval_local_hits.py`): HITS on the G3 edges inside each query's top-50, `bm25/max + λ·authority/max`. Use in LSI: statute authority is reported and shown in the demo only, because per-statute thresholds absorb any per-statute constant.

## 7. Statute classifiers (`classify/`)
All one-vs-rest over 100 statutes; per-statute threshold (or a global top-k + threshold) chosen on dev to maximise macro-F1.
- **Multinomial NB** (per statute, classes {c, ¬c}):
  `log P(c|d) ∝ log P(c) + Σ_t tf_{t,d} · log ((count(t,c)+α) / (Σ_t' count(t',c) + α·V))`, α = `NB_ALPHA`; vocabulary pruned to the top 50k terms by df.
- **Rocchio**: centroid μ_c = mean of tf-idf vectors of docs labelled c; score = cos(d, μ_c).
- **kNN**: top-`KNN_K` training docs by cosine; score(c) = Σ_{neighbours with label c} similarity.

## 8. Neural models (`neural/`, trained in `notebooks/` on Kaggle)
**Chunking:** text items (split into ≤ 150-word pieces) packed greedily into windows of ≤ `CHUNK_TOKENS` word-piece tokens, **without overlap** (simpler; `CHUNK_STRIDE` is unused); ≤ `MAX_CHUNKS_PER_DOC` chunks (the first, then evenly spaced). Citations are stripped and mask tags removed first, as in the lexical pipeline. LSI documents use ≤ 16 chunks.

**Student bi-encoder** (candidate: MiniLM-L6 / e5-small class, ~22–33M params; final choice in `DECISIONS.md`):
- chunk embedding = mean pooling + L2 norm; document score = mean over query chunks of max over candidate chunks of cosine (MaxSim-style aggregation) *(compare with mean-of-chunks on dev)*.
- **Training:** InfoNCE, temperature τ = 0.05; per query: 1 positive (relevant candidate), in-batch negatives + 4 **BM25 hard negatives** (top BM25 candidates that are not relevant); lr 2e-5, batch 16 pairs, 3 epochs, fp16 on T4 *(tune on dev)*.

**Teacher cross-encoder** (larger encoder, e.g. a base-size BERT/legal BERT): input `[CLS] query_chunk [SEP] cand_chunk`; fine-tuned on PCR train pairs with BCE; document score = max over chunk pairs (pairs capped).
**Distillation:** student learns teacher margins (Margin-MSE): `L = MSE( s_S(q,p) − s_S(q,n), s_T(q,p) − s_T(q,n) )`, optionally + InfoNCE.
**Export:** `torch.onnx.export` (opset ≥ 17, dynamic batch/seq axes) → `onnxruntime.quantization.quantize_dynamic` (INT8). Check cosine(fp32, int8) ≥ 0.98 on 200 dev chunks.

**Statute cross-attention classifier:**
```
H = student chunk embeddings of the case            [n_chunks × d]   (frozen or lightly fine-tuned)
S = embeddings of the 100 statute descriptions      [100 × d]        (learned projection)
A = MultiHeadAttention(query=S, key=H, value=H, heads=4, key_padding_mask)   [100 × d]
logit_c = w · LayerNorm(A_c + S_c) + b              → 100 logits
loss    = BCEWithLogits with pos_weight = (neg/pos) per statute (clipped)
```

## 9. Engine and API contract
**Engine** (`search.py`):
```
engine = CaseTrailEngine(split="test")           # loads index, graph, ONNX model, classifiers
engine.search(text, system="bm25_bigram", k=10)  # systems: bm25_bigram, neural, fusion (if the dense artifacts exist)
engine.predict_statutes(text, method="nb")       # methods: nb, rocchio, knn (neural attention: evaluated offline, P6)
engine.available()                               # what is loaded; missing artifacts just remove a system
```
**API** (`api/main.py`, FastAPI, JSON):
```
GET  /health                → {"status": "ok", "systems": [...], "methods": [...], "loaded": {...}}
POST /search                body {"text": str, "system": str = "full", "k": int = 10}
  → {"query_terms": [{"term", "df", "idf"}],
     "system", "known_query_id" (set when the pasted text is an IL-TUR test query), "num_query_terms",
     "results": [{"rank", "doc_id", "title", "snippet", "matched_passages": [str],
                  "is_relevant_in_benchmark" (display only, from the benchmark labels; null for new text),
                  "breakdown": {"bm25", "neural", "fusion", "pagerank_citation", "pagerank_similarity"}}],
     "timings_ms": {"tokenise", "bm25", "neural", "total"}}
     systems: bm25_bigram (always), neural and fusion (when a dense index + encoder exist)
POST /statutes              body {"text": str, "method": str = "nb", "top": int = 5}   (methods: nb, rocchio, knn)
  → {"method", "statutes": [{"id", "title", "score", "threshold", "probability" (NB only, else null), "above_threshold": bool}]}
     sorted by score − threshold (how far above its own dev-tuned threshold each statute is)
GET  /examples              → real test-set queries {id, title, text, num_relevant} for the demo buttons
GET  /results               → contents of results/*.csv|json (for the /evidence page)
```
Errors: `{"error": str, "detail": str}` with 400 (bad input), 503 (artifact missing → name the script to run).

**Frontend types** (`frontend/lib/types.ts`) mirror these JSON shapes exactly; `frontend/lib/api.ts` is the only file that calls the API.

## 10. Results file formats (`results/`)
Long-format CSV, columns `task, system, split, metric, value, n`, e.g. `pcr,bm25_qr,test,micro_f1@k,0.xx,237`.
JSON for structured stats (`data_stats.json`, `graph_stats.json`, `tuned_params.json`). PNG figures from `scripts/make_plots.py`.

## 11. Testing strategy (`tests/`)
- Text: protected legal references survive tokenisation (`"Section 498-A IPC"` → `sec_498a`).
- Index: postings sorted, df = postings length, save/load round-trip.
- BM25: matches a hand-computed toy example; **reference check** against `rank_bm25` on a toy corpus (tests only).
- PageRank/HITS: 3–4 node graphs with known answers; scores sum to 1 (PageRank); dangling-node case.
- NB/Rocchio/kNN: toy datasets with known predictions; reference check vs sklearn in tests only.
- Metrics: hand-computed P/R/F1, micro-F1@K, macro-F1.
- No-leakage test on split ids. API: `/health` and one `/search` call with a tiny fixture index.

## 12. Performance budget (laptop CPU, warm)
| Stage | Budget |
|---|---|
| tokenise + query reduction | < 50 ms |
| BM25 over one candidate pool | < 150 ms |
| encode query chunks (INT8) | < 300 ms |
| dense scoring (numpy over pool chunks) | < 100 ms |
| fusion + rerank top-50 | < 300 ms |
| statute prediction | < 100 ms |
| **total** | **< 1 s median** |
Measured and reported in `results/efficiency.csv`.
