# CLAUDE.md — CaseTrail (CSD358 IR course project)

We are Devansh Pandey (2310110461), Anamika Pal (2310110037) and Abhinav Bachchas (2310110383).
This file tells Claude Code how to work in this repo. **Read it, then `PLAN.md` and `ROADMAP.md`, before doing anything.**

## What we are building
**CaseTrail: Precedent and Statute Retrieval for Indian Court Judgments using Link Analysis and Distilled Neural Models.**

Given an Indian Supreme Court judgment (or a description of a case), CaseTrail:
1. **Prior case retrieval (PCR):** returns the most similar earlier judgments (precedents).
2. **Legal statute identification (LSI):** predicts which of 100 IPC sections apply, with confidence scores.
3. Explains every result: keyword score (BM25), neural score, graph authority (PageRank / HITS).

Data: the **IL-TUR** benchmark (Joshi et al., ACL 2024), tasks `pcr` and `lsi`. Full details in `docs/DATA.md`.

- Master plan: `PLAN.md`
- Phases and done-criteria: `ROADMAP.md`
- System design: `docs/HLD.md` (components, data flow, deployment) and `docs/LLD.md` (data formats, algorithms, defaults, API contract, tests). **Build to these; if you need to deviate, say why and update the doc.**
- Stack choices and trade-offs: `docs/DECISIONS.md` (add a dated entry whenever a decision changes)
- Report format required by the professor: `docs/REPORT_FORMAT.md`
- Frontend design system (based on our Waypoint project): `docs/FRONTEND_GUIDE.md`
- Lecture concept → code map (keep it updated): `docs/IR_CONCEPTS.md`
- Metrics and protocol: `docs/EVALUATION.md`
- Data sources and credits: `docs/DATA.md`
- Submission checklist: `docs/SUBMISSION.md`

## Grading (25 points) — optimise for this
Demo 10 · Viva 5 · Report 6 · Novelty 4. So: a **working live demo** that we can explain well, a research-paper report with
a comparison against a published paper, and clear novelty over the obvious baseline.

## Hard rules
1. **Never commit, push, tag or create branches.** Devansh commits himself. You may run read-only git commands (`git status`, `git diff`, `git log`). At the end of each phase, list what changed and *suggest* a commit message.
2. **Never fabricate or hard-code results.** Every number in the README, report, frontend or demo must come from a script in `scripts/` that writes to `results/`. If a number is not produced yet, show "not run yet" — never a placeholder value.
3. **No test leakage.** The IL-TUR **test** splits are used only for the final evaluation. All tuning, training, model selection, thresholds and early stopping use **train/dev** only.
4. **We write the core IR ourselves:** inverted index, BM25, tf-idf, PageRank, HITS, Naive Bayes, Rocchio, kNN, metrics. Do not use `rank_bm25`, `pyserini`, `networkx.pagerank`, sklearn's `MultinomialNB`/`NearestCentroid`/`TfidfVectorizer` etc. in the main pipeline. They are allowed **only in `tests/`** as reference checks. Neural parts may use PyTorch / Hugging Face / ONNX Runtime.
5. **Don't copy code** from Devansh's VidhiVault / AdhivaktaAI project, LipiSetu, or any online repo or tutorial. Write it fresh; cite ideas in the report.
6. **Never commit or create large files in git:** `data/`, `models/`, `indexes/`, `.venv/`, `frontend/node_modules/` and any file over ~5 MB stay gitignored. Small CSV/JSON/PNG in `results/` are fine.
7. **Ask before** any download over ~1 GB, any training run expected to take over ~30 minutes, installing system-level software, or changing the plan/architecture.
8. **Follow the phase order** in `ROADMAP.md`. If a later phase is requested while an earlier one is broken, say so first.
9. **Credit everything.** Every dataset, model and library goes into `docs/DATA.md` and the dependency files.
10. **Respect licences.** IL-TUR is CC BY-NC-SA 4.0 and gated on Hugging Face. Never put a Hugging Face token in code or git (use the `HF_TOKEN` env var or `huggingface-cli login`).
11. Don't create extra log/notes files about our conversations. Keep the repo to code and the docs listed above. `docs/AI_USE.md` stays short and truthful.
12. **Explain as you go.** After each phase, give a short plain-English summary of what was built and how it works, so each of us can explain it in the viva.

## How the code should be written
Write the code the way a good student writes it: **simple, direct and readable**, so we can explain every line in the viva. If there is a plain way and a clever way, use the plain way. Prefer a little repetition over an abstraction that is hard to explain.

### General rules (all languages)
- **Plain functions first.** Use a class only when it really holds data plus the methods that work on it (`InvertedIndex`, `CaseTrailEngine`, PyTorch `nn.Module`s). Methods are ordinary methods: no class hierarchies, no abstract base classes, no mixins, no factories, no registries, no dependency injection, no plugin systems.
- **One obvious way to run things.** Scripts and functions call each other directly. No event buses, task queues, background workers, caching layers or config frameworks.
- **Communication between frontend and backend is plain REST over HTTP with JSON** (FastAPI routes, `fetch` on the frontend). No RPC frameworks (gRPC, tRPC, JSON-RPC), no GraphQL, no WebSockets/streaming unless we explicitly ask for it.
- **Few dependencies.** Don't add a library for something a short function can do. Ask before adding any new dependency, and say why it is needed.
- Short functions (roughly under 40 lines), clear names, early returns instead of deep nesting.
- Comments are short and in plain English, explaining the **idea**, not repeating the code. No decorative banners, no emojis, no "Step 1/Step 2" essays, no long docstrings on every tiny helper.
- Errors: check inputs where they enter the system (API request, file loading) and raise a clear message. Don't wrap everything in `try/except`, and never silently swallow errors.

### Python (backend, IR, ML)
**Use:** `def` functions, `for`/`while` loops, `if/elif/else`, lists, dicts, sets, tuples, simple one-level list comprehensions (`[t for t in tokens if t not in stop_words]`), f-strings, `with open(...)`, `argparse`, `print()` for progress, `numpy`, `torch`.
**Avoid:**
- `dataclasses`, `NamedTuple`, `Enum`, `typing.Protocol`, `TypeVar`/generics, `abc`, `__slots__`, metaclasses, descriptors, `@property` chains, custom context-manager classes.
- Decorators other than the ones a library requires (`@app.get`/`@app.post` in FastAPI, `@torch.no_grad()` is fine). No `functools.lru_cache`/`partial`/`reduce`, no `itertools` tricks.
- Walrus `:=`, `match` statements, nested or multi-line comprehensions, `lambda` (except a simple sort key), generator pipelines (`yield` only when reading a huge file), `map`/`filter` chains, chained ternaries, star-unpacking tricks.
- `async`/`await` (FastAPI routes are plain `def`), threading/multiprocessing unless we ask.
- Heavy type hints. Simple hints on function signatures are fine (`def bm25_score(query_terms: list, doc_id: int) -> float:`); no complex nested types.
- Logging frameworks, custom exception hierarchies, environment/config libraries. A plain `config.py` with constants is enough.
- For FastAPI request bodies, a small `pydantic.BaseModel` per endpoint is OK (FastAPI needs it); keep it to plain fields.

**Conventions:**
- Every file starts with a one- or two-line docstring naming the concept and the lecture, e.g. `"""HITS hubs and authorities (Lecture: Link analysis - HITS)."""`
- A one-line comment before each important block explains the IR idea, e.g. `# PageRank: a judgment is important if important judgments point to it`.
- Names: `doc_id`, `term_freq`, `postings`, `query_terms`, `statute_id`, `scores`. No single letters except `i`/`j` in short loops and standard maths (`k1`, `b`, `d` for damping) with a comment.
- Scripts: `argparse` + `def main():` + `if __name__ == "__main__": main()`.
- Save/load with `json`, `pickle`, `numpy.save`, `torch.save`; text I/O always `encoding="utf-8"`.
- Fix the random seed everywhere (`SEED = 13` in `config.py`). Neural results are reported as mean ± std over 3 seeds where time allows.
- Tests: plain `pytest` functions with simple `assert`s and tiny hand-made examples; no fixtures framework magic, no mocking libraries unless unavoidable.

### TypeScript / Next.js (frontend)
**Use:** function components, `useState`, `useEffect`, props with a simple `type` or `interface`, `fetch` inside `frontend/lib/api.ts`, Tailwind classes, `next/font` for fonts. Follow `docs/FRONTEND_GUIDE.md` for the look.
**Avoid:** Redux/Zustand/Jotai or any global state library, React Query/SWR, server actions, tRPC, complex generics, higher-order components, render props, custom hook factories (a small `useSomething` hook is fine), CSS-in-JS libraries, animation libraries, large UI kits.
**Conventions:** one component per file, under ~150 lines; the frontend only displays what the API returns (no ranking or scoring logic in TypeScript); show loading and error states plainly.

## Repository layout (target)
```
CLAUDE.md  PLAN.md  ROADMAP.md  README.md  .gitignore  requirements.txt
src/casetrail/
  config.py                 # paths, seed, default parameters
  data.py                   # load IL-TUR pcr + lsi splits, statute descriptions
  text/                     # normalisation, tokenizer, stop words, legal-section patterns
  index/inverted_index.py   # dictionary + postings, doc lengths
  retrieval/bm25.py         # BM25 (+ tf-idf for comparison)
  retrieval/chunking.py     # long-judgment handling (passages, query reduction)
  graph/build_graph.py      # judgment–statute bipartite graph (+ citation graph if available)
  graph/pagerank.py         # PageRank (power iteration, damping, dangling nodes)
  graph/hits.py             # HITS hubs/authorities
  classify/naive_bayes.py   # multinomial NB, one-vs-rest for multi-label
  classify/rocchio.py       # Rocchio / nearest centroid
  classify/knn.py           # kNN on tf-idf vectors
  neural/encoder.py         # bi-encoder (student)
  neural/train_retriever.py # contrastive training with BM25 hard negatives
  neural/distill.py         # cross-encoder teacher -> student distillation
  neural/statute_attention.py # statute-description cross-attention classifier
  neural/export_onnx.py     # ONNX export + INT8 quantisation
  fusion/combine.py         # score fusion (RRF / learned weights) of BM25 + neural + graph
  eval/metrics.py           # P, R, F1, micro-F1@K, macro-F1, MAP, nDCG
  eval/significance.py      # paired randomisation / bootstrap tests
  search.py                 # CaseTrailEngine: one entry point for every system
api/main.py                 # FastAPI backend for the demo
frontend/                   # Next.js + TypeScript + Tailwind (see docs/FRONTEND_GUIDE.md)
scripts/                    # one script per step / experiment
results/                    # committed CSV/JSON/PNG (small)
notebooks/                  # only Kaggle training notebooks, kept minimal
tests/                      # pytest; reference checks live here
docs/                       # plan docs and the report
data/ models/ indexes/      # gitignored, rebuilt by scripts
```

## Environment notes
- Windows 11 laptop: Intel i5-12450H, 16 GB RAM, **NVIDIA GTX 1650 (4 GB)**. Python 3.13 (3.10–3.13 must work).
- Heavy training (teacher cross-encoder, contrastive training) runs on **Kaggle free GPUs (T4)** via a notebook in `notebooks/`; the laptop runs indexing, classical IR, inference and the demo.
- Windows: set `PYTHONUTF8=1`. If Windows Smart App Control blocks a scikit-learn DLL, use `scikit-learn==1.7.2`.
- Keep inference CPU-friendly (INT8 ONNX). The demo must run on the laptop **without internet** after setup.

## Gotchas
- IL-TUR is **gated**: accept the conditions at https://huggingface.co/datasets/Exploration-Lab/IL-TUR and log in before downloading. The card shows `load_dataset("Exploration-Lab/IL-TUR", "<task>", revision="script")` — verify this works before relying on it.
- In PCR, the `text` field is a **list of sentences**; queries and candidates are full judgments (very long). Check how citations to other cases appear in the text (they may be masked) before building a citation graph.
- PCR has separate train / dev / test **query and candidate pools** (test: 237 queries, ~1.7k candidates). Retrieve only from the candidate pool of the same split.
- LSI is **multi-label** over 100 IPC statutes, with a `statutes` split of statute descriptions. Report macro-F1 (the published baseline metric) plus micro-F1.
- Use the metric definitions from the IL-TUR paper (e.g. micro-F1@K for PCR) so our numbers are comparable with the published baselines.
- IPC was replaced by the Bharatiya Nyaya Sanhita (BNS) from 1 July 2024. The dataset uses IPC; mention this as a limitation / future work.

## Definition of done
The checklists in `ROADMAP.md`. The README must rebuild everything from a fresh clone. The final step before submission is a clean-folder reproduction test and the ZIP described in `docs/SUBMISSION.md`.
