"""Project-wide constants: paths, random seed and default parameters (LLD section 1).

Values marked "tune on dev" are starting points; tuned values are written to results/tuned_params.json.
"""
from pathlib import Path

# Paths (everything except results/ is gitignored and rebuilt by scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = PROJECT_ROOT / "indexes"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset
HF_DATASET = "Exploration-Lab/IL-TUR"
TASKS = ["pcr", "lsi"]
PCR_SPLITS = ["train", "dev", "test"]
LSI_SPLITS = ["train", "dev", "test"]

# Reproducibility
SEED = 13

# BM25 (tune on dev)
BM25_K1 = 1.2
BM25_B = 0.75

# Query reduction: keep the top-N query terms by tf-idf (tune on dev; 0 = full text)
QUERY_TOP_TERMS = 50

# Candidate generation and final list sizes
CANDIDATES_PER_SIGNAL = 200
RERANK_TOP = 50
FINAL_K = 10

# Chunking long judgments for the neural encoder
CHUNK_TOKENS = 256
CHUNK_STRIDE = 192
MAX_CHUNKS_PER_DOC = 32

# PageRank (d = damping factor)
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-8
PAGERANK_MAX_ITER = 100

# HITS
HITS_MAX_ITER = 100
HITS_TOL = 1e-8

# Classifiers (tune on dev)
NB_ALPHA = 1.0
KNN_K = 25

# Reciprocal rank fusion constant
RRF_K = 60

# Demo API
API_HOST = "127.0.0.1"
API_PORT = 8000
