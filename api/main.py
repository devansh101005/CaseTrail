"""FastAPI backend for the CaseTrail demo (HLD C11; API contract in LLD section 9).

Run:  uvicorn api.main:app --host 127.0.0.1 --port 8000
Plain JSON over HTTP; errors are {"error", "detail"} with 400 (bad input) or 503 (artifact missing).
"""
import csv
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casetrail import config
from casetrail.engine_parts import title_of
from casetrail.search import CaseTrailEngine

app = FastAPI(title="CaseTrail API")
# The Next.js dev server runs on another port of the same machine
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

ENGINE = {"engine": None, "error": None}
MAX_TEXT_CHARS = 2_000_000


class SearchBody(BaseModel):
    text: str
    system: str = "bm25_bigram"
    k: int = 10


class StatuteBody(BaseModel):
    text: str
    method: str = "nb"
    top: int = 5


def error(status: int, message: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message, "detail": detail})


@app.exception_handler(ValueError)
def bad_input(request: Request, exc: ValueError):
    return error(400, "bad input", str(exc))


@app.exception_handler(FileNotFoundError)
def missing_artifact(request: Request, exc: FileNotFoundError):
    return error(503, "artifact missing", str(exc))


def engine() -> CaseTrailEngine:
    """Load everything once (first request or start-up); a missing artifact names the script to run."""
    if ENGINE["engine"] is None:
        ENGINE["engine"] = CaseTrailEngine(split="test")
    return ENGINE["engine"]


def warm_up():
    """Load the engine and run one query of each kind when the server starts, so the first demo query is fast."""
    try:
        text = "dowry death of the bride within seven years of marriage"
        engine().search(text, k=1)
        for method in engine().available()["methods"]:
            engine().predict_statutes(text, method=method, top=1)
    except (FileNotFoundError, KeyError) as exc:
        ENGINE["error"] = str(exc)
        print(f"CaseTrail engine not ready: {exc}")


warm_up()


def check_text(text: str):
    if not text.strip():
        raise ValueError("text is empty: paste a judgment or describe the facts")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")


@app.get("/health")
def health():
    if ENGINE["engine"] is None:
        return {"status": "not ready", "detail": ENGINE["error"]}
    return dict(engine().available(), status="ok")


@app.post("/search")
def search(body: SearchBody):
    check_text(body.text)
    if not 1 <= body.k <= 50:
        raise ValueError("k must be between 1 and 50")
    return engine().search(body.text, system=body.system, k=body.k)


@app.post("/statutes")
def statutes(body: StatuteBody):
    check_text(body.text)
    if not 1 <= body.top <= 20:
        raise ValueError("top must be between 1 and 20")
    return engine().predict_statutes(body.text, method=body.method, top=body.top)


@app.get("/examples")
def examples(limit: int = 8):
    """Real test-set query judgments for the demo buttons (text included so the frontend can fill the box)."""
    queries = engine().pool["queries"]
    rows = []
    for query_id in sorted(queries)[:limit]:
        sentences = queries[query_id]["sentences"]
        rows.append({"id": query_id, "title": title_of(sentences), "text": "\n".join(sentences),
                     "num_relevant": len(queries[query_id]["relevant"])})
    return {"examples": rows}


@app.get("/results")
def results():
    """Everything in results/ that the evidence page shows (CSV as rows, JSON as is)."""
    output = {}
    for path in sorted(config.RESULTS_DIR.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            output[path.stem] = list(csv.DictReader(f))
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            output[path.stem] = json.load(f)
    output["figures"] = sorted(path.name for path in config.RESULTS_DIR.glob("fig_*.png"))
    return output
