"""API checks: input validation always; /health and one /search + /statutes when the built artifacts exist."""
import pytest
from fastapi.testclient import TestClient

from casetrail import config

ARTIFACTS = [config.INDEX_DIR / "pcr_test_bi_stem" / "meta.json", config.RESULTS_DIR / "tuned_params.json",
             config.MODEL_DIR / "lsi_classical.npz", config.INDEX_DIR / "lsi_train" / "meta.json"]


def client():
    from api.main import app
    return TestClient(app)


def test_empty_text_is_rejected_with_400():
    if not all(path.exists() for path in ARTIFACTS):
        pytest.skip("artifacts not built (run the scripts in README)")
    with client() as api:
        response = api.post("/search", json={"text": "   "})
        assert response.status_code == 400
        assert response.json()["error"] == "bad input"


def test_search_and_statutes_end_to_end():
    if not all(path.exists() for path in ARTIFACTS):
        pytest.skip("artifacts not built (run the scripts in README)")
    with client() as api:
        assert api.get("/health").json()["status"] == "ok"
        text = "The husband and his mother harassed the bride for dowry and she died of burns within seven years."
        found = api.post("/search", json={"text": text, "system": "bm25_bigram", "k": 5}).json()
        assert len(found["results"]) == 5
        assert found["results"][0]["breakdown"]["bm25"] >= found["results"][-1]["breakdown"]["bm25"]
        statutes = api.post("/statutes", json={"text": text, "method": "nb", "top": 3}).json()["statutes"]
        assert len(statutes) == 3 and all(s["id"].startswith("Section") for s in statutes)
        bad = api.post("/statutes", json={"text": text, "method": "unknown"})
        assert bad.status_code == 400
