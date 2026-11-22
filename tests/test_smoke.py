"""Smoke test: the package imports and the config is consistent."""
from casetrail import config


def test_seed_is_fixed():
    assert config.SEED == 13


def test_paths_are_inside_project():
    for folder in [config.DATA_DIR, config.RAW_DIR, config.INDEX_DIR, config.MODEL_DIR, config.RESULTS_DIR]:
        assert config.PROJECT_ROOT in folder.parents


def test_defaults_are_sane():
    assert 0 < config.PAGERANK_DAMPING < 1
    assert 0 <= config.BM25_B <= 1
    assert config.CHUNK_STRIDE <= config.CHUNK_TOKENS
    assert config.FINAL_K <= config.RERANK_TOP <= config.CANDIDATES_PER_SIGNAL
