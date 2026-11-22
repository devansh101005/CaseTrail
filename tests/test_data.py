"""Loader format and no-leakage checks on the real IL-TUR files (skipped if data/raw/ is not downloaded)."""
import pytest

from casetrail import config
from casetrail.data import lsi_label_names, load_lsi, load_pcr, load_statutes


def needs_data():
    if not (config.RAW_DIR / "pcr").exists() or not (config.RAW_DIR / "lsi").exists():
        pytest.skip("IL-TUR not downloaded (run scripts/download_data.py)")


def relevant_pairs(pcr_split: dict) -> set:
    return {(q_id, c_id) for q_id, q in pcr_split["queries"].items() for c_id in q["relevant"]}


def test_pcr_counts_match_card():
    needs_data()
    sizes = {split: len(load_pcr(split)["queries"]) for split in config.PCR_SPLITS}
    assert sizes == {"train": 827, "dev": 118, "test": 237}


def test_pcr_format():
    needs_data()
    dev = load_pcr("dev")
    query = next(iter(dev["queries"].values()))
    assert isinstance(query["sentences"], list) and isinstance(query["sentences"][0], str)
    assert len(query["relevant"]) >= 1
    assert query["id"] not in query["relevant"]


def test_pcr_no_leakage_between_splits():
    # Candidate pools share old judgments by design (identical text), so the rule we enforce is:
    # query sets are disjoint and no labelled (query, relevant) pair of test/dev appears in train.
    needs_data()
    pcr = {split: load_pcr(split) for split in config.PCR_SPLITS}
    test_queries = set(pcr["test"]["queries"])
    assert not test_queries & set(pcr["train"]["queries"])
    assert not test_queries & set(pcr["dev"]["queries"])
    assert not set(pcr["dev"]["queries"]) & set(pcr["train"]["queries"])
    assert not relevant_pairs(pcr["test"]) & (relevant_pairs(pcr["train"]) | relevant_pairs(pcr["dev"]))


def test_lsi_labels_and_statutes_line_up():
    needs_data()
    names = lsi_label_names()
    statutes = load_statutes()
    assert len(names) == 100
    assert names == list(statutes)
    assert statutes["Section 302"]["title"]


def test_lsi_no_leakage_between_splits():
    needs_data()
    ids = {split: {doc["id"] for doc in load_lsi(split)} for split in config.LSI_SPLITS}
    assert not ids["test"] & (ids["train"] | ids["dev"])
    assert not ids["dev"] & ids["train"]
