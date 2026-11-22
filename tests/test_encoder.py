"""Chunking and MaxSim scoring with a fake whitespace tokenizer and hand-made vectors (no model download needed)."""
import numpy as np

from casetrail.neural.encoder import chunk_judgment, clean_sentence, maxsim_scores, mean_vector_scores, pick_evenly


def fake_tokenizer(texts, add_special_tokens=False):
    return {"input_ids": [text.split() for text in texts]}


def test_clean_sentence_removes_tags_and_citations():
    assert clean_sentence("<NAME> relied on <CITATION> AIR 1974 SC 753 here") == "relied on here"


def test_chunks_respect_the_token_budget():
    sentences = ["w " * 60, "x " * 60, "y " * 60, "z " * 10]
    chunks = chunk_judgment(sentences, fake_tokenizer, max_tokens=130, max_chunks=10)
    assert [len(chunk.split()) for chunk in chunks] == [120, 70]


def test_pick_evenly_keeps_first_and_last():
    assert pick_evenly(list(range(10)), 4) == [0, 3, 6, 9]
    assert pick_evenly([1, 2], 4) == [1, 2]


def test_maxsim_takes_best_chunk_per_query_chunk():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    chunks = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])   # doc 0 = chunks 0-1, doc 1 = chunk 2
    offsets = np.array([0, 2, 3])
    scores = maxsim_scores(query, chunks, offsets)
    assert np.allclose(scores, [(1.0 + 0.8) / 2, (0.0 + 1.0) / 2])
    mean_scores = mean_vector_scores(query, chunks, offsets)
    assert mean_scores.shape == (2,)
