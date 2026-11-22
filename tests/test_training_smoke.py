"""CPU smoke tests of the training code with a tiny random BERT (no download): shapes, masking, losses, one step."""
import random

import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

from casetrail.neural.distill import score_pairs, teacher_examples, train_teacher_epoch
from casetrail.neural.train_retriever import (batch_scores, contrastive_loss, dense_dev_run, make_examples,
                                              margin_mse_loss, maxsim_matrix, train_epoch)

WORDS = ["dowry", "death", "bride", "murder", "knife", "bail", "theft", "land", "tax", "appeal"]


def tiny_tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path))


def tiny_config():
    return BertConfig(vocab_size=15, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=600, num_labels=1)


def toy_data():
    chunks = {name: [" ".join(random.Random(i * 7 + j).choices(WORDS, k=8)) for j in range(3)]
              for i, name in enumerate(["q1", "q2", "p1", "p2", "n1", "n2", "n3", "n4"])}
    items = [{"query_id": "q1", "positives": ["p1"], "hard_negatives": ["n1", "n2"]},
             {"query_id": "q2", "positives": ["p2"], "hard_negatives": ["n3", "n4"]}]
    relevant = {"q1": {"p1"}, "q2": {"p2", "n1"}}  # n1 is (unlabelled-for-q1) relevant to q2
    return chunks, items, relevant


def test_maxsim_matrix_shape():
    scores = maxsim_matrix(torch.randn(2, 4, 8), torch.randn(6, 4, 8))
    assert scores.shape == (2, 6)


def test_contrastive_masks_other_relevant_candidates(tmp_path):
    torch.manual_seed(13)
    chunks, items, relevant = toy_data()
    model, tokenizer = BertModel(tiny_config()), tiny_tokenizer(tmp_path)
    batch = make_examples(items, hard_negatives=2, rng=random.Random(13))
    scores, doc_ids = batch_scores(model, tokenizer, batch, chunks, torch.device("cpu"), random.Random(13))
    assert scores.shape == (2, 6)
    loss = contrastive_loss(scores, batch, doc_ids, relevant)
    assert torch.isfinite(loss)
    teacher = {"q1": {"p1": 2.0, "n1": 0.0, "n2": -1.0}, "q2": {"p2": 1.0, "n3": 0.0, "n4": 0.5}}
    assert torch.isfinite(margin_mse_loss(scores, batch, teacher))


def test_one_student_epoch_and_dev_run(tmp_path):
    torch.manual_seed(13)
    chunks, items, relevant = toy_data()
    model, tokenizer = BertModel(tiny_config()), tiny_tokenizer(tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    examples = make_examples(items, 2, random.Random(13))
    settings = {"seed": 13, "batch_size": 2, "distill_weight": 1.0}
    loss = train_epoch(model, tokenizer, optimizer, examples, {"chunks": chunks, "relevant": relevant, "teacher": None},
                       settings, torch.device("cpu"))
    assert loss > 0
    run = dense_dev_run(model, tokenizer, chunks, {"q1": None}, torch.device("cpu"), depth=3)
    assert len(run["q1"]) == 3 and "q1" not in run["q1"]


def test_one_teacher_epoch_and_scoring(tmp_path):
    torch.manual_seed(13)
    chunks, items, relevant = toy_data()
    model, tokenizer = BertForSequenceClassification(tiny_config()), tiny_tokenizer(tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    examples = teacher_examples(items, random.Random(13))
    assert sorted(label for q, d, label in examples) == [0.0, 0.0, 1.0, 1.0]
    loss = train_teacher_epoch(model, tokenizer, optimizer, examples, chunks, {"seed": 13, "batch_size": 2},
                               torch.device("cpu"))
    assert loss > 0
    scores = score_pairs(model, tokenizer, [("q1", "p1"), ("q1", "n1")], chunks, torch.device("cpu"))
    assert len(scores) == 2
    assert scores == score_pairs(model, tokenizer, [("q1", "p1"), ("q1", "n1")], chunks, torch.device("cpu"))
