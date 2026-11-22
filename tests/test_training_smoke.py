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
    loss = train_teacher_epoch(model, tokenizer, optimizer, examples, chunks, {"seed": 13, "batch_size": 2, "lr": 1e-3},
                               torch.device("cpu"))
    assert loss > 0
    scores = score_pairs(model, tokenizer, [("q1", "p1"), ("q1", "n1")], chunks, torch.device("cpu"))
    assert len(scores) == 2
    assert scores == score_pairs(model, tokenizer, [("q1", "p1"), ("q1", "n1")], chunks, torch.device("cpu"))


def test_selected_pairs_take_the_most_similar_chunks():
    import numpy as np
    from casetrail.neural.distill import SELECTED_PAIRS, selected_pairs
    query_chunks, doc_chunks = ["q0", "q1"], ["d0", "d1", "d2"]
    query_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc_vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    pairs = selected_pairs(query_chunks, doc_chunks, query_vectors, doc_vectors)
    assert len(pairs) == SELECTED_PAIRS
    assert set(pairs[:2]) == {("q0", "d1"), ("q1", "d0")}   # the two cosine-1.0 pairs come first
    one = selected_pairs(["q0"], ["d0"], np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert one == [("q0", "d0")] * SELECTED_PAIRS            # tiny judgments: the pair is repeated


def test_warmup_raises_the_learning_rate_linearly():
    from casetrail.neural.distill import set_learning_rate
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    settings = {"lr": 2e-5, "warmup_steps": 10}
    set_learning_rate(optimizer, settings, 5)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12
    set_learning_rate(optimizer, settings, 50)
    assert optimizer.param_groups[0]["lr"] == 2e-5


def test_teacher_with_selected_pairs_and_dev_check(tmp_path):
    import numpy as np
    from casetrail.neural.distill import dev_pair_accuracy
    torch.manual_seed(13)
    chunks, items, relevant = toy_data()
    vectors = {name: np.random.default_rng(i).normal(size=(3, 4)) for i, name in enumerate(sorted(chunks))}
    model, tokenizer = BertForSequenceClassification(tiny_config()), tiny_tokenizer(tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    examples = teacher_examples(items, random.Random(13))
    settings = {"seed": 13, "batch_size": 2, "lr": 1e-3, "warmup_steps": 1}
    loss = train_teacher_epoch(model, tokenizer, optimizer, examples, chunks, settings, torch.device("cpu"), vectors)
    assert loss > 0
    accuracy = dev_pair_accuracy(model, tokenizer, {"q1": {"p1"}}, {"q1": ["p1", "n1", "n2"]}, chunks,
                                 torch.device("cpu"), vectors)
    assert 0.0 <= accuracy <= 1.0


def test_matching_chunks_pick_the_candidate_parts_that_match_the_query():
    import numpy as np
    from casetrail.neural.train_retriever import matching_chunks
    query_vectors = np.array([[1.0, 0.0]])
    doc_vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]])
    assert matching_chunks(query_vectors, doc_vectors, 2) == [1, 2]
    assert matching_chunks(query_vectors, doc_vectors[:1], 3) == [0, 0, 0]


def test_student_epoch_with_selection(tmp_path):
    import numpy as np
    torch.manual_seed(13)
    chunks, items, relevant = toy_data()
    selection = {name: np.random.default_rng(i).normal(size=(3, 4)) for i, name in enumerate(sorted(chunks))}
    model, tokenizer = BertModel(tiny_config()), tiny_tokenizer(tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    examples = make_examples(items, 2, random.Random(13))
    data = {"chunks": chunks, "relevant": relevant, "teacher": None, "selection": selection}
    loss = train_epoch(model, tokenizer, optimizer, examples, data, {"seed": 13, "batch_size": 2, "distill_weight": 1.0},
                       torch.device("cpu"))
    assert loss > 0
