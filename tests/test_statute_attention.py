"""Statute cross-attention on random toy embeddings: shapes, masking, and that training reduces the loss."""
import numpy as np
import torch

from casetrail.neural.statute_attention import (StatuteAttention, pad_batch, positive_weights, predict_scores,
                                                train_epoch)


def test_pad_batch_and_mask():
    chunks, mask = pad_batch([np.ones((2, 4)), np.ones((3, 4))])
    assert chunks.shape == (2, 3, 4)
    assert mask.tolist() == [[False, False, True], [False, False, False]]


def test_padding_does_not_change_the_output():
    torch.manual_seed(13)
    model = StatuteAttention(np.random.default_rng(13).normal(size=(5, 8)).astype(np.float32), heads=2).eval()
    doc = np.random.default_rng(1).normal(size=(2, 8)).astype(np.float32)
    alone, alone_mask = pad_batch([doc])
    padded, padded_mask = pad_batch([doc, np.zeros((4, 8), dtype=np.float32)])
    with torch.no_grad():
        assert torch.allclose(model(alone, alone_mask)[0], model(padded, padded_mask)[0], atol=1e-5)


def test_positive_weights():
    labels = np.array([[1, 0], [0, 0], [0, 0], [1, 1]])
    assert positive_weights(labels).tolist() == [1.0, 3.0]


def test_training_learns_a_separable_toy_task():
    rng = np.random.default_rng(13)
    torch.manual_seed(13)
    statutes = rng.normal(size=(2, 8)).astype(np.float32)
    docs, labels = [], []
    for i in range(64):
        label = i % 2
        docs.append((rng.normal(size=(3, 8)) * 0.1 + statutes[label]).astype(np.float32))
        labels.append([1 - label, label])
    labels = np.array(labels)
    model = StatuteAttention(statutes, heads=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = train_epoch(model, optimizer, docs, labels, positive_weights(labels), 16, torch.device("cpu"), rng)
    for epoch in range(10):
        last = train_epoch(model, optimizer, docs, labels, positive_weights(labels), 16, torch.device("cpu"), rng)
    assert last < first
    scores = predict_scores(model, docs, 16, torch.device("cpu"))
    accuracy = ((scores[:, 1] > scores[:, 0]) == labels[:, 1].astype(bool)).mean()
    assert accuracy > 0.9
