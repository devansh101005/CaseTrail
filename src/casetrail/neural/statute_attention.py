"""Statute-description cross-attention classifier for LSI (beyond syllabus; LLD section 8).

Each of the 100 IPC statutes is a query vector (its description embedded by the encoder, then a learned
projection). It attends over the chunk vectors of a case (multi-head attention, padding masked), so every
statute gathers the parts of the judgment that talk about it. A shared linear layer turns
LayerNorm(attended + statute) into one logit per statute; the attention weights explain the decision.
"""
import numpy as np
import torch


class StatuteAttention(torch.nn.Module):
    def __init__(self, statute_vectors: np.ndarray, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        num_statutes, dim = statute_vectors.shape
        self.statutes = torch.nn.Parameter(torch.tensor(statute_vectors, dtype=torch.float32))
        self.statute_proj = torch.nn.Linear(dim, dim)
        self.chunk_proj = torch.nn.Linear(dim, dim)
        self.attention = torch.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.score = torch.nn.Linear(dim, 1)
        self.statute_bias = torch.nn.Parameter(torch.zeros(num_statutes))

    def forward(self, chunks: torch.Tensor, padding_mask: torch.Tensor, return_attention: bool = False):
        """chunks [B, n, dim]; padding_mask [B, n] True where padded -> logits [B, num_statutes]."""
        batch = chunks.shape[0]
        queries = self.statute_proj(self.statutes).unsqueeze(0).expand(batch, -1, -1)
        keys = self.chunk_proj(chunks)
        attended, weights = self.attention(queries, keys, keys, key_padding_mask=padding_mask)
        hidden = self.norm(self.dropout(attended) + queries)
        logits = self.score(hidden).squeeze(-1) + self.statute_bias
        if return_attention:
            return logits, weights
        return logits


def pad_batch(doc_chunks: list) -> tuple:
    """List of [n_i, dim] arrays -> (tensor [B, max_n, dim], padding mask [B, max_n])."""
    longest = max(len(chunks) for chunks in doc_chunks)
    dim = doc_chunks[0].shape[1]
    padded = np.zeros((len(doc_chunks), longest, dim), dtype=np.float32)
    mask = np.ones((len(doc_chunks), longest), dtype=bool)
    for i, chunks in enumerate(doc_chunks):
        padded[i, :len(chunks)] = chunks
        mask[i, :len(chunks)] = False
    return torch.tensor(padded), torch.tensor(mask)


def positive_weights(labels: np.ndarray, clip: float = 50.0) -> torch.Tensor:
    """pos_weight = (#negatives / #positives) per statute, clipped, so rare statutes are not ignored."""
    positives = labels.sum(axis=0)
    weights = (len(labels) - positives) / np.maximum(positives, 1)
    return torch.tensor(np.minimum(weights, clip), dtype=torch.float32)


def train_epoch(model, optimizer, doc_chunks: list, labels: np.ndarray, pos_weight: torch.Tensor,
                batch_size: int, device, rng: np.random.Generator) -> float:
    model.train()
    order = rng.permutation(len(doc_chunks))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    losses = []
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        chunks, mask = pad_batch([doc_chunks[i] for i in ids])
        target = torch.tensor(labels[ids], dtype=torch.float32, device=device)
        loss = loss_fn(model(chunks.to(device), mask.to(device)), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict_scores(model, doc_chunks: list, batch_size: int, device) -> np.ndarray:
    """Sigmoid probabilities [num_docs x num_statutes]."""
    model.eval()
    scores = []
    for start in range(0, len(doc_chunks), batch_size):
        chunks, mask = pad_batch(doc_chunks[start:start + batch_size])
        scores.append(torch.sigmoid(model(chunks.to(device), mask.to(device))).cpu().numpy())
    return np.concatenate(scores)
