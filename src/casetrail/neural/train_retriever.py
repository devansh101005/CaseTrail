"""Contrastive training of the bi-encoder student with BM25 hard negatives (+ optional Margin-MSE distillation).

One training example = (query judgment, one relevant candidate, H hard negatives mined by BM25).
Each judgment is represented by a few randomly sampled chunks; score(q, d) = MaxSim over those chunks.
InfoNCE loss (temperature tau): for every query, its positive must outscore every other candidate in the batch
(its own hard negatives and the other queries' candidates). Candidates that are relevant to the query but are
not its target are masked out, so a true precedent is never pushed away.
Distillation (Hofstaetter et al., 2020): Margin-MSE between the student's and the teacher's margins
s(q, pos) - s(q, neg) for the query's own hard negatives.
"""
import random

import numpy as np
import torch

from casetrail.neural.encoder import load_encoder

QUERY_CHUNKS = 4
DOC_CHUNKS = 4
TAU = 0.05


def sample_chunks(chunks: list, count: int, rng: random.Random) -> list:
    """`count` chunks of a judgment (with repetition if it has fewer)."""
    if len(chunks) >= count:
        return rng.sample(chunks, count)
    return [chunks[i % len(chunks)] for i in range(count)]


def encode_batch(model, tokenizer, texts: list, device, max_tokens: int = 256) -> torch.Tensor:
    """Unit-length mean-pooled vectors, with gradients (training)."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
    token_vectors = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1)


def maxsim_matrix(query_vectors: torch.Tensor, doc_vectors: torch.Tensor) -> torch.Tensor:
    """query_vectors [Q, qc, dim], doc_vectors [D, dc, dim] -> MaxSim scores [Q, D]."""
    similarity = torch.einsum("iqd,jcd->ijqc", query_vectors, doc_vectors)
    return similarity.max(dim=3).values.mean(dim=2)


def make_examples(train_items: list, hard_negatives: int, rng: random.Random) -> list:
    """train_items: [{"query_id", "positives", "hard_negatives"}] -> [(query, positive, [negatives])]."""
    examples = []
    for item in train_items:
        for positive in item["positives"]:
            negatives = item["hard_negatives"][:]
            rng.shuffle(negatives)
            if len(negatives) >= hard_negatives:
                examples.append((item["query_id"], positive, negatives[:hard_negatives]))
    rng.shuffle(examples)
    return examples


def sample_indices(count_available: int, count: int, rng: random.Random) -> list:
    """`count` chunk positions of a judgment (with repetition if it has fewer)."""
    if count_available >= count:
        return rng.sample(range(count_available), count)
    return [i % count_available for i in range(count)]


def matching_chunks(query_vectors: np.ndarray, doc_vectors: np.ndarray, count: int) -> list:
    """Positions of the `count` candidate chunks that best match the chosen query chunks (selection vectors).
    Full MaxSim at evaluation lets every query chunk find its best candidate chunk; training on these chunks
    copies that, where random candidate chunks would mostly miss the matching part."""
    best_match = (query_vectors @ doc_vectors.T).max(axis=0)
    order = np.argsort(-best_match).tolist()
    return [order[i % len(order)] for i in range(count)]


def batch_scores(model, tokenizer, batch: list, chunks: dict, device, rng: random.Random, selection: dict = None) -> tuple:
    """MaxSim scores [B, B * (1 + H)] between the batch's queries and all candidates in the batch.
    selection = {doc_id: chunk vectors from a fixed encoder}: candidate chunks are picked to match the query's
    chunks instead of at random (query chunks stay random, so every part of the query gets trained)."""
    query_texts, doc_texts, doc_ids = [], [], []
    for query_id, positive, negatives in batch:
        picked = sample_indices(len(chunks[query_id]), QUERY_CHUNKS, rng)
        query_texts.extend([chunks[query_id][i] for i in picked])
        for doc_id in [positive] + negatives:
            if selection is None:
                doc_texts.extend(sample_chunks(chunks[doc_id], DOC_CHUNKS, rng))
            else:
                positions = matching_chunks(selection[query_id][picked], selection[doc_id], DOC_CHUNKS)
                doc_texts.extend([chunks[doc_id][i] for i in positions])
            doc_ids.append(doc_id)
    vectors = encode_batch(model, tokenizer, query_texts + doc_texts, device)
    dim = vectors.shape[-1]
    query_vectors = vectors[:len(query_texts)].reshape(len(batch), QUERY_CHUNKS, dim)
    doc_vectors = vectors[len(query_texts):].reshape(len(doc_ids), DOC_CHUNKS, dim)
    return maxsim_matrix(query_vectors, doc_vectors), doc_ids


def contrastive_loss(scores: torch.Tensor, batch: list, doc_ids: list, relevant: dict) -> torch.Tensor:
    """InfoNCE with the query's own positive as target; other relevant candidates are masked out."""
    group = len(doc_ids) // len(batch)
    targets = torch.arange(len(batch), device=scores.device) * group
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for i, (query_id, positive, negatives) in enumerate(batch):
        for j, doc_id in enumerate(doc_ids):
            if j != i * group and doc_id in relevant[query_id]:
                mask[i, j] = True
    logits = (scores / TAU).masked_fill(mask, -1e4)
    return torch.nn.functional.cross_entropy(logits, targets)


def margin_mse_loss(scores: torch.Tensor, batch: list, teacher: dict) -> torch.Tensor:
    """MSE between student and teacher margins s(q, pos) - s(q, neg) over each query's own negatives."""
    group = scores.shape[1] // len(batch)
    student_margins, teacher_margins = [], []
    for i, (query_id, positive, negatives) in enumerate(batch):
        for n, negative in enumerate(negatives, start=1):
            student_margins.append((scores[i, i * group] - scores[i, i * group + n]) / TAU)
            teacher_margins.append(teacher[query_id][positive] - teacher[query_id][negative])
    target = torch.tensor(teacher_margins, device=scores.device, dtype=scores.dtype)
    return torch.nn.functional.mse_loss(torch.stack(student_margins), target)


def train_epoch(model, tokenizer, optimizer, examples: list, data: dict, settings: dict, device) -> float:
    """One pass over the examples; returns the mean loss. data = {"chunks", "relevant", "teacher" or None}."""
    model.train()
    rng = random.Random(settings["seed"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    for start in range(0, len(examples), settings["batch_size"]):
        batch = examples[start:start + settings["batch_size"]]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            scores, doc_ids = batch_scores(model, tokenizer, batch, data["chunks"], device, rng, data.get("selection"))
            loss = contrastive_loss(scores.float(), batch, doc_ids, data["relevant"])
            if data.get("teacher") is not None:
                loss = loss + settings["distill_weight"] * margin_mse_loss(scores.float(), batch, data["teacher"])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        if len(losses) % 50 == 0:
            print(f"    step {len(losses)}: loss {np.mean(losses[-50:]):.4f}")
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def encode_all(model, tokenizer, texts: list, device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batch_vectors = encode_batch(model, tokenizer, texts[start:start + batch_size], device)
        vectors.append(batch_vectors.float().cpu().numpy())
    return np.concatenate(vectors)


def dense_dev_run(model, tokenizer, dev_chunks: dict, dev_queries: dict, device, depth: int = 100) -> dict:
    """Rank the dev pool for every dev query with MaxSim over all chunks (the query itself skipped)."""
    names = sorted(dev_chunks)
    texts, offsets = [], [0]
    for name in names:
        texts.extend(dev_chunks[name])
        offsets.append(len(texts))
    vectors = encode_all(model, tokenizer, texts, device)
    offsets = np.array(offsets)
    position = {name: i for i, name in enumerate(names)}
    run = {}
    for query_id in dev_queries:
        i = position[query_id]
        similarity = vectors[offsets[i]:offsets[i + 1]] @ vectors.T
        scores = np.maximum.reduceat(similarity, offsets[:-1], axis=1).mean(axis=0)
        scores[i] = -np.inf
        order = np.argsort(-scores)[:depth]
        run[query_id] = [names[j] for j in order]
    return run


def chunk_vectors(encoder_dir: str, chunk_sets: list, device) -> dict:
    """{doc_id: [n_chunks x dim] unit vectors} from a fixed encoder, used only to SELECT chunk pairs
    (student training here, the teacher in distill.py); the model being trained is never used for this."""
    tokenizer, model = load_encoder(encoder_dir)
    model.to(device)
    vectors = {}
    for chunks in chunk_sets:
        names = sorted(chunks)
        texts, offsets = [], [0]
        for name in names:
            texts.extend(chunks[name])
            offsets.append(len(texts))
        encoded = encode_all(model, tokenizer, texts, device, batch_size=128)
        for i, name in enumerate(names):
            vectors[name] = encoded[offsets[i]:offsets[i + 1]]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  selection vectors for {len(vectors)} judgments")
    return vectors
