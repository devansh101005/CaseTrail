"""Cross-encoder teacher for PCR (beyond syllabus: re-ranking, knowledge distillation).

The teacher reads a query chunk and a candidate chunk together ([CLS] q [SEP] d [SEP]) and outputs one
relevance logit, so it can compare the two texts word by word (more accurate, but far too slow for a whole pool).
A judgment pair is scored by the max logit over a few chunk pairs; trained with binary cross-entropy
(relevant candidate = 1, BM25 hard negative = 0). Its scores then teach the bi-encoder (Margin-MSE, see
train_retriever.py), and it re-ranks the BM25 top-50 as a quality reference.

Which chunk pairs it sees matters. Run 1 drew 3 x 3 RANDOM chunks: two random parts of two long judgments rarely
contain the legal point that links them, the labels looked like noise, and the loss stayed at ln 2 (chance).
With `vectors` (chunk embeddings from a bi-encoder) the teacher instead sees the SELECTED_PAIRS most similar
(query chunk, candidate chunk) pairs, the parts most likely to explain the citation.
"""
import random

import numpy as np
import torch

from casetrail.neural.train_retriever import sample_chunks

PAIR_QUERY_CHUNKS = 3
PAIR_DOC_CHUNKS = 3
SELECTED_PAIRS = 4


def random_pairs(query_chunks: list, doc_chunks: list, rng: random.Random) -> list:
    pairs = []
    for query_text in sample_chunks(query_chunks, PAIR_QUERY_CHUNKS, rng):
        for doc_text in sample_chunks(doc_chunks, PAIR_DOC_CHUNKS, rng):
            pairs.append((query_text, doc_text))
    return pairs


def selected_pairs(query_chunks: list, doc_chunks: list, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> list:
    """The SELECTED_PAIRS chunk pairs with the highest cosine (repeated if a judgment has very few chunks)."""
    similarity = query_vectors @ doc_vectors.T
    best = np.argsort(-similarity, axis=None)[:SELECTED_PAIRS]
    rows, cols = np.unravel_index(best, similarity.shape)
    found = [(query_chunks[i], doc_chunks[j]) for i, j in zip(rows.tolist(), cols.tolist())]
    pairs = []
    for n in range(SELECTED_PAIRS):
        pairs.append(found[n % len(found)])
    return pairs


def chunk_pairs(query_id: str, doc_id: str, chunks: dict, rng: random.Random, vectors: dict = None) -> list:
    if vectors is None:
        return random_pairs(chunks[query_id], chunks[doc_id], rng)
    return selected_pairs(chunks[query_id], chunks[doc_id], vectors[query_id], vectors[doc_id])


def pair_logits(model, tokenizer, pairs: list, device, max_tokens: int = 512) -> torch.Tensor:
    inputs = tokenizer([q for q, d in pairs], [d for q, d in pairs], padding=True, truncation="longest_first",
                       max_length=max_tokens, return_tensors="pt").to(device)
    return model(**inputs).logits.squeeze(-1)


def doc_pair_scores(model, tokenizer, doc_pairs: list, chunks: dict, device, rng: random.Random,
                    vectors: dict = None) -> torch.Tensor:
    """Max chunk-pair logit for each (query_id, doc_id); every judgment pair contributes the same number of pairs."""
    all_pairs = []
    for query_id, doc_id in doc_pairs:
        all_pairs.extend(chunk_pairs(query_id, doc_id, chunks, rng, vectors))
    logits = pair_logits(model, tokenizer, all_pairs, device)
    per_doc = len(all_pairs) // len(doc_pairs)
    return logits.reshape(len(doc_pairs), per_doc).max(dim=1).values


def teacher_examples(train_items: list, rng: random.Random) -> list:
    """[(query_id, doc_id, label)]: every positive, and as many hard negatives."""
    examples = []
    for item in train_items:
        negatives = item["hard_negatives"][:]
        rng.shuffle(negatives)
        for positive in item["positives"]:
            examples.append((item["query_id"], positive, 1.0))
        for negative in negatives[:len(item["positives"])]:
            examples.append((item["query_id"], negative, 0.0))
    rng.shuffle(examples)
    return examples


def set_learning_rate(optimizer, settings: dict, step: int):
    """Linear warm-up from 0 to settings['lr'] over settings['warmup_steps'] steps, then constant."""
    factor = min(1.0, step / max(settings.get("warmup_steps", 0), 1))
    for group in optimizer.param_groups:
        group["lr"] = settings["lr"] * factor


def train_teacher_epoch(model, tokenizer, optimizer, examples: list, chunks: dict, settings: dict, device,
                        vectors: dict = None) -> float:
    model.train()
    rng = random.Random(settings["seed"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    for start in range(0, len(examples), settings["batch_size"]):
        batch = examples[start:start + settings["batch_size"]]
        set_learning_rate(optimizer, settings, len(losses) + 1)
        labels = torch.tensor([label for q, d, label in batch], device=device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            scores = doc_pair_scores(model, tokenizer, [(q, d) for q, d, label in batch], chunks, device, rng, vectors)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores.float(), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        if len(losses) % 50 == 0:
            print(f"    step {len(losses)}: loss {np.mean(losses[-50:]):.4f}")
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def score_pairs(model, tokenizer, doc_pairs: list, chunks: dict, device, batch_size: int = 16, seed: int = 13,
                vectors: dict = None) -> list:
    """Teacher score for every (query_id, doc_id), in order (fixed seed, so repeatable)."""
    model.eval()
    rng = random.Random(seed)
    scores = []
    for start in range(0, len(doc_pairs), batch_size):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batch_scores = doc_pair_scores(model, tokenizer, doc_pairs[start:start + batch_size], chunks, device, rng,
                                           vectors)
        scores.extend(batch_scores.float().cpu().tolist())
    return scores


def rerank_run(model, tokenizer, base_run: dict, chunks: dict, device, depth: int = 50, vectors: dict = None) -> dict:
    """Re-rank the top `depth` of a run with the teacher; the rest keeps its order below."""
    reranked = {}
    for query_id, ranked in base_run.items():
        top = ranked[:depth]
        scores = score_pairs(model, tokenizer, [(query_id, doc_id) for doc_id in top], chunks, device, vectors=vectors)
        order = sorted(zip(scores, top), key=lambda item: -item[0])
        reranked[query_id] = [doc_id for score, doc_id in order] + ranked[depth:]
    return reranked


def dev_pair_accuracy(model, tokenizer, dev_qrels: dict, bm25_run: dict, chunks: dict, device, vectors: dict = None,
                      max_queries: int = 40) -> float:
    """How often a cited case outscores a BM25 hard negative on dev (0.5 = chance). A quick check that the
    teacher learned something, made before its scores are used for anything."""
    wins, total = 0, 0
    for query_id in sorted(dev_qrels)[:max_queries]:
        positives = [d for d in dev_qrels[query_id] if d in chunks][:3]
        negatives = [d for d in bm25_run[query_id] if d not in dev_qrels[query_id]][:3]
        if not positives or not negatives:
            continue
        scores = score_pairs(model, tokenizer, [(query_id, d) for d in positives + negatives], chunks, device,
                             vectors=vectors)
        for p in scores[:len(positives)]:
            for n in scores[len(positives):]:
                wins += int(p > n)
                total += 1
    return wins / total if total else 0.0
