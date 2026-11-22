"""Bi-encoder for long judgments: chunking, mean-pooled embeddings and MaxSim scoring (beyond syllabus: dense retrieval).

A judgment is cut into chunks of at most CHUNK_TOKENS word-piece tokens (whole sentences packed greedily),
at most MAX_CHUNKS_PER_DOC per judgment (the first chunk, then evenly spaced ones). Each chunk becomes one
unit-length vector (mean of the token vectors). Two judgments are compared with MaxSim:
  score(q, d) = mean over query chunks of ( max over d's chunks of cosine )
so every part of the query looks for its best-matching part of the candidate.
"""
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from casetrail import config
from casetrail.text.citations import strip_citations

MASK_TAG = re.compile(r"<[A-Z_]+>")
SPACES = re.compile(r"\s+")
MAX_SENTENCE_WORDS = 150   # longer text items are split into pieces of this many words before packing


def clean_sentence(text: str) -> str:
    """Same leakage-safe cleaning as the lexical pipeline, but keep the text readable for the encoder."""
    text = MASK_TAG.sub(" ", strip_citations(text))
    return SPACES.sub(" ", text).strip()


def sentence_pieces(sentences: list) -> list:
    pieces = []
    for sentence in sentences:
        words = clean_sentence(sentence).split()
        for start in range(0, len(words), MAX_SENTENCE_WORDS):
            pieces.append(" ".join(words[start:start + MAX_SENTENCE_WORDS]))
    return [piece for piece in pieces if piece]


def pick_evenly(chunks: list, limit: int) -> list:
    """Keep the first chunk and evenly spaced others, so long judgments are covered from start to end."""
    if len(chunks) <= limit:
        return chunks
    positions = np.linspace(0, len(chunks) - 1, limit).round().astype(int)
    return [chunks[i] for i in sorted(set(positions.tolist()))]


def chunk_judgment(sentences: list, tokenizer, max_tokens: int = config.CHUNK_TOKENS,
                   max_chunks: int = config.MAX_CHUNKS_PER_DOC) -> list:
    pieces = sentence_pieces(sentences)
    if not pieces:
        return [""]
    lengths = [len(ids) for ids in tokenizer(pieces, add_special_tokens=False)["input_ids"]]
    chunks, current, current_len = [], [], 0
    budget = max_tokens - 2  # room for [CLS] and [SEP]
    for piece, length in zip(pieces, lengths):
        if current and current_len + length > budget:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(piece)
        current_len += length
    chunks.append(" ".join(current))
    return pick_evenly(chunks, max_chunks)


def load_encoder(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def mean_pool(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(tokenizer, model, texts: list, batch_size: int = 32, prefix: str = "",
                 max_tokens: int = config.CHUNK_TOKENS) -> np.ndarray:
    """Unit-length float32 vectors, one per text (on whatever device the model sits on)."""
    device = next(model.parameters()).device
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        output = model(**inputs).last_hidden_state
        pooled = torch.nn.functional.normalize(mean_pool(output, inputs["attention_mask"]), dim=-1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors) if vectors else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def maxsim_scores(query_vectors: np.ndarray, chunk_vectors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Score of every document; doc i owns chunk_vectors[offsets[i]:offsets[i+1]]."""
    similarity = query_vectors @ chunk_vectors.T                       # [query chunks x all chunks]
    best_per_doc = np.maximum.reduceat(similarity, offsets[:-1], axis=1)  # [query chunks x docs]
    return best_per_doc.mean(axis=0)


def mean_vector_scores(query_vectors: np.ndarray, chunk_vectors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Alternative: one averaged vector per document, compared by cosine."""
    doc_means = np.add.reduceat(chunk_vectors, offsets[:-1], axis=0)
    doc_means /= np.maximum(np.linalg.norm(doc_means, axis=1, keepdims=True), 1e-12)
    query_mean = query_vectors.mean(axis=0)
    query_mean /= max(np.linalg.norm(query_mean), 1e-12)
    return doc_means @ query_mean
