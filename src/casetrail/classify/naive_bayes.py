"""Multinomial Naive Bayes, one-vs-rest over statutes (Lecture: Naive Bayes text classification).

For statute c the two classes are c and not-c. With Laplace (add-alpha) smoothing:
  theta_tc = (count(t, c) + alpha) / (sum_t' count(t', c) + alpha * V)
  score_c(d) = log P(c) - log P(not c) + sum_t tf_td * (log theta_tc - log theta_t,notc)
i.e. the log-odds that d belongs to c. Logs avoid underflow from multiplying thousands of small probabilities.
"""
import numpy as np


def term_label_sums(vectors: list, labels: np.ndarray, num_terms: int) -> np.ndarray:
    """S[t, c] = sum over documents labelled c of the document's value for term t (tf or tf-idf weight)."""
    sums = np.zeros((num_terms, labels.shape[1]), dtype=np.float64)
    for (term_ids, values), label_row in zip(vectors, labels):
        columns = np.flatnonzero(label_row)
        if len(columns) and len(term_ids):
            np.add.at(sums, (term_ids[:, None], columns[None, :]), values[:, None])
    return sums


def train_naive_bayes(vectors: list, labels: np.ndarray, num_terms: int, alpha: float) -> dict:
    counts_in = term_label_sums(vectors, labels, num_terms)        # count(t, c)
    total = np.zeros(num_terms, dtype=np.float64)
    for term_ids, tfs in vectors:
        np.add.at(total, term_ids, tfs)
    counts_out = total[:, None] - counts_in                        # count(t, not c)
    log_theta_in = np.log(counts_in + alpha) - np.log(counts_in.sum(axis=0) + alpha * num_terms)
    log_theta_out = np.log(counts_out + alpha) - np.log(counts_out.sum(axis=0) + alpha * num_terms)
    docs_in = labels.sum(axis=0)
    prior = np.log(docs_in) - np.log(len(labels) - docs_in)
    return {"weights": log_theta_in - log_theta_out, "prior": prior}


def naive_bayes_scores(vectors: list, model: dict) -> np.ndarray:
    """Log-odds score of every statute for every document: [num_docs x num_statutes]."""
    scores = np.zeros((len(vectors), len(model["prior"])), dtype=np.float64)
    for d, (term_ids, tfs) in enumerate(vectors):
        scores[d] = model["prior"] + tfs @ model["weights"][term_ids]
    return scores
