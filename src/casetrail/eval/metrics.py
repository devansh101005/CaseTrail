"""Evaluation metrics (Lecture: Evaluation - precision, recall, F1, P@K, MAP, nDCG, MRR; macro/micro averaging).

PCR headline metric, micro-F1@K (IL-TUR / U-CREAT App. A):
  P = (# correctly retrieved over all queries) / (# retrieved over all queries)
  R = (# correctly retrieved over all queries) / (# relevant over all queries)
  F1 = 2PR / (P + R), with the top K candidates retrieved for every query; K is chosen on dev.
A run is {query_id: [doc ids, best first]}; qrels is {query_id: set of relevant doc ids}.
"""
import math


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def micro_prf_at_k(run: dict, qrels: dict, k: int) -> dict:
    correct = retrieved = relevant = 0
    for query_id, relevant_ids in qrels.items():
        top = run.get(query_id, [])[:k]
        correct += len(set(top) & relevant_ids)
        retrieved += len(top)
        relevant += len(relevant_ids)
    precision = correct / retrieved if retrieved else 0.0
    recall = correct / relevant if relevant else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall)}


def best_k(run: dict, qrels: dict, k_values: list) -> tuple:
    """The K with the highest micro-F1@K (only ever called on dev); returns (k, f1)."""
    best = (k_values[0], -1.0)
    for k in k_values:
        score = micro_prf_at_k(run, qrels, k)["f1"]
        if score > best[1]:
            best = (k, score)
    return best


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked: list, relevant: set) -> float:
    """Mean of the precision values at the rank of each relevant document (unretrieved ones count as 0)."""
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Binary-relevance nDCG: DCG = sum rel_i / log2(i + 1), divided by the DCG of a perfect ranking."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked[:k], start=1) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(ranked: list, relevant: set) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_over_queries(run: dict, qrels: dict, metric, k: int = None) -> float:
    """Average a per-query metric over all judged queries; k is passed on for the @K metrics."""
    values = []
    for query_id, relevant in qrels.items():
        ranked = run.get(query_id, [])
        values.append(metric(ranked, relevant) if k is None else metric(ranked, relevant, k))
    return sum(values) / len(values) if values else 0.0


def multilabel_f1(predicted: list, gold: list, labels: list) -> dict:
    """Macro and micro precision/recall/F1 for multi-label classification (LSI).

    predicted[i] and gold[i] are sets of labels of document i. Macro = mean over labels of the per-label F1
    (a label with no true positives gets F1 = 0, as in scikit-learn); micro pools all decisions.
    """
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    for pred_set, gold_set in zip(predicted, gold):
        for label in pred_set:
            if label in gold_set:
                tp[label] += 1
            else:
                fp[label] += 1
        for label in gold_set - pred_set:
            fn[label] += 1
    per_label = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        per_label[label] = {"precision": p, "recall": r, "f1": f1(p, r), "support": tp[label] + fn[label]}
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    count = len(labels)
    return {
        "macro_precision": sum(v["precision"] for v in per_label.values()) / count,
        "macro_recall": sum(v["recall"] for v in per_label.values()) / count,
        "macro_f1": sum(v["f1"] for v in per_label.values()) / count,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": f1(micro_p, micro_r),
        "per_label": per_label,
    }


def pcr_run_metrics(run: dict, qrels: dict, k: int) -> dict:
    """All PCR metrics we report for one run: micro P/R/F1@K at the dev-chosen K, P@K / R@K for K = 5, 10, 20
    (averaged over queries), MAP over the run depth (100), nDCG@10 and MRR."""
    micro = micro_prf_at_k(run, qrels, k)
    metrics = {"micro_f1@k": micro["f1"], "micro_precision@k": micro["precision"], "micro_recall@k": micro["recall"],
               "k": k}
    for cutoff in [5, 10, 20]:
        metrics[f"p@{cutoff}"] = mean_over_queries(run, qrels, precision_at_k, cutoff)
        metrics[f"r@{cutoff}"] = mean_over_queries(run, qrels, recall_at_k, cutoff)
    metrics["map@100"] = mean_over_queries(run, qrels, average_precision)
    metrics["ndcg@10"] = mean_over_queries(run, qrels, ndcg_at_k, 10)
    metrics["mrr"] = mean_over_queries(run, qrels, reciprocal_rank)
    return metrics
