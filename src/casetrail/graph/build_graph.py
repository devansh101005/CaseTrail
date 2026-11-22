"""Graphs over judgments for link analysis (Lectures: PageRank, HITS; LLD section 6).

Nodes are judgment ids (strings) mapped to 0..N-1. Three graphs:
  citation graph  - train query -> each relevant (cited) candidate, from TRAIN labels only;
  G2 bipartite    - judgment -> legal reference it mentions (sec_302, art_21), weight = mention count;
  G3 similarity   - judgment -> its top-n BM25 neighbours in the same pool, weight = share of the neighbour scores.
"""
from collections import Counter


def node_ids(names) -> dict:
    """Map node names to 0..N-1 in sorted order (deterministic)."""
    return {name: i for i, name in enumerate(sorted(set(names)))}


def citation_edges(train_queries: dict) -> tuple:
    """(node id map, [(query, cited candidate, 1.0)]) from the training relevance labels."""
    names = []
    pairs = []
    for query_id, query in train_queries.items():
        names.append(query_id)
        for cand_id in query["relevant"]:
            names.append(cand_id)
            pairs.append((query_id, cand_id))
    ids = node_ids(names)
    return ids, [(ids[source], ids[target], 1.0) for source, target in pairs]


def legal_reference_edges(item_tokens: dict, min_df: int = 2) -> tuple:
    """Bipartite judgment -> reference edges. Returns (judgment ids, reference ids, [(judgment, reference, count)])."""
    counts = {}
    df = Counter()
    for doc_id, items in item_tokens.items():
        refs = Counter()
        for item in items:
            refs.update(token for token in item if token.startswith(("sec_", "art_")))
        counts[doc_id] = refs
        df.update(refs.keys())
    # A reference mentioned by a single judgment links nothing
    kept = [ref for ref, n in df.items() if n >= min_df]
    judgment_ids = node_ids(item_tokens)
    reference_ids = node_ids(kept)
    edges = []
    for doc_id, refs in counts.items():
        for ref, count in refs.items():
            if ref in reference_ids:
                edges.append((judgment_ids[doc_id], reference_ids[ref], float(count)))
    return judgment_ids, reference_ids, edges


def similarity_edges(neighbour_run: dict, doc_names: list) -> tuple:
    """G3 from {doc: [(neighbour, bm25 score)]}: weight = neighbour score / sum of the doc's neighbour scores."""
    ids = node_ids(doc_names)
    edges = []
    for doc_id, neighbours in neighbour_run.items():
        total = sum(score for name, score in neighbours if score > 0)
        for name, score in neighbours:
            if score > 0:
                edges.append((ids[doc_id], ids[name], score / total))
    return ids, edges
