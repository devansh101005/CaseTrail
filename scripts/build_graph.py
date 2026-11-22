"""Build the link-analysis graphs and their PageRank / HITS scores -> indexes/graph/ + results/graph_stats.json.

PCR (per split pool): citation graph from TRAIN labels (PageRank), G2 judgment -> legal reference (HITS),
G3 BM25 similarity graph (PageRank). LSI: judgment -> statute graph from LSI TRAIN labels (HITS -> statute authority).
"""
import argparse
import json

from casetrail import config
from casetrail.data import load_lsi, load_pcr
from casetrail.graph.build_graph import citation_edges, legal_reference_edges, node_ids, similarity_edges
from casetrail.graph.hits import hits
from casetrail.graph.pagerank import pagerank
from casetrail.retrieval.pcr_lexical import load_item_tokens, rank_split

GRAPH_DIR = config.INDEX_DIR / "graph"
NEIGHBOURS = 10


def scores_by_name(ids: dict, scores) -> dict:
    return {name: float(scores[i]) for name, i in ids.items()}


def top_items(score_map: dict, n: int = 10) -> list:
    ordered = sorted(score_map.items(), key=lambda item: -item[1])
    return [[name, round(score, 6)] for name, score in ordered[:n]]


def save_scores(name: str, score_map: dict):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(score_map, f)


def citation_graph(stats: dict):
    ids, edges = citation_edges(load_pcr("train")["queries"])
    scores, iterations = pagerank(len(ids), edges, config.PAGERANK_DAMPING, config.PAGERANK_TOL,
                                  config.PAGERANK_MAX_ITER)
    score_map = scores_by_name(ids, scores)
    save_scores("pcr_citation_pagerank", score_map)
    stats["pcr_citation_graph"] = {"built_from": "train relevance labels only", "nodes": len(ids),
                                   "edges": len(edges), "pagerank_iterations": iterations,
                                   "top_pagerank": top_items(score_map)}
    print(f"citation graph: {len(ids)} nodes, {len(edges)} edges, PageRank in {iterations} iterations")


def reference_graph(split: str, stem: bool, stats: dict):
    judgment_ids, reference_ids, edges = legal_reference_edges(load_item_tokens(split, stem))
    hub, authority, iterations = hits(len(judgment_ids), len(reference_ids), edges, config.HITS_TOL,
                                      config.HITS_MAX_ITER)
    save_scores(f"pcr_{split}_reference_hub", scores_by_name(judgment_ids, hub))
    authorities = scores_by_name(reference_ids, authority)
    stats[f"pcr_{split}_reference_graph"] = {"judgments": len(judgment_ids), "references": len(reference_ids),
                                             "edges": len(edges), "hits_iterations": iterations,
                                             "top_authorities": top_items(authorities)}
    print(f"{split} G2: {len(reference_ids)} references, {len(edges)} edges, HITS in {iterations} iterations")


def similarity_graph(split: str, system: dict, stats: dict):
    tokens = load_item_tokens(split, system["stem"])
    neighbour_run = rank_split(split, system, tokens, sorted(tokens), {}, with_scores=True, depth=NEIGHBOURS)
    ids, edges = similarity_edges(neighbour_run, sorted(tokens))
    scores, iterations = pagerank(len(ids), edges, config.PAGERANK_DAMPING, config.PAGERANK_TOL,
                                  config.PAGERANK_MAX_ITER)
    score_map = scores_by_name(ids, scores)
    save_scores(f"pcr_{split}_similarity_pagerank", score_map)
    stats[f"pcr_{split}_similarity_graph"] = {"neighbours_per_judgment": NEIGHBOURS, "nodes": len(ids),
                                              "edges": len(edges), "pagerank_iterations": iterations,
                                              "top_pagerank": top_items(score_map)}
    print(f"{split} G3: {len(ids)} nodes, {len(edges)} edges, PageRank in {iterations} iterations")


def lsi_statute_graph(stats: dict):
    docs = load_lsi("train")
    doc_ids = node_ids([doc["id"] for doc in docs])
    all_statutes = []
    for doc in docs:
        all_statutes.extend(doc["statutes"])
    statute_ids = node_ids(all_statutes)
    edges = []
    for doc in docs:
        for statute in doc["statutes"]:
            edges.append((doc_ids[doc["id"]], statute_ids[statute], 1.0))
    hub, authority, iterations = hits(len(doc_ids), len(statute_ids), edges, config.HITS_TOL, config.HITS_MAX_ITER)
    authorities = scores_by_name(statute_ids, authority)
    save_scores("lsi_statute_authority", authorities)
    stats["lsi_statute_graph"] = {"built_from": "LSI train labels only", "judgments": len(doc_ids),
                                  "statutes": len(statute_ids), "edges": len(edges), "hits_iterations": iterations,
                                  "top_authorities": top_items(authorities)}
    print(f"LSI statute graph: {len(edges)} edges, HITS in {iterations} iterations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=config.PCR_SPLITS)
    args = parser.parse_args()

    with open(config.RESULTS_DIR / "tuned_params.json", encoding="utf-8") as f:
        pcr_params = json.load(f)["pcr"]
    stats = {}
    citation_graph(stats)
    for split in args.splits:
        reference_graph(split, pcr_params["stemming"], stats)
        similarity_graph(split, pcr_params["systems"]["bm25_bigram"], stats)
    lsi_statute_graph(stats)
    with open(config.RESULTS_DIR / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Wrote results/graph_stats.json")


if __name__ == "__main__":
    main()
