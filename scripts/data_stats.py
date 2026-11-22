"""Dataset statistics for IL-TUR pcr + lsi -> results/data_stats.json (counts, lengths, labels, masking, overlaps).

Also measures two data issues found in P1:
  - citation leakage: reporter citations that escaped the <CITATION> mask and reveal relevant candidates;
  - overlap: near-duplicate judgments shared between splits (and between the PCR and LSI tasks).
"""
import argparse
import json
import re
from collections import Counter

from casetrail import config
from casetrail.data import load_lsi, load_pcr, load_statutes
from casetrail.text.citations import body_citation_keys, citator_keys, own_citation_keys

WORD = re.compile(r"[A-Za-z0-9]+")
TAG = re.compile(r"<[A-Z_]+>")
# Shingles start at these words, so the same passage gives the same shingles however the text was split into sentences
ANCHOR = re.compile(r"(?=which|therefore|evidence|accused|section)")
SHINGLE_CHARS = 60


def length_stats(docs: list) -> dict:
    sentences = [len(doc["sentences"]) for doc in docs]
    words = [len(WORD.findall(" ".join(doc["sentences"]))) for doc in docs]
    return {
        "docs": len(docs),
        "sentences_mean": round(sum(sentences) / len(docs), 1),
        "sentences_max": max(sentences),
        "words_mean": round(sum(words) / len(docs), 1),
        "words_median": sorted(words)[len(words) // 2],
        "words_max": max(words),
    }


def tag_counts(docs: list) -> dict:
    counts = Counter()
    for doc in docs:
        counts.update(TAG.findall(" ".join(doc["sentences"])))
    return dict(counts.most_common())


def citation_leakage(pcr_split: dict) -> dict:
    """How many relevant (query, candidate) pairs can be found from citation strings alone."""
    queries, candidates = pcr_split["queries"], pcr_split["candidates"]
    key_to_cands = {}
    for cand_id, cand in candidates.items():
        for key in own_citation_keys(cand["sentences"]):
            key_to_cands.setdefault(key, set()).add(cand_id)
    pairs = 0
    via_body = set()
    via_citator = set()
    for query_id, query in queries.items():
        relevant = set(query["relevant"])
        pairs += len(relevant)
        found = set()
        for key in body_citation_keys(query["sentences"]):
            found |= key_to_cands.get(key, set())
        for cand_id in found & relevant:
            via_body.add((query_id, cand_id))
        query_keys = own_citation_keys(query["sentences"])
        for cand_id in relevant:
            if cand_id in candidates and citator_keys(candidates[cand_id]["sentences"]) & query_keys:
                via_citator.add((query_id, cand_id))
    return {
        "relevant_pairs": pairs,
        "candidates_with_own_citation_header": sum(1 for c in candidates.values() if own_citation_keys(c["sentences"])),
        "pairs_found_via_unmasked_body_citations": len(via_body),
        "pairs_found_via_candidate_citator_info": len(via_citator),
        "pairs_found_either_way": len(via_body | via_citator),
        "fraction_found_either_way": round(len(via_body | via_citator) / pairs, 4),
    }


def pcr_split_stats(pcr_split: dict) -> dict:
    queries, candidates = pcr_split["queries"], pcr_split["candidates"]
    relevant_counts = [len(q["relevant"]) for q in queries.values()]
    missing = sum(1 for q in queries.values() for c in q["relevant"] if c not in candidates)
    self_relevant = sum(1 for q_id, q in queries.items() if q_id in q["relevant"])
    return {
        "queries": length_stats(list(queries.values())),
        "candidates": length_stats(list(candidates.values())),
        "relevant_per_query_mean": round(sum(relevant_counts) / len(relevant_counts), 2),
        "relevant_per_query_min": min(relevant_counts),
        "relevant_per_query_max": max(relevant_counts),
        "relevant_ids_not_in_candidate_pool": missing,
        "queries_also_in_candidate_pool": sum(1 for q_id in queries if q_id in candidates),
        "queries_listing_themselves_as_relevant": self_relevant,
        "mask_tags_candidates": tag_counts(list(candidates.values())),
        "citation_leakage": citation_leakage(pcr_split),
    }


def lsi_split_stats(docs: list, statute_ids: list) -> dict:
    label_counts = [len(doc["statutes"]) for doc in docs]
    freq = Counter()
    for doc in docs:
        freq.update(doc["statutes"])
    per_statute = {s: freq.get(s, 0) for s in statute_ids}
    ordered = sorted(per_statute.items(), key=lambda item: -item[1])
    return {
        "docs": length_stats(docs),
        "labels_per_doc_mean": round(sum(label_counts) / len(docs), 2),
        "labels_per_doc_max": max(label_counts),
        "docs_without_labels": sum(1 for n in label_counts if n == 0),
        "statutes_used": sum(1 for n in per_statute.values() if n > 0),
        "top10_statutes": ordered[:10],
        "rarest10_statutes": ordered[-10:],
        "mask_tags": tag_counts(docs),
        "label_frequency": per_statute,
    }


def shingles(sentences: list) -> set:
    """Hashes of 60-char pieces of the text (lower-case letters/digits only, mask tags removed)."""
    text = re.sub(r"[^a-z0-9]", "", TAG.sub("", " ".join(sentences)).lower())
    found = set()
    for match in ANCHOR.finditer(text):
        piece = text[match.start():match.start() + SHINGLE_CHARS]
        if len(piece) == SHINGLE_CHARS:
            found.add(hash(piece))
    return found


def shingle_pool(docs: list) -> set:
    """All shingles of a document collection, minus boilerplate shingles that occur in 3 or more documents."""
    df = Counter()
    for doc in docs:
        df.update(shingles(doc["sentences"]))
    return {h for h, n in df.items() if n < 3}


def near_duplicates(docs_a: list, pool_b: set, min_containment: float = 0.5) -> int:
    """Number of docs in A whose shingles mostly (>= 50%) also appear somewhere in collection B."""
    count = 0
    for doc in docs_a:
        own = shingles(doc["sentences"])
        if len(own) >= 10 and len(own & pool_b) / len(own) >= min_containment:
            count += 1
    return count


def overlap_stats(pcr: dict, lsi: dict) -> dict:
    stats = {}
    pcr_docs = {}
    for split in config.PCR_SPLITS:
        pool = dict(pcr[split]["candidates"])
        pool.update(pcr[split]["queries"])
        pcr_docs[split] = list(pool.values())
    ids = {split: {d["id"] for d in pcr_docs[split]} for split in config.PCR_SPLITS}
    stats["pcr_test_ids_in_train_or_dev"] = len(ids["test"] & (ids["train"] | ids["dev"]))
    stats["pcr_dev_ids_in_train"] = len(ids["dev"] & ids["train"])
    lsi_ids = {split: {d["id"] for d in lsi[split]} for split in config.LSI_SPLITS}
    stats["lsi_test_ids_in_train_or_dev"] = len(lsi_ids["test"] & (lsi_ids["train"] | lsi_ids["dev"]))

    print("  shingling PCR pools ...")
    pcr_train_dev = shingle_pool(pcr_docs["train"] + pcr_docs["dev"])
    pcr_test_queries = list(pcr["test"]["queries"].values())
    stats["pcr_test_queries_near_dup_in_pcr_train_dev"] = near_duplicates(pcr_test_queries, pcr_train_dev)
    stats["pcr_test_pool_near_dup_in_pcr_train_dev"] = near_duplicates(pcr_docs["test"], pcr_train_dev)
    print("  shingling LSI train ...")
    lsi_train = shingle_pool(lsi["train"])
    stats["lsi_test_near_dup_in_lsi_train"] = near_duplicates(lsi["test"], lsi_train)
    stats["lsi_dev_near_dup_in_lsi_train"] = near_duplicates(lsi["dev"], lsi_train)
    stats["pcr_test_pool_near_dup_in_lsi_train"] = near_duplicates(pcr_docs["test"], lsi_train)
    lsi_test = shingle_pool(lsi["test"])
    stats["pcr_train_pool_near_dup_in_lsi_test"] = near_duplicates(pcr_docs["train"], lsi_test)
    stats["near_dup_rule"] = "a doc counts if >= 50% of its 60-char anchored shingles occur in the other collection"
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "data_stats.json"))
    parser.add_argument("--skip-overlap", action="store_true", help="skip the slower near-duplicate check")
    args = parser.parse_args()

    stats = {"pcr": {}, "lsi": {}}
    pcr = {}
    for split in config.PCR_SPLITS:
        print(f"PCR {split} ...")
        pcr[split] = load_pcr(split)
        stats["pcr"][split] = pcr_split_stats(pcr[split])

    statutes = load_statutes()
    statute_ids = list(statutes)
    stats["lsi"]["statutes"] = len(statutes)
    lsi = {}
    for split in config.LSI_SPLITS:
        print(f"LSI {split} ...")
        lsi[split] = load_lsi(split)
        stats["lsi"][split] = lsi_split_stats(lsi[split], statute_ids)

    if not args.skip_overlap:
        print("Overlap checks ...")
        stats["overlap"] = overlap_stats(pcr, lsi)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
