"""Build the PCR inverted indexes for each candidate pool -> indexes/pcr_<split>_<variant>/ (LLD section 4).

Variants: unigram / unigram+bigram terms, with or without light stemming, whole judgments and passages.
Tokens are cached in data/cache/ so each pool is tokenised only once per stemming setting.
"""
import argparse
import time

from casetrail import config
from casetrail.data import load_pcr
from casetrail.retrieval.pcr_lexical import build_doc_index, build_passages, load_item_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=config.PCR_SPLITS)
    parser.add_argument("--stem", nargs="+", default=["off", "on"], choices=["off", "on"])
    parser.add_argument("--ngrams", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    parser.add_argument("--no-passages", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        pool = load_pcr(split)["candidates"]
        for stem_setting in args.stem:
            stem = stem_setting == "on"
            item_tokens = load_item_tokens(split, stem, pool)
            for ngram in args.ngrams:
                start = time.time()
                index = build_doc_index(split, ngram, stem, item_tokens)
                print(f"{split} ngram={ngram} stem={stem}: {index.num_docs()} docs, {len(index.term_to_id)} terms, "
                      f"{len(index.doc_ids)} postings ({time.time() - start:.1f} s)")
            if not args.no_passages:
                start = time.time()
                index = build_passages(split, 1, stem, item_tokens, pool)
                print(f"{split} passages stem={stem}: {index.num_docs()} passages, avg {index.avg_len:.0f} terms "
                      f"({time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
