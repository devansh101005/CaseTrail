"""Print the highest-document-frequency terms of the PCR train candidates (used to choose legal boilerplate stop words).

Only the train split is used, so the stop-word list is not tuned on test.
"""
import argparse
import time
from collections import Counter

from casetrail.data import doc_text, load_pcr
from casetrail.text.tokenizer import tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=150)
    args = parser.parse_args()

    candidates = load_pcr("train")["candidates"]
    start = time.time()
    df = Counter()
    for cand in candidates.values():
        df.update(set(tokenize(doc_text(cand))))
    print(f"Tokenised {len(candidates)} docs in {time.time() - start:.1f} s; vocabulary {len(df)}")
    for term, count in df.most_common(args.top):
        print(f"{term:20s} {count:5d}  {count / len(candidates):.2f}")


if __name__ == "__main__":
    main()
