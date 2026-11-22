"""Normalisation and tokenisation of judgments (Lecture: Vocabulary - tokens, normalisation, stop words, stemming).

Pipeline (LLD section 3): strip leftover citations -> drop mask tags -> NFC + lower-case ->
protect legal references (sec_302, art_21) -> split on [a-z0-9_]+ -> drop 1-char tokens and stop words ->
optional light stemming. Word bigrams can be added afterwards with add_bigrams().
"""
import re
import unicodedata

from casetrail.text.citations import strip_citations
from casetrail.text.legal_refs import protect_legal_refs
from casetrail.text.stemmer import stem
from casetrail.text.stopwords import STOP_WORDS

MASK_TAG = re.compile(r"<[A-Z_]+>")
TOKEN = re.compile(r"[a-z0-9_]+")


def normalise(text: str, keep_citations: bool = False) -> str:
    # keep_citations=True is only for the leakage ablation (scripts/ablation_citation_strip.py)
    if not keep_citations:
        text = strip_citations(text)      # needs the original case (AIR, SCR)
    text = MASK_TAG.sub(" ", text)        # <NAME>, <ORG>, <CITATION>, <ENTITY>, <SECTION>, <ACT>
    text = unicodedata.normalize("NFC", text).lower()
    return protect_legal_refs(text)


def tokenize(text: str, use_stemming: bool = False, keep_citations: bool = False) -> list:
    """Index words of a text, in order."""
    words = []
    for token in TOKEN.findall(normalise(text, keep_citations)):
        if len(token) < 2 or token in STOP_WORDS:
            continue
        words.append(stem(token) if use_stemming else token)
    return words


def add_bigrams(words: list) -> list:
    """Unigrams followed by word bigrams ("dowry__death"); a bigram keeps phrase information BM25 otherwise loses."""
    return words + [f"{words[i]}__{words[i + 1]}" for i in range(len(words) - 1)]
