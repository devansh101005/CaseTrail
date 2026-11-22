"""Reporter citations in Supreme Court judgments (used for leakage checks and cleaning; Lecture: Vocabulary - normalisation).

IL-TUR masks most case citations as <CITATION>, but some survive: in the body ("AIR 1974 SC 753",
"[1962] 3 SCR 618") and in the SCR header ("CITATION: 1966 AIR 1053 ...", "CITATOR INFO : F 1967 SC 819").
A case is identified here by a key (reporter, year, page).
"""
import re

# Header forms: "1966 AIR 1053", "1966 SCR (2) 944", "1989 SCR Supl. (1) 292"
AIR_HEADER = re.compile(r"\b(\d{4})\s+AIR\s*(\d+)")
SCR_HEADER = re.compile(r"\b(\d{4})\s+SCR\s*(?:Supl\.\s*)?\(\s*\d+\s*\)\s*(\d+)")
# Citator entries: a treatment code then "1967 SC 819", e.g. "F 1967 SC 819 (5)"
AIR_CITATOR = re.compile(r"\b[A-Z]{1,3}\s+(\d{4})\s*SC\s*(\d+)")
# Body forms: "AIR 1974 SC 753", "[1962] 3 SCR 618", "(1965) 3 S.C.R. 411"
AIR_BODY = re.compile(r"\bAIR\s*(\d{4})\s*SC\s*(\d+)")
SCR_BODY = re.compile(r"[\[\(](\d{4})[\]\)]\s*(?:Supp\.\s*)?\(?\d*\)?\s*S\.?C\.?R\.?\s*(\d+)")

# A header field runs until the next upper-case "FIELD:" label (ACT:, HEADNOTE:, CITATOR INFO :); length is capped
HEADER_CITATION = re.compile(r"CITATION\s*:(.{0,400}?)(?=[A-Z][A-Z ]{2,}\s?:|$)")
HEADER_CITATOR = re.compile(r"CITATOR INFO\s*:(.{0,3000}?)(?=[A-Z][A-Z ]{2,}\s?:|$)")

# Everything that looks like a reporter citation, for removal from text before indexing
ANY_REPORTER = re.compile(
    r"\bAIR\s*\d{4}\s*[A-Z][A-Za-z]*\.?\s*\d+"         # AIR 1974 SC 753, AIR 1919 Cal 904
    r"|\b\d{4}\s+AIR\s*\d+"                             # 1966 AIR 1053
    r"|[\[\(]\d{4}[\]\)]\s*(?:Supp\.\s*)?\(?\d*\)?\s*S\.?C\.?[RC]\.?\s*\d+"  # [1962] 3 SCR 618, (1998) 1 SCC 219
    r"|\b\d{4}\s+SC[RC]\s*(?:Supl\.\s*)?\(\s*\d+\s*\)\s*\d+"               # 1966 SCR (2) 944
    r"|\b\d{4}\s*\(\d+\)\s*SC[RC]\s*\d+"                                    # 1998(1) SCC 219
    r"|\b[A-Z]{1,3}\s+\d{4}\s*SC\s*\d+(?:\s*\([\d,]+\))?"                    # citator: R 1969 SC 840 (11)
)


def header_text(sentences: list) -> str:
    """The SCR header, when present, sits in the first one or two list items."""
    return " ".join(sentences[:2])


def own_citation_keys(sentences: list) -> set:
    """Keys of the judgment itself, read from its 'CITATION:' header field."""
    match = HEADER_CITATION.search(header_text(sentences))
    if match is None:
        return set()
    field = match.group(1)
    keys = {("AIR", year, page) for year, page in AIR_HEADER.findall(field)}
    for year, page in SCR_HEADER.findall(field):
        keys.add(("SCR", year, page))
    return keys


def citator_keys(sentences: list) -> set:
    """Keys of LATER judgments that cite this one ('CITATOR INFO' header field)."""
    match = HEADER_CITATOR.search(header_text(sentences))
    if match is None:
        return set()
    return {("AIR", year, page) for year, page in AIR_CITATOR.findall(match.group(1))}


def body_citation_keys(sentences: list) -> set:
    """Keys of judgments cited in the body that escaped the <CITATION> mask."""
    text = " ".join(sentences)
    keys = {("AIR", year, page) for year, page in AIR_BODY.findall(text)}
    for year, page in SCR_BODY.findall(text):
        keys.add(("SCR", year, page))
    return keys


def strip_citations(text: str) -> str:
    """Remove header citation fields and unmasked reporter citations (they can leak relevance labels)."""
    text = HEADER_CITATOR.sub(" ", text)
    text = HEADER_CITATION.sub(" ", text)
    return ANY_REPORTER.sub(" <CITATION> ", text)
