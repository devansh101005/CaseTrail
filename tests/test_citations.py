"""Reporter-citation parsing and stripping on hand-made header/body snippets."""
from casetrail.text.citations import body_citation_keys, citator_keys, own_citation_keys, strip_citations

HEADER = ("PETITIONER: <ORG> Vs. RESPONDENT: COMMISSIONER OF INCOME-TAX DATE OF JUDGMENT: 08/12/1965 "
          "BENCH: <NAME>, S.M. CITATION: 1966 AIR 1053 1966 SCR (2) 944 "
          "CITATOR INFO : F 1967 SC 819 (5) R 1969 SC 840 (11) ACT: Indian Income-tax Act , 1922")
BODY = "In State of U.P. v. <NAME>, AIR 1974 SC 753 (757) observed and in <CITATION> [1962] 3 SCR 618 it was held"


def test_own_citation_keys():
    assert own_citation_keys([HEADER, "1. Facts."]) == {("AIR", "1966", "1053"), ("SCR", "1966", "944")}


def test_citator_keys():
    assert citator_keys([HEADER]) == {("AIR", "1967", "819"), ("AIR", "1969", "840")}


def test_body_citation_keys():
    assert body_citation_keys([BODY]) == {("AIR", "1974", "753"), ("SCR", "1962", "618")}


def test_no_header_gives_no_keys():
    assert own_citation_keys(["JUDGMENT <NAME>, J.", "1. This appeal ..."]) == set()


def test_strip_removes_citations_but_keeps_other_text():
    cleaned = strip_citations(HEADER)
    assert "1053" not in cleaned and "819" not in cleaned and "840" not in cleaned
    assert "Indian Income-tax Act" in cleaned and "PETITIONER" in cleaned
    body = strip_citations(BODY)
    assert "753" not in body and "618" not in body
    assert "observed" in body and "it was held" in body
