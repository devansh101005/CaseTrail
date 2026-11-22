"""Text pipeline on hand-made sentences."""
from casetrail.text.legal_refs import protect_legal_refs
from casetrail.text.stemmer import stem
from casetrail.text.tokenizer import add_bigrams, tokenize


def test_section_498a_ipc_survives():
    assert "sec_498a" in tokenize("He was charged under Section 498-A IPC for cruelty.")


def test_other_legal_reference_forms():
    assert protect_legal_refs("u/s 34 and s. 302").split() == ["sec_34", "and", "sec_302"]
    assert "sec_304b" in protect_legal_refs("convicted under 304-b of the i.p.c.")
    assert "art_21" in protect_legal_refs("violates article 21 of the constitution")
    assert protect_legal_refs("section 34 i.e. common intention").split()[0] == "sec_34"


def test_mask_tags_stop_words_and_citations_removed():
    tokens = tokenize("<NAME>, J. The learned counsel for the appellant relied on <CITATION> AIR 1974 SC 753.")
    assert tokens == ["relied"]


def test_bigrams():
    tokens = add_bigrams(tokenize("dowry death of the bride"))
    assert tokens == ["dowry", "death", "bride", "dowry__death", "death__bride"]


def test_stemmer_merges_inflections():
    assert stem("appeals") == stem("appealed") == stem("appealing") == "appeal"
    assert stem("committed") == "commit"
    assert stem("offences") == "offence"
    assert stem("sec_302") == "sec_302"
    assert stem("pass") == "pass"
