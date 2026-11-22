"""Stop words (Lecture: Vocabulary - stop words).

ENGLISH: a standard list of English function words, written out by hand.
LEGAL_BOILERPLATE: words that appear in almost every Indian judgment and say nothing about the facts.
They were picked by hand from the highest-document-frequency terms of the PCR *train* candidates
(`python scripts/top_df_terms.py` prints that list), keeping only procedural/boilerplate words.
"""
ENGLISH = set("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can could did do does doing done down during each either else ever every few for from further had has
have having he her here hers herself him himself his how however i if in into is it its itself just may me might
more most must my myself neither no nor not now of off on once only or other ought our ours ourselves out over own
per same shall she should so some such than that the their theirs them themselves then there thereby therefore
therein thereof thereon these they this those through thus to too under until up upon us very was we were what
when where whereas whether which while who whom whose why will with within without would yet you your yours
yourself yourselves also said viz etc ie eg
""".split())

LEGAL_BOILERPLATE = set("""
hon ble learned counsel appellant appellants respondent respondents petitioner petitioners court courts judgment
judgments order orders case cases para paragraph honourable lordship lordships sri shri smt mr mrs ms dr
""".split())

STOP_WORDS = ENGLISH | LEGAL_BOILERPLATE
