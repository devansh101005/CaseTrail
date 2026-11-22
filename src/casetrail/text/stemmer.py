"""Light suffix-stripping stemmer (Lecture: Vocabulary - stemming).

A small, explainable stemmer in the spirit of Harman's "S-stemmer" (1991), extended with -ing/-ed/-ly.
It is deliberately conservative: it only merges common inflections such as
"appeals/appealed/appealing" -> "appeal". Porter's full algorithm is not needed for BM25 on legal text,
and whether stemming helps at all is decided on dev.
"""
VOWELS = set("aeiou")


def has_vowel(stem: str) -> bool:
    return any(ch in VOWELS for ch in stem)


def stem(word: str) -> str:
    # Keep short words, numbers and protected legal tokens (sec_302) untouched
    if len(word) <= 4 or not word.isalpha():
        return word
    # Plurals (Harman S-stemmer rules)
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    # Verb forms and adverbs, only if a vowel remains in the stem
    for suffix in ["ingly", "edly", "ing", "ed", "ly"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and has_vowel(word[:-len(suffix)]):
            return undouble(word[:-len(suffix)])
    return word


def undouble(stem_part: str) -> str:
    """'committ' -> 'commit', 'admitt' -> 'admit' (but keep 'll', 'ss', 'zz' as in 'call', 'pass')."""
    if len(stem_part) >= 3 and stem_part[-1] == stem_part[-2] and stem_part[-1] not in "lsz":
        return stem_part[:-1]
    return stem_part
