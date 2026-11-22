"""Protect legal references as single tokens before tokenising (Lecture: Vocabulary - tokenisation, special tokens).

"Section 498-A IPC" -> "sec_498a", "s. 302" -> "sec_302", "302 I.P.C." -> "sec_302", "Article 21" -> "art_21".
Works on lower-cased text. The act is not resolved ("section 302" could be IPC or CrPC); the token keeps the number only.
"""
import re

# A section number with an optional letter attached directly or by a hyphen: 302, 498a, 498-a, 304-b
NUMBER = r"(\d{1,4})(?:-?([a-z]))?\b"
# section 302 / sections 302 / sec. 498-a / s. 10(2) / u/s 34
SECTION = re.compile(r"\b(?:sections?|secs?\.?|u/s\.?|s\.)\s*" + NUMBER)
# 302 ipc / 302 i.p.c. / 304-b of the ipc
SECTION_BEFORE_IPC = re.compile(r"\b" + NUMBER + r"\s+(?:of\s+the\s+)?(?:ipc\b|i\.\s?p\.\s?c\.?)")
# article 21 / art. 14 / articles 32
ARTICLE = re.compile(r"\b(?:articles?|art\.)\s*" + NUMBER)


def section_token(match) -> str:
    return f" sec_{match.group(1)}{match.group(2) or ''} "


def article_token(match) -> str:
    return f" art_{match.group(1)}{match.group(2) or ''} "


def protect_legal_refs(text: str) -> str:
    text = SECTION_BEFORE_IPC.sub(section_token, text)
    text = SECTION.sub(section_token, text)
    return ARTICLE.sub(article_token, text)
