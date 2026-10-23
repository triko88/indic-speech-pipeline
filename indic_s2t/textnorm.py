"""Text normalisation shared by references and hypotheses, so both sides of every metric see identical text."""

import unicodedata
from functools import lru_cache

import regex
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

ZERO_WIDTH_CHARACTERS = regex.compile("[​‌‍﻿]")
PUNCTUATION_AND_SYMBOLS = regex.compile(r"[\p{P}\p{S}]+")  # covers danda (।) and double danda (॥)
SCRIPT_BLOCKS = {"hi": (0x0900, 0x097F), "ta": (0x0B80, 0x0BFF)}


@lru_cache(maxsize=None)
def indic_normalizer(lang, strip_nukta):
    return IndicNormalizerFactory().get_normalizer(lang, remove_nuktas=strip_nukta)


def normalize(text, lang, strip_nukta=False):
    """NFC; strip ZWJ/ZWNJ/ZWSP/BOM; indic-nlp normaliser; Indic digits -> ASCII; drop \\p{P}\\p{S}; collapse spaces; lowercase."""
    composed_text = ZERO_WIDTH_CHARACTERS.sub("", unicodedata.normalize("NFC", text))
    indic_normalized = indic_normalizer(lang, strip_nukta).normalize(composed_text)
    ascii_digits = regex.sub(r"\d", lambda digit_match: str(unicodedata.digit(digit_match.group())), indic_normalized)
    return " ".join(PUNCTUATION_AND_SYMBOLS.sub(" ", ascii_digits).split()).lower()


def words(text):
    return tuple(text.split())


def aksharas(text):
    """Grapheme clusters (`\\X`), the closest Unicode notion of an akshara."""
    return tuple(regex.findall(r"\X", text))


def script_letter_share(text, lang):
    """Share of letters/marks that fall in the language's Unicode block; 0.0 when there are none."""
    block_start, block_end = SCRIPT_BLOCKS[lang]
    letters = [character for character in text if unicodedata.category(character)[0] in "LM"]
    if not letters:
        return 0.0
    return sum(block_start <= ord(character) <= block_end for character in letters) / len(letters)
