from indic_s2t.textnorm import aksharas, normalize, script_letter_share, words

ZWJ, ZWNJ, ZWSP, BOM = "‍", "‌", "​", "﻿"


def test_danda_and_double_danda_are_stripped():
    assert normalize("राम घर गया। वह खुश है॥", "hi") == "राम घर गया वह खुश है"


def test_zero_width_characters_are_stripped():
    assert normalize(f"क्{ZWJ}या{ZWSP} बात{BOM}", "hi") == "क्या बात"
    assert normalize(f"Ta{ZWJ}j{ZWNJ} Mahal{BOM}", "en") == "taj mahal"


def test_nukta_kept_by_default_and_stripped_on_flag():
    precomposed_qa = "क़"  # क़ as a single code point; NFC decomposes it to क + ़
    assert normalize(f"{precomposed_qa}िला", "hi") == "क़िला"
    assert normalize(f"{precomposed_qa}िला", "hi", strip_nukta=True) == "किला"


def test_nasal_spellings_are_unified_for_every_language():
    assert normalize("हिंदू", "hi") == normalize("हिन्दू", "hi")
    assert normalize("संबंध", "hi") == normalize("सम्बन्ध", "hi")
    assert normalize("संसार", "hi") == "संसार"  # anusvara before a non-stop consonant stays
    assert normalize("शंकरः", "sa") == normalize("शङ्करः", "sa")
    assert normalize("bombay", "en") == "bombay"  # no indic-nlp normaliser for English


def test_indic_digits_become_ascii():
    assert normalize("१२३ रुपये", "hi") == "123 रुपये"
    assert normalize("१२ रूप्यकाणि", "sa") == "12 रूप्यकाणि"


def test_english_is_normalised_without_indic_nlp():
    assert normalize("The Taj, in Agra!", "en") == "the taj in agra"


def test_punctuation_symbols_whitespace_and_case():
    assert normalize("  Hello,   WORLD! ₹5 — ok?  ", "hi") == "hello world 5 ok"


def test_words_and_aksharas():
    assert words("राम  घर") == ("राम", "घर")
    assert aksharas("नमस्ते") == ("न", "म", "स्ते")
    assert "".join(aksharas("சென்னையில்")) == "சென்னையில்"


def test_script_letter_share():
    assert script_letter_share("दिल्ली", "hi") == 1.0
    assert script_letter_share("Delhi", "hi") == 0.0
    assert script_letter_share("Delhi", "en") == 1.0
    assert script_letter_share("दिल्ली", "en") == 0.0
    assert script_letter_share("संस्कृतम्", "sa") == 1.0
    assert script_letter_share("123 ...", "sa") == 0.0
