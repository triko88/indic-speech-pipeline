from indic_s2t.textnorm import aksharas, normalize, script_letter_share, words

ZWJ, ZWNJ, ZWSP, BOM = "‍", "‌", "​", "﻿"


def test_danda_and_double_danda_are_stripped():
    assert normalize("राम घर गया। वह खुश है॥", "hi") == "राम घर गया वह खुश है"


def test_zero_width_characters_are_stripped():
    assert normalize(f"क्{ZWJ}या{ZWSP} बात{BOM}", "hi") == "क्या बात"
    assert normalize(f"சென்னை{ZWNJ}யில்", "ta") == "சென்னையில்"


def test_nukta_kept_by_default_and_stripped_on_flag():
    precomposed_qa = "क़"  # क़ as a single code point; NFC decomposes it to क + ़
    assert normalize(f"{precomposed_qa}िला", "hi") == "क़िला"
    assert normalize(f"{precomposed_qa}िला", "hi", strip_nukta=True) == "किला"


def test_indic_digits_become_ascii():
    assert normalize("१२३ रुपये", "hi") == "123 रुपये"
    assert normalize("௧௨ ரூபாய்", "ta") == "12 ரூபாய்"


def test_punctuation_symbols_whitespace_and_case():
    assert normalize("  Hello,   WORLD! ₹5 — ok?  ", "hi") == "hello world 5 ok"


def test_words_and_aksharas():
    assert words("राम  घर") == ("राम", "घर")
    assert aksharas("नमस्ते") == ("न", "म", "स्ते")
    assert "".join(aksharas("சென்னையில்")) == "சென்னையில்"


def test_script_letter_share():
    assert script_letter_share("दिल्ली", "hi") == 1.0
    assert script_letter_share("Delhi", "hi") == 0.0
    assert script_letter_share("சென்னை", "hi") == 0.0
    assert script_letter_share("சென்னை", "ta") == 1.0
    assert script_letter_share("123 ...", "ta") == 0.0
