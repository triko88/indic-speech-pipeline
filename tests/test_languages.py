import dataclasses
from pathlib import Path

import pytest

from indic_s2t.languages import LANGUAGES, load_languages

FIXTURES = Path(__file__).parent / "fixtures"


def test_toy_file_loads_frozen_languages_with_tuple_fields():
    toy_language = load_languages(FIXTURES / "languages_toy.toml")["xx"]
    assert toy_language.name == "Toy"
    assert toy_language.script_block == (0x41, 0x7A)
    assert toy_language.abstain_phrases == ("no answer", "nope")
    assert toy_language.tts_speakers == ("Ann", "Bob")
    assert not toy_language.indicnlp and toy_language.kathbath is None and toy_language.squad_url is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        toy_language.name = "Other"


def test_unknown_key_is_rejected(tmp_path):
    typo_file = tmp_path / "languages.toml"
    typo_file.write_text('[xx]\nname = "Toy"\nscript_block = [65, 122]\nasr_model = "a"\nkathbat = "toy"\n',
                         encoding="utf-8")
    with pytest.raises(TypeError, match="kathbat"):
        load_languages(typo_file)


def test_repo_file_has_hindi_english_tamil():
    assert tuple(LANGUAGES) == ("hi", "en", "ta")
    assert LANGUAGES["en"].fleurs == "en_us" and LANGUAGES["en"].squad_url
    assert LANGUAGES["ta"].kathbath == "tamil" and LANGUAGES["ta"].indicqa == "ta"
    assert all("no answer" in language.abstain_phrases for language in LANGUAGES.values())
