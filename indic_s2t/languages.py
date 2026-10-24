"""Per-language settings from languages.toml: every language-specific value the pipeline uses lives there."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

LANGUAGES_FILE = Path(__file__).resolve().parent.parent / "languages.toml"


@dataclass(frozen=True)
class Language:
    name: str
    script_block: tuple[int, int]
    asr_model: str
    indicnlp: bool = False
    kathbath: str | None = None
    fleurs: str | None = None
    indicqa: str | None = None
    squad_url: str | None = None
    abstain_phrases: tuple[str, ...] = ()
    tts_speakers: tuple[str, ...] = ()


def load_languages(path=LANGUAGES_FILE):
    """Language code -> Language, in file order. An unknown key raises TypeError naming it."""
    with open(path, "rb") as languages_file:
        tables = tomllib.load(languages_file)
    return MappingProxyType({code: Language(**{key: tuple(value) if isinstance(value, list) else value
                                               for key, value in table.items()})
                             for code, table in tables.items()})


LANGUAGES = load_languages()
