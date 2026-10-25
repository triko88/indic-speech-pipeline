"""Reviewed gold-label corrections from gold_overrides.toml, applied to LLM rows at scoring time.

Responses never depend on the gold, so a correction re-scores an existing run without rerunning the LLM."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from indic_s2t.perturb import SEMANTIC_CONDITIONS

GOLD_OVERRIDES_FILE = Path(__file__).resolve().parent.parent / "gold_overrides.toml"


@dataclass(frozen=True)
class GoldOverride:
    question: str  # guard: the item's gold question, so a re-sampled id never gets another question's gold
    answers: tuple[str, ...]
    answerable: bool
    evidence: str  # the passage text that justifies the correction


def load_gold_overrides(path=GOLD_OVERRIDES_FILE):
    """Item id -> GoldOverride. An unknown or missing key raises TypeError naming it."""
    with open(path, "rb") as overrides_file:
        tables = tomllib.load(overrides_file)
    return MappingProxyType({item_id: GoldOverride(**{key: tuple(value) if isinstance(value, list) else value
                                                      for key, value in table.items()})
                             for item_id, table in tables.items()})


def corrected_rows(llm_rows, overrides):
    """LLM rows with the reviewed gold. Semantic conditions keep their unanswerable-by-construction gold."""
    def corrected(row):
        override = overrides.get(row["id"])
        if override is None:
            return row
        if row["ref_question"] != override.question:
            raise ValueError(f"gold override {row['id']} is for «{override.question}», but this run asked "
                             f"«{row['ref_question']}»: the QA sample changed, update gold_overrides.toml")
        if row["condition"] in SEMANTIC_CONDITIONS:
            return row
        return {**row, "answers": list(override.answers), "answerable": override.answerable}

    return tuple(corrected(row) for row in llm_rows)
