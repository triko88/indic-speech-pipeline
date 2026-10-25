import pytest

from indic_s2t.gold import GOLD_OVERRIDES_FILE, corrected_rows, load_gold_overrides

OVERRIDES_TOML = """
[qa_hi_010]
question = "एनडीए की पहली सरकार का नेतृत्व किसने किया?"
answers = ["अटल बिहारी वाजपेयी"]
answerable = true
evidence = "अटल बिहारी वाजपेयी के नेतृत्व में राष्ट्रीय जनतांत्रिक गठबंधन"
"""


def llm_row(condition, item_id="qa_hi_010", ref_question="एनडीए की पहली सरकार का नेतृत्व किसने किया?",
            answers=("मनमोहन सिंह",), answerable=True):
    return {"id": item_id, "condition": condition, "lang": "hi", "ref_question": ref_question,
            "answers": list(answers), "answerable": answerable, "response": "अटल बिहारी वाजपेयी"}


@pytest.fixture
def overrides(tmp_path):
    overrides_path = tmp_path / "gold_overrides.toml"
    overrides_path.write_text(OVERRIDES_TOML, encoding="utf-8")
    return load_gold_overrides(overrides_path)


def test_loader_reads_answers_answerable_and_evidence(overrides):
    override = overrides["qa_hi_010"]
    assert override.answers == ("अटल बिहारी वाजपेयी",) and override.answerable
    assert override.evidence.startswith("अटल")


def test_override_replaces_gold_on_acoustic_oracle_and_syntactic_rows(overrides):
    rows = tuple(llm_row(condition) for condition in ("clean", "oracle", "scramble"))
    assert all(row["answers"] == ["अटल बिहारी वाजपेयी"] for row in corrected_rows(rows, overrides))
    assert rows[0]["answers"] == ["मनमोहन सिंह"]  # input rows left untouched


def test_semantic_rows_stay_unanswerable_and_other_items_unchanged(overrides):
    wrong_passage = llm_row("wrong_passage", answers=(), answerable=False)
    other_item = llm_row("clean", item_id="qa_hi_011", ref_question="कुछ और")
    assert corrected_rows((wrong_passage, other_item), overrides) == (wrong_passage, other_item)


def test_override_for_a_different_question_raises(overrides):
    with pytest.raises(ValueError, match="qa_hi_010"):
        corrected_rows((llm_row("clean", ref_question="दूसरा प्रश्न"),), overrides)


def test_repo_overrides_file_loads():
    assert all(override.evidence for override in load_gold_overrides(GOLD_OVERRIDES_FILE).values())
