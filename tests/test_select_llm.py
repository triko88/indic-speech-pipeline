import pytest

from scripts import select_llm


def candidate(name, family, pilot_accuracy, median_latency_s):
    return {"name": name, "family": family, "pilot": {"accuracy": pilot_accuracy, "median_latency_s": median_latency_s}}


def test_recommend_picks_best_accuracy_and_judge_from_other_family():
    results = (candidate("A", "gemma", 0.70, 1.0), candidate("B", "qwen", 0.80, 2.0), candidate("C", "sarvam", 0.60, 5.0))
    assert select_llm.recommend(results) == ("B", "A")


def test_recommend_tie_breaks_on_lower_latency():
    results = (candidate("A", "gemma", 0.75, 1.5), candidate("B", "qwen", 0.75, 0.9), candidate("C", "sarvam", 0.5, 5.0))
    assert select_llm.recommend(results)[0] == "B"


def test_recommend_never_picks_c_as_answerer():
    results = (candidate("A", "gemma", 0.40, 1.0), candidate("B", "qwen", 0.45, 1.0), candidate("C", "sarvam", 0.95, 5.0))
    assert select_llm.recommend(results) == ("B", "C")


def test_recommend_judge_skips_same_family():
    results = (candidate("A", "qwen", 0.90, 1.0), candidate("B", "qwen", 0.85, 1.0), candidate("C", "sarvam", 0.50, 5.0))
    assert select_llm.recommend(results) == ("A", "C")


def test_recommend_without_other_family_has_no_judge():
    assert select_llm.recommend((candidate("A", "gemma", 0.5, 1.0),)) == ("A", None)


def test_fertility_is_total_tokens_over_total_chars():
    assert select_llm.fertility(token_counts=(10, 30), char_counts=(20, 60)) == pytest.approx(0.5)


def test_pilot_summary_accuracy_off_script_and_leaks():
    pilot_items = (
        {"lang": "hi", "label": "CORRECT", "off_script": False, "reasoning_text": "", "total_s": 1.0, "ttft_s": 0.2,
         "timings": {"prompt_per_second": 400.0, "predicted_per_second": 30.0}},
        {"lang": "hi", "label": "CORRECT_ABSTAIN", "off_script": True, "reasoning_text": "hmm", "total_s": 3.0, "ttft_s": 0.4,
         "timings": {"prompt_per_second": 600.0, "predicted_per_second": 40.0}},
        {"lang": "ta", "label": "HALLUCINATED", "off_script": False, "reasoning_text": "", "total_s": 2.0, "ttft_s": 0.3,
         "timings": {"prompt_per_second": 500.0, "predicted_per_second": 35.0}},
        {"lang": "ta", "label": "FALSE_ABSTAIN", "off_script": False, "reasoning_text": "", "total_s": 4.0, "ttft_s": 0.5,
         "timings": {"prompt_per_second": 500.0, "predicted_per_second": 35.0}},
    )
    summary = select_llm.pilot_summary(pilot_items)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["accuracy_by_lang"] == {"hi": 1.0, "ta": 0.0}
    assert summary["off_script_rate"] == pytest.approx(0.25)
    assert summary["reasoning_leak_count"] == 1
    assert summary["median_latency_s"] == pytest.approx(2.5)
    assert summary["median_prompt_tok_s"] == pytest.approx(500.0)
    assert summary["median_decode_tok_s"] == pytest.approx(35.0)


def test_pilot_items_takes_n_per_language_in_file_order():
    questions = tuple({"id": f"{lang}{index}", "lang": lang} for index in range(15) for lang in ("hi", "ta"))
    pilot_questions = select_llm.pilot_questions(questions, per_lang=10)
    assert [question["id"] for question in pilot_questions[:2]] == ["hi0", "ta0"]
    assert sum(question["lang"] == "hi" for question in pilot_questions) == 10
    assert sum(question["lang"] == "ta" for question in pilot_questions) == 10


def test_fertility_passages_are_unique_contexts_per_language():
    questions = ({"lang": "hi", "context": "क"}, {"lang": "hi", "context": "क"}, {"lang": "hi", "context": "ख"},
                 {"lang": "ta", "context": "த"})
    assert select_llm.fertility_passages(questions, "hi", limit=20) == ("क", "ख")


def test_summary_markdown_has_tables_and_recommendation():
    results = (candidate("A", "gemma", 0.70, 1.0) | {"fertility": {"hi": 0.4, "ta": 0.6}},
               candidate("B", "qwen", 0.80, 2.0) | {"fertility": {"hi": 0.5, "ta": 0.9}})
    markdown = select_llm.summary_markdown(results)
    assert "| A |" in markdown and "| B |" in markdown
    assert "Answerer: **B**" in markdown and "Judge: **A**" in markdown
