import csv
import json

import pytest

from scripts import make_report

PASSAGE = "ताजमहल आगरा शहर में यमुना नदी के किनारे स्थित है। इसे शाहजहाँ ने बनवाया था।"
# response text -> rule label, given gold "आगरा" on an answerable question
CORRECT, FALSE_ABSTAIN, WRONG_GROUNDED, HALLUCINATED = "आगरा", "NO_ANSWER", "शाहजहाँ", "मुंबई"


def llm_row(item_id, condition, response, question="ताजमहल कहाँ है", answerable=True, ttft_s=0.2, total_s=0.5):
    return {"id": item_id, "condition": condition, "lang": "hi", "question": question,
            "ref_question": "ताजमहल कहाँ है", "context": PASSAGE, "answers": ["आगरा"] if answerable else [],
            "answerable": answerable, "response": response, "reasoning_text": "", "ttft_s": ttft_s,
            "total_s": total_s, "timings": {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 4, "predicted_ms": 40.0}}


def attribution_rows():
    # oracle correct on q0,q1; clean correct on q0,q2  ->  both 1, ASR-induced 1, ASR-only 1, LM failure 1
    oracle = [CORRECT, CORRECT, HALLUCINATED, HALLUCINATED]
    clean = [CORRECT, HALLUCINATED, CORRECT, HALLUCINATED]
    return tuple(llm_row(f"q{index}", "oracle", response) for index, response in enumerate(oracle)) + tuple(
        llm_row(f"q{index}", "clean", response) for index, response in enumerate(clean))


def test_attribution_counts_delta_and_mcnemar():
    rows = attribution_rows()
    table = make_report.attribution_table(rows, make_report.rule_verdicts(rows))
    assert len(table) == 1
    clean_row = table[0]
    assert (clean_row["lang"], clean_row["condition"]) == ("hi", "clean")
    assert (clean_row["both_correct"], clean_row["asr_induced"], clean_row["lm_failure"],
            clean_row["asr_only_correct"]) == (1, 1, 1, 1)
    assert clean_row["delta_pp"] == pytest.approx(0.0)
    assert clean_row["p_vs_oracle"] == pytest.approx(1.0)
    assert clean_row["p_vs_clean"] is None  # clean is not compared with itself


def test_p_vs_clean_for_degraded_condition():
    rows = attribution_rows() + tuple(llm_row(f"q{index}", "telephony", HALLUCINATED) for index in range(4))
    table = {row["condition"]: row for row in make_report.attribution_table(rows, make_report.rule_verdicts(rows))}
    # clean correct on q0,q2, telephony on none -> discordant (2, 0) -> exact p = 2 * 1/4 = 0.5
    assert table["telephony"]["p_vs_clean"] == pytest.approx(0.5)
    assert table["telephony"]["delta_pp"] == pytest.approx(-50.0)


def linguistic_rows():
    # scramble correct on q0 only (oracle: q0, q1); wrong_passage (unanswerable) abstains on q0, q1, answers q2, q3
    scramble = [CORRECT, HALLUCINATED, HALLUCINATED, HALLUCINATED]
    wrong_passage = [FALSE_ABSTAIN, FALSE_ABSTAIN, HALLUCINATED, HALLUCINATED]
    return attribution_rows() + tuple(
        llm_row(f"q{index}", "scramble", response, question="कहाँ ताजमहल है") for index, response in enumerate(scramble)
    ) + tuple(llm_row(f"q{index}", "wrong_passage", response, answerable=False)
              for index, response in enumerate(wrong_passage))


def test_linguistic_table_compares_text_conditions_with_oracle():
    rows = linguistic_rows()
    verdicts = make_report.rule_verdicts(rows)
    table = {row["condition"]: row for row in make_report.linguistic_table(rows, verdicts)}
    assert set(table) == {"scramble", "wrong_passage"}
    assert table["scramble"]["delta_pp"] == pytest.approx(-25.0)
    assert table["scramble"]["oracle_correctness"] == pytest.approx(0.5)
    assert table["wrong_passage"]["hallucination_rate"] == pytest.approx(0.5)
    assert table["wrong_passage"]["abstention_recall"] == pytest.approx(0.5)


def test_acoustic_only_tables_ignore_text_conditions():
    rows = linguistic_rows()
    verdicts = make_report.rule_verdicts(rows)
    assert [row["condition"] for row in make_report.attribution_table(rows, verdicts)] == ["clean"]
    assert sum(row["n"] for row in make_report.wer_bucket_table(rows, verdicts)) == 4  # the clean rows only
    examples = make_report.qualitative_examples(rows, verdicts)
    assert {row["condition"] for row in examples["ASR_INDUCED"]} == {"clean"}
    assert [row["id"] for row in examples["scramble"]] == ["q1", "q2", "q3"]  # failures under each text condition


def test_length_table_buckets_gold_question_words_for_oracle_and_clean():
    short_question, long_question = "ताजमहल कहाँ है", "ताजमहल किस शहर में और किस नदी के उत्तरी किनारे पर स्थित है"  # 13 words
    rows = tuple({**llm_row(f"q{index}", condition, response), "ref_question": question}
                 for index, (question, response) in enumerate(((short_question, CORRECT), (long_question, HALLUCINATED)))
                 for condition in ("oracle", "clean", "scramble"))
    table = make_report.length_table(rows, make_report.rule_verdicts(rows))
    assert [(row["condition"], row["bucket"], row["n"], row["correctness"]) for row in table] == [
        ("clean", "3-7", 1, 1.0), ("clean", "12+", 1, 0.0), ("oracle", "3-7", 1, 1.0), ("oracle", "12+", 1, 0.0)]


def test_wer_buckets():
    question_words = "एक दो तीन चार पाँच छह सात आठ नौ दस".split()
    one_wrong = " ".join(["गलत", *question_words[1:]])  # 10% WER
    four_wrong = " ".join(["गलत"] * 4 + question_words[4:])  # 40% WER
    rows = (llm_row("q0", "clean", CORRECT, question=" ".join(question_words)),
            llm_row("q1", "clean", CORRECT, question=one_wrong),
            llm_row("q2", "clean", HALLUCINATED, question=four_wrong),
            llm_row("q3", "clean", HALLUCINATED, question="गलत"),
            llm_row("q0", "oracle", CORRECT, question=" ".join(question_words)))
    rows = tuple({**row, "ref_question": " ".join(question_words)} for row in rows)
    table = {row["bucket"]: row for row in make_report.wer_bucket_table(rows, make_report.rule_verdicts(rows))}
    assert [table[bucket]["n"] for bucket in ("0", "(0,20%]", "(20,50%]", ">50%")] == [1, 1, 1, 1]
    assert table["0"]["correctness"] == 1.0 and table[">50%"]["correctness"] == 0.0


def test_asr_table_numbers_come_from_rows():
    asr_rows = ({"id": "hi_000", "condition": "clean", "track": "asr", "lang": "hi", "ref_text": "राम घर गया।",
                 "hyp": "राम घर गया", "asr_s": 0.5, "audio_s": 2.0},
                {"id": "hi_001", "condition": "clean", "track": "asr", "lang": "hi", "ref_text": "सीता वन गई",
                 "hyp": "सीता बन गई", "asr_s": 0.5, "audio_s": 2.0})
    table = make_report.asr_table(asr_rows)
    assert len(table) == 1
    assert table[0]["n"] == 2
    assert table[0]["wer"] == pytest.approx(1 / 6)  # danda normalised away; one substitution in 6 words
    assert table[0]["exact_match"] == pytest.approx(0.5)


def test_latency_e2e_is_asr_plus_llm():
    asr_rows = ({"id": "q0", "condition": "clean", "track": "qa", "lang": "hi", "ref_text": "x", "hyp": "x",
                 "asr_s": 1.0, "audio_s": 4.0},)
    llm_rows = (llm_row("q0", "clean", CORRECT, ttft_s=0.25, total_s=0.75), llm_row("q0", "oracle", CORRECT))
    table = {row["lang"]: row for row in make_report.latency_table(asr_rows, llm_rows)}
    assert table["hi"]["speech_end_to_first_token_p50"] == pytest.approx(1.25)
    assert table["hi"]["e2e_p50"] == pytest.approx(1.75)
    assert table["hi"]["asr_rtf"] == pytest.approx(0.25)
    assert table["hi"]["prompt_tok_s_p50"] == pytest.approx(2000.0)
    assert table["hi"]["decode_tok_s_p50"] == pytest.approx(100.0)


def make_judge_row(row, label):
    return {"id": row["id"], "condition": row["condition"], "judge_label": label, "attempts": 1}


def test_judge_agreement_kappa_and_confusion():
    rows = attribution_rows()
    verdicts = make_report.rule_verdicts(rows)
    judge_rows = tuple(make_judge_row(row, verdicts[(row["id"], row["condition"])].label) for row in rows)
    agreement = make_report.judge_agreement(verdicts, judge_rows)
    assert agreement["kappa_5way"] == pytest.approx(1.0)
    assert agreement["kappa_correct"] == pytest.approx(1.0)
    assert agreement["judge_errors"] == 0
    error_rows = judge_rows[:-1] + (make_judge_row(rows[-1], "JUDGE_ERROR"),)
    assert make_report.judge_agreement(verdicts, error_rows)["judge_errors"] == 1


def many_rows():
    responses = (CORRECT, FALSE_ABSTAIN, WRONG_GROUNDED, HALLUCINATED)
    answerable_rows = tuple(llm_row(f"q{index}", "clean", responses[index % 4]) for index in range(40))
    unanswerable_rows = tuple(llm_row(f"u{index}", "clean", (FALSE_ABSTAIN, HALLUCINATED)[index % 2],
                                      answerable=False) for index in range(20))
    return answerable_rows + unanswerable_rows


def test_audit_sheet_is_40_rows_stratified_blind_and_kappa_on_fill(tmp_path):
    rows = many_rows()
    verdicts = make_report.rule_verdicts(rows)
    sheet_path = tmp_path / "audit_sheet.csv"
    make_report.write_audit_sheet(rows, verdicts, sheet_path, size=40, seed=0)
    with sheet_path.open(encoding="utf-8") as sheet_file:
        sheet_rows = list(csv.DictReader(sheet_file))
    assert len(sheet_rows) == 40
    assert all(sheet_row["human_label"] == "" for sheet_row in sheet_rows)
    assert "rule_label" not in sheet_rows[0] and "judge_label" not in sheet_rows[0]  # blind audit
    sampled_labels = [verdicts[(sheet_row["id"], sheet_row["condition"])].label for sheet_row in sheet_rows]
    assert all(sampled_labels.count(label) == 8 for label in set(sampled_labels)) and len(set(sampled_labels)) == 5

    for sheet_row, rule_label in zip(sheet_rows, sampled_labels):
        sheet_row["human_label"] = rule_label
    with sheet_path.open("w", encoding="utf-8", newline="") as sheet_file:
        writer = csv.DictWriter(sheet_file, fieldnames=list(sheet_rows[0]))
        writer.writeheader()
        writer.writerows(sheet_rows)
    judge_rows = tuple(make_judge_row(row, verdicts[(row["id"], row["condition"])].label) for row in rows)
    audit = make_report.audit_agreement(sheet_path, verdicts, judge_rows)
    assert audit["n"] == 40
    assert audit["kappa_human_rule"] == pytest.approx(1.0)
    assert audit["kappa_human_judge"] == pytest.approx(1.0)


def test_report_builds_every_section_from_a_run_dir(tmp_path):
    run_dir = tmp_path / "main"
    run_dir.mkdir()
    rows = many_rows() + tuple({**row, "condition": "oracle"} for row in many_rows())
    asr_rows = tuple({"id": row["id"], "condition": "clean", "track": "qa", "lang": "hi", "ref_text": row["ref_question"],
                      "hyp": row["question"], "asr_s": 0.4, "audio_s": 3.0} for row in many_rows())
    for name, stage_rows in (("asr", asr_rows), ("llm", rows)):
        (run_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stage_rows), encoding="utf-8")
    report_text = make_report.build_report(run_dir, selection_dir=tmp_path / "missing", variance_dirs=())
    for heading in ("## Setup", "## Metric definitions", "## LLM selection", "## ASR", "## Spoken QA",
                    "## Error attribution", "## Correctness by question WER", "## Linguistic robustness",
                    "## Correctness by question length", "## Latency", "## Variance",
                    "## Judge agreement", "## Human audit", "## Qualitative examples"):
        assert heading in report_text
    assert "limited validity" in report_text


def test_latency_flags_throttled_requests_and_gpu_state():
    fast_timings = {"prompt_n": 500, "prompt_ms": 500.0, "predicted_n": 5, "predicted_ms": 70.0}
    slow_timings = {"prompt_n": 500, "prompt_ms": 2400.0, "predicted_n": 5, "predicted_ms": 1000.0}
    llm_rows = tuple({**llm_row(f"q{index}", "oracle", CORRECT), "timings": fast_timings,
                      "gpu": {"edge_c": 70.0, "junction_c": 80.0 + index, "sclk_mhz": 2600.0, "power_w": 60.0}}
                     for index in range(8)) + tuple({**llm_row(f"s{index}", "oracle", CORRECT), "timings": slow_timings}
                                                    for index in range(2))
    table = {row["lang"]: row for row in make_report.latency_table((), llm_rows)}
    assert table["hi"]["throttled_requests"] == 2  # server prompt speed < 50% of the run's p90 (unthrottled) speed
    assert table["hi"]["gpu_sclk_mhz_p50"] == pytest.approx(2600.0)
    assert table["hi"]["gpu_junction_c_max"] == pytest.approx(87.0)
    assert table["hi"]["ttft_p50_unthrottled"] == pytest.approx(0.2)

    # most requests throttled: the median is itself slow, so the reference must be the unthrottled p90
    mostly_slow = llm_rows[:2] + tuple({**llm_row(f"m{index}", "oracle", CORRECT), "timings": slow_timings}
                                       for index in range(8))
    assert make_report.latency_table((), mostly_slow)[0]["throttled_requests"] == 8


def test_report_shows_latency_for_extra_runs(tmp_path):
    for run_name in ("main", "rerun"):
        (tmp_path / run_name).mkdir()
        (tmp_path / run_name / "llm.jsonl").write_text(
            json.dumps(llm_row("q0", "oracle", CORRECT), ensure_ascii=False) + "\n", encoding="utf-8")
    report_text = make_report.build_report(tmp_path / "main", selection_dir=tmp_path / "missing", variance_dirs=(),
                                           latency_runs=(tmp_path / "rerun",))
    assert "### Latency: main" in report_text and "### Latency: rerun" in report_text


def test_stale_llm_selection_is_reported_not_fatal(tmp_path, monkeypatch):
    from tests.test_select_llm import write_stale_selection
    selection_dir, questions_path = write_stale_selection(tmp_path)
    monkeypatch.setattr(make_report, "QUESTIONS_PATH", questions_path)
    section_text = make_report.selection_section(selection_dir)
    assert "stale" in section_text and "qa_ta_000" in section_text and "make select" in section_text
