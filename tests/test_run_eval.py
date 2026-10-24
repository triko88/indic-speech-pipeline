import json

import pytest

from scripts import run_eval

ITEMS = (
    {"id": "hi_000", "track": "asr", "lang": "hi", "ref_text": "a"},
    {"id": "hi_001", "track": "asr", "lang": "hi", "ref_text": "b"},
    {"id": "en_000", "track": "asr", "lang": "en", "ref_text": "c"},
    {"id": "qa_hi_000", "track": "qa", "lang": "hi", "ref_text": "प्रश्न एक"},
    {"id": "qa_hi_001", "track": "qa", "lang": "hi", "ref_text": "प्रश्न दो"},
    {"id": "qa_en_000", "track": "qa", "lang": "en", "ref_text": "question"},
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_select_items_limits_per_track_and_language_in_order():
    selected_ids = [item["id"] for item in run_eval.select_items(ITEMS, limit=1)]
    assert selected_ids == ["hi_000", "en_000", "qa_hi_000", "qa_en_000"]
    assert run_eval.select_items(ITEMS, limit=None) == ITEMS


def test_work_keys_per_stage():
    conditions = ("clean", "telephony")
    asr_keys = run_eval.work_keys("asr", ITEMS, conditions)
    assert len(asr_keys) == len(ITEMS) * 2 and ("hi_000", "telephony") in asr_keys
    llm_keys = run_eval.work_keys("llm", ITEMS, conditions)
    assert set(llm_keys) == {(item_id, condition) for item_id in ("qa_hi_000", "qa_hi_001", "qa_en_000")
                             for condition in ("clean", "telephony", "oracle")}
    assert run_eval.work_keys("judge", ITEMS, conditions) == llm_keys


def test_llm_work_keys_add_text_conditions_for_qa_items_only():
    llm_keys = run_eval.work_keys("llm", ITEMS, ("clean",), ("scramble", "wrong_passage"))
    assert ("qa_hi_000", "scramble") in llm_keys and ("qa_en_000", "wrong_passage") in llm_keys
    assert len(llm_keys) == 3 * 4  # clean, oracle, scramble, wrong_passage
    assert len(run_eval.work_keys("asr", ITEMS, ("clean",), ("scramble",))) == len(ITEMS)  # never sent to ASR


def test_qa_input_for_acoustic_oracle_and_text_conditions():
    item = {**ITEMS[3], "context": "पहला अनुच्छेद", "answers": ["एक"], "answerable": True}
    language_contexts = ("पहला अनुच्छेद", "दूसरा अनुच्छेद")
    asr_hypotheses = {("qa_hi_000", "clean"): "प्रश्न एक गलत"}
    heard = run_eval.qa_input_for(item, "clean", asr_hypotheses, language_contexts)
    assert (heard.question, heard.context, heard.answers, heard.answerable) == (
        "प्रश्न एक गलत", "पहला अनुच्छेद", ("एक",), True)
    assert run_eval.qa_input_for(item, "oracle", asr_hypotheses, language_contexts).question == "प्रश्न एक"
    mismatched = run_eval.qa_input_for(item, "wrong_passage", asr_hypotheses, language_contexts)
    assert (mismatched.context, mismatched.answerable) == ("दूसरा अनुच्छेद", False)


def test_text_conditions_default_to_all_and_can_be_turned_off():
    assert run_eval.parse_args([]).text_conditions == list(run_eval.TEXT_CONDITIONS)
    assert run_eval.parse_args(["--text_conditions"]).text_conditions == []


def test_run_pending_resumes_and_excludes_warmup(tmp_path):
    out_path = tmp_path / "asr.jsonl"
    write_jsonl(out_path, [{"id": "hi_000", "condition": "clean", "hyp": "done earlier"}])
    keys = (("hi_000", "clean"), ("hi_001", "clean"), ("en_000", "clean"))
    processed_keys = []

    def process(key):
        processed_keys.append(key)
        return {"id": key[0], "condition": key[1], "hyp": "new"}

    written_count = run_eval.run_pending(keys, process, out_path, warmup=2)
    assert written_count == 2
    # two warm-up calls on the first pending key, then each pending key once; warm-ups are never written
    assert processed_keys == [("hi_001", "clean")] * 2 + [("hi_001", "clean"), ("en_000", "clean")]
    rows = run_eval.read_jsonl(out_path)
    assert [(row["id"], row["condition"]) for row in rows] == list(keys)

    processed_keys.clear()
    assert run_eval.run_pending(keys, process, out_path, warmup=2) == 0
    assert processed_keys == []  # nothing pending: no warm-up either


def test_force_truncates_stage_output(tmp_path):
    out_path = tmp_path / "llm.jsonl"
    write_jsonl(out_path, [{"id": "x", "condition": "clean"}])
    run_eval.prepare_output(out_path, force=False)
    assert run_eval.done_keys(out_path) == {("x", "clean")}
    run_eval.prepare_output(out_path, force=True)
    assert run_eval.done_keys(out_path) == set()


def test_question_for_oracle_uses_gold_text_else_asr_hypothesis():
    item = ITEMS[3]
    asr_hypotheses = {("qa_hi_000", "clean"): "प्रश्न एक गलत"}
    assert run_eval.question_for(item, "oracle", asr_hypotheses) == "प्रश्न एक"
    assert run_eval.question_for(item, "clean", asr_hypotheses) == "प्रश्न एक गलत"
    with pytest.raises(KeyError):
        run_eval.question_for(item, "telephony", asr_hypotheses)


def test_judge_retries_once_with_strict_reminder():
    valid_output = '{"abstained": false, "correct": true, "supported": true}'
    strict_flags = []

    def call_judge(strict):
        strict_flags.append(strict)
        return "Sure! The answer looks right." if not strict else valid_output

    judgement = run_eval.judge_with_retry(call_judge, answerable=True)
    assert strict_flags == [False, True]
    assert judgement["judge_label"] == "CORRECT" and judgement["attempts"] == 2

    always_malformed = run_eval.judge_with_retry(lambda strict: "no json here", answerable=True)
    assert always_malformed["judge_label"] == "JUDGE_ERROR" and always_malformed["attempts"] == 2
    first_try = run_eval.judge_with_retry(lambda strict: valid_output, answerable=False)
    assert first_try["judge_label"] == "HALLUCINATED" and first_try["attempts"] == 1


def test_each_stage_runs_in_its_own_child_process():
    launched_commands = []
    run_eval.run_stages(("asr", "llm", "judge"), ["--limit", "5"],
                        runner=lambda command, check: launched_commands.append(command))
    assert [command[command.index("--_stage") + 1] for command in launched_commands] == ["asr", "llm", "judge"]
    assert all(command[command.index("--limit") + 1] == "5" for command in launched_commands)


def test_pending_keys_lets_stages_skip_model_loading(tmp_path):
    out_path = tmp_path / "asr.jsonl"
    write_jsonl(out_path, [{"id": "hi_000", "condition": "clean"}])
    assert run_eval.pending_keys((("hi_000", "clean"), ("hi_001", "clean")), out_path) == (("hi_001", "clean"),)
    assert run_eval.pending_keys((("hi_000", "clean"),), out_path) == ()
