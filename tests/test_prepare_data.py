import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import prepare_data

TOY_SQUAD_PATH = Path(__file__).parent / "fixtures" / "s2_toy_squad.json"


def load_toy_questions():
    return prepare_data.parse_squad(json.loads(TOY_SQUAD_PATH.read_text(encoding="utf-8")))


def make_question(passage_id, question_index, answerable=True):
    return prepare_data.QAQuestion(
        src_id=f"{passage_id}.{question_index}", passage_id=passage_id,
        question=f"प्रश्न संख्या {passage_id} {question_index}", context="संदर्भ",
        answers=("उत्तर",) if answerable else (), answerable=answerable)


def test_parse_squad_skips_and_counts_malformed_entries():
    toy_questions, malformed_count = load_toy_questions()
    assert malformed_count == 2
    assert len(toy_questions) == 12
    assert len({question.src_id for question in toy_questions}) == 12


def test_parse_squad_marks_empty_answer_text_unanswerable():
    toy_questions, _ = load_toy_questions()
    unanswerable_questions = [question for question in toy_questions if not question.answerable]
    assert {question.question for question in unanswerable_questions} == {
        "भारत का राष्ट्रीय पशु कौन सा है ?", "இந்தியாவின் தேசிய விலங்கு எது ?"}
    assert all(question.answers == () for question in unanswerable_questions)


def test_each_filter_removes_exactly_the_expected_questions():
    # The toy file mixes Hindi and Tamil: questions in the other script count as off-script.
    toy_questions, _ = load_toy_questions()
    kept_questions, removed_counts = prepare_data.apply_question_filters(toy_questions, "hi")
    assert removed_counts == (("digits_or_off_script", 7), ("word_count", 2), ("passage_length", 0))
    assert {question.question for question in kept_questions} == {
        "भारत की राजधानी क्या है ?",
        "भारत का राष्ट्रीय पशु कौन सा है ?",
        "गंगा नदी कहाँ से निकलती है ?",
    }


def test_english_questions_survive_the_script_filter():
    english_questions = tuple(prepare_data.QAQuestion(src_id=str(index), passage_id=index, question=question_text,
                                                      context="A short passage.", answers=("x",), answerable=True)
                              for index, question_text in enumerate(("Where is the Taj Mahal located?",
                                                                     "What is H2O made of?",
                                                                     "Who built the ताजमहल?")))
    kept_questions, _ = prepare_data.apply_question_filters(english_questions, "en")
    assert [question.question for question in kept_questions] == ["Where is the Taj Mahal located?"]


def test_parse_fleurs_tsv_counts_each_recording_as_its_own_speaker(tmp_path):
    tsv_text = ("1980\t127.wav\tUnfortunately, traffic is hard.\tunfortunately traffic is hard\tu n\t202240\tMALE\n"
                "1980\t905.wav\tUnfortunately, traffic is hard.\tunfortunately traffic is hard\tu n\t126720\tFEMALE\n"
                "1914\t621.wav\tTwo pools: hidden and expressed.\ttwo pools hidden and expressed\tt w\t94080\tFEMALE\n")
    utterances = prepare_data.parse_fleurs_tsv(tsv_text, tmp_path)
    assert [utterance.text for utterance in utterances][2] == "Two pools: hidden and expressed."
    assert [utterance.speaker_id for utterance in utterances] == ["127", "905", "621"]
    assert [utterance.gender for utterance in utterances] == ["male", "female", "female"]
    assert utterances[0].path == tmp_path / "127.wav"


def test_round_robin_uses_every_passage_before_reusing_one():
    pool_questions = tuple(make_question(passage_id, question_index)
                           for passage_id in range(5) for question_index in range(4))
    first_round = prepare_data.sample_round_robin(pool_questions, 5, 0, seed=0)
    assert len({question.passage_id for question in first_round}) == 5
    two_rounds = prepare_data.sample_round_robin(pool_questions, 10, 0, seed=0)
    assert Counter(question.passage_id for question in two_rounds) == {passage_id: 2 for passage_id in range(5)}


def test_round_robin_honours_unanswerable_quota_and_is_seeded():
    pool_questions = tuple(make_question(passage_id, question_index, answerable=question_index % 2 == 0)
                           for passage_id in range(10) for question_index in range(4))
    sampled_questions = prepare_data.sample_round_robin(pool_questions, 8, 2, seed=3)
    assert len(sampled_questions) == 10
    assert sum(not question.answerable for question in sampled_questions) == 2
    assert sampled_questions == prepare_data.sample_round_robin(pool_questions, 8, 2, seed=3)
    assert sampled_questions != prepare_data.sample_round_robin(pool_questions, 8, 2, seed=4)


def test_round_robin_raises_when_pool_too_small():
    pool_questions = (make_question(0, 0), make_question(1, 0, answerable=False))
    with pytest.raises(ValueError):
        prepare_data.sample_round_robin(pool_questions, 1, 2, seed=0)


@pytest.mark.parametrize("question_text, duration, expected", [
    ("क" * 30, 2.5, True),    # 12 chars/s
    ("क" * 30, 0.7, False),   # too short
    ("क" * 40, 1.0, False),   # 40 chars/s, too fast
    ("क" * 10, 5.0, False),   # 2 chars/s, too slow
])
def test_tts_qc_rule(question_text, duration, expected):
    assert prepare_data.passes_tts_qc(question_text, duration) is expected


def test_voice_descriptions_rotate_through_four_clean_voices():
    for lang in ("hi", "en"):
        descriptions = [prepare_data.voice_description(lang, index) for index in range(8)]
        assert len(set(descriptions[:4])) == 4
        assert descriptions[4:] == descriptions[:4]
        assert all("very clear audio" in description for description in descriptions)
    assert prepare_data.voice_description("en", 1).startswith("Thoma ")


def test_eta_uses_moving_average_of_recent_items():
    assert prepare_data.eta_seconds((2.0, 4.0, 6.0), 10) == pytest.approx(40.0)
    assert prepare_data.eta_seconds((100.0,) + (1.0,) * 10, 5) == pytest.approx(5.0)
    assert prepare_data.eta_seconds((), 5) is None


def test_question_and_track_rows_have_exactly_the_required_fields():
    sampled_questions = (make_question(0, 0), make_question(1, 0, answerable=False))
    question_rows = prepare_data.question_rows(sampled_questions, "hi")
    assert [row["id"] for row in question_rows] == ["qa_hi_000", "qa_hi_001"]
    assert all(set(row) == set(prepare_data.QA_QUESTION_FIELDS) for row in question_rows)
    assert question_rows[1]["answers"] == [] and question_rows[1]["answerable"] is False
    track_row = prepare_data.qa_track_row(question_rows[0], "data/qa/qa_hi_000.wav", 2.5, "Divya speaks", 1)
    assert set(track_row) == set(prepare_data.QA_TRACK_FIELDS)
    assert json.loads(json.dumps(track_row, ensure_ascii=False)) == track_row


def test_manifest_validation_reports_missing_fields():
    track_row = prepare_data.qa_track_row(
        prepare_data.question_rows((make_question(0, 0),), "en")[0], "x.wav", 2.0, "Mary speaks", 1)
    prepare_data.validate_manifest_row(track_row)
    incomplete_row = {field: value for field, value in track_row.items() if field != "audio"}
    with pytest.raises(ValueError, match="audio"):
        prepare_data.validate_manifest_row(incomplete_row)


def test_import_manifest_normalises_audio_to_16k(tmp_path):
    import numpy as np
    import soundfile
    soundfile.write(tmp_path / "recorded.wav", np.zeros((44100, 2), dtype=np.float32), 44100)
    question_row = prepare_data.question_rows((make_question(0, 0),), "hi")[0]
    manifest_row = {**question_row, "audio": "recorded.wav", "duration": 1.0, "tts_description": "human"}
    manifest_path = tmp_path / "manifest.jsonl"
    prepare_data.write_jsonl(manifest_path, (manifest_row,))
    out_dir = tmp_path / "out"
    prepare_data.import_manifest(manifest_path, out_dir)
    (track_row,) = prepare_data.read_jsonl(out_dir / "qa_track.jsonl")
    assert set(track_row) == set(prepare_data.QA_TRACK_FIELDS)
    audio_info = soundfile.info(track_row["audio"])
    assert audio_info.samplerate == 16000 and audio_info.channels == 1 and track_row["duration"] == 1.0


def test_asr_row_has_exactly_the_required_fields():
    asr_row = prepare_data.asr_row("hi", 0, "data/asr/hi_000.wav", 3.2, "नमस्ते", "kathbath/test_unknown",
                                   {"speaker_id": "12", "original_id": "844-12-f"})
    assert asr_row["id"] == "hi_000" and asr_row["track"] == "asr" and asr_row["source"] == "kathbath/test_unknown"
    assert set(asr_row) == {"id", "track", "lang", "audio", "duration", "ref_text", "source", "meta"}


def test_kathbath_speaker_id_comes_from_file_name():
    assert prepare_data.kathbath_speaker_id("844424932919645-1202-f.m4a") == "1202"
