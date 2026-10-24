import pytest

from indic_s2t.scoring import (judge_label, judge_messages, label_response, parse_judge, passage_support, qa_rates,
                               token_f1)

HINDI_PASSAGE = "ताजमहल आगरा शहर में यमुना नदी के किनारे स्थित है। इसे शाहजहाँ ने बनवाया था।"
TAMIL_PASSAGE = "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும். இது வங்காள விரிகுடா கடற்கரையில் அமைந்துள்ளது."


def label(response, gold_answers, passage, answerable, lang):
    return label_response(response, gold_answers, passage, answerable, lang).label


def test_answerable_no_answer_token_is_false_abstain():
    assert label("NO_ANSWER", ("आगरा",), HINDI_PASSAGE, True, "hi") == "FALSE_ABSTAIN"


def test_answerable_abstain_phrase_is_false_abstain():
    assert label("इस अनुच्छेद में उत्तर नहीं है।", ("आगरा",), HINDI_PASSAGE, True, "hi") == "FALSE_ABSTAIN"
    assert label("பதில் இல்லை", ("சென்னை",), TAMIL_PASSAGE, True, "ta") == "FALSE_ABSTAIN"


def test_correct_by_substring():
    assert label("आगरा में।", ("आगरा",), HINDI_PASSAGE, True, "hi") == "CORRECT"


def test_tamil_suffix_case_is_correct():
    assert label("சென்னையில்", ("சென்னை",), TAMIL_PASSAGE, True, "ta") == "CORRECT"


def test_correct_by_token_f1_when_no_substring():
    # gold "यमुना नदी के किनारे" vs response "यमुना किनारे": no substring, F1 = 2*(1*0.5)/(1+0.5) = 0.667
    assert label("यमुना किनारे", ("यमुना नदी के किनारे",), HINDI_PASSAGE, True, "hi") == "CORRECT"


def test_single_akshara_gold_needs_f1_not_substring():
    # gold "क" (1 akshara) is a substring of "कमल" but must not count via substring; F1 is 0
    assert label("कमल", ("क",), "क कमल", True, "hi") == "WRONG_GROUNDED"


def test_wrong_but_grounded_in_passage():
    assert label("शाहजहाँ", ("आगरा",), HINDI_PASSAGE, True, "hi") == "WRONG_GROUNDED"


def test_wrong_and_ungrounded_is_hallucinated():
    assert label("मुंबई महाराष्ट्र", ("आगरा",), HINDI_PASSAGE, True, "hi") == "HALLUCINATED"


def test_unanswerable_abstained_is_correct_abstain():
    assert label("NO_ANSWER", (), HINDI_PASSAGE, False, "hi") == "CORRECT_ABSTAIN"


def test_unanswerable_answered_is_hallucinated():
    assert label("शाहजहाँ", (), HINDI_PASSAGE, False, "hi") == "HALLUCINATED"


def test_verbose_and_off_script_flags():
    long_response = " ".join(["आगरा"] * 26)
    verdict = label_response(long_response, ("आगरा",), HINDI_PASSAGE, True, "hi")
    assert verdict.verbose and not verdict.off_script
    latin_verdict = label_response("Agra", ("आगरा",), HINDI_PASSAGE, True, "hi")
    assert latin_verdict.off_script and not latin_verdict.verbose
    assert not label_response("NO_ANSWER", ("आगरा",), HINDI_PASSAGE, True, "hi").off_script
    # digits only: no letters to judge the script by, so not off-script
    assert not label_response("१९५४-५६", ("१९५४",), HINDI_PASSAGE, True, "hi").off_script


def test_token_f1_and_passage_support():
    assert token_f1("a b", "a b") == 1.0
    assert token_f1("a", "b") == 0.0
    # "चेन्नई" unseen, "आगरा" seen -> 0.5
    assert passage_support("आगरा चेन्नई", HINDI_PASSAGE, "hi") == pytest.approx(0.5)
    # சென்னையை = செ ன் னை யை (4 aksharas) -> ⌈2.8⌉ = 3-akshara prefix சென்னை is in the passage
    assert passage_support("சென்னையை", TAMIL_PASSAGE, "ta") == 1.0
    # சென்னையின் = 5 aksharas -> 4-akshara prefix சென்னையி is not: longer suffixes fall outside the 70% rule
    assert passage_support("சென்னையின்", TAMIL_PASSAGE, "ta") == 0.0


def test_qa_rates():
    labels_and_answerable = (("CORRECT", True), ("CORRECT", True), ("FALSE_ABSTAIN", True), ("HALLUCINATED", True),
                             ("CORRECT_ABSTAIN", False), ("HALLUCINATED", False))
    rates = qa_rates(labels_and_answerable)
    assert rates["n"] == 6
    assert rates["correctness"] == pytest.approx(3 / 6)
    assert rates["answerable_accuracy"] == pytest.approx(2 / 4)
    assert rates["abstention_recall"] == pytest.approx(1 / 2)
    assert rates["false_abstain_rate"] == pytest.approx(1 / 4)
    assert rates["hallucination_rate"] == pytest.approx(2 / 6)
    assert rates["hallucination_rate_non_abstained"] == pytest.approx(2 / 4)
    assert rates["correctness_ci"][0] < 0.5 < rates["correctness_ci"][1]


def test_judge_parse_and_label():
    assert parse_judge('```json\n{"abstained": false, "correct": true, "supported": true}\n```') == {
        "abstained": False, "correct": True, "supported": True}
    assert parse_judge("The answer is correct.") is None
    assert parse_judge('{"abstained": "no", "correct": true, "supported": true}') is None
    assert judge_label(None, True) == "JUDGE_ERROR"
    assert judge_label({"abstained": True, "correct": False, "supported": False}, True) == "FALSE_ABSTAIN"
    assert judge_label({"abstained": False, "correct": True, "supported": True}, True) == "CORRECT"
    assert judge_label({"abstained": False, "correct": False, "supported": True}, True) == "WRONG_GROUNDED"
    assert judge_label({"abstained": False, "correct": False, "supported": False}, True) == "HALLUCINATED"
    assert judge_label({"abstained": True, "correct": True, "supported": True}, False) == "CORRECT_ABSTAIN"
    assert judge_label({"abstained": False, "correct": False, "supported": True}, False) == "HALLUCINATED"


def test_judge_messages_contain_inputs_and_strict_reminder():
    messages = judge_messages(HINDI_PASSAGE, "ताजमहल कहाँ है", ("आगरा",), "आगरा", "hi")
    joined = "\n".join(message["content"] for message in messages)
    assert HINDI_PASSAGE in joined and "ताजमहल कहाँ है" in joined and "आगरा" in joined
    strict_joined = "\n".join(message["content"] for message in judge_messages(
        HINDI_PASSAGE, "q", ("a",), "r", "hi", strict=True))
    assert "ONLY" in strict_joined and len(strict_joined) > len("\n".join(
        message["content"] for message in judge_messages(HINDI_PASSAGE, "q", ("a",), "r", "hi")))
