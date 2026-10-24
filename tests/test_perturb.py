import pytest

from indic_s2t.perturb import TEXT_CONDITIONS, perturb
from indic_s2t.textnorm import normalize, words

PASSAGE = "ताजमहल आगरा शहर में यमुना नदी के किनारे स्थित है। इसे शाहजहाँ ने बनवाया था।"
OTHER_PASSAGE = "कुतुब मीनार दिल्ली में है। इसका निर्माण कुतुबुद्दीन ऐबक ने शुरू करवाया।"
GOLD_LEAK_PASSAGE = "आगरा उत्तर प्रदेश का एक शहर है।"  # holds the gold answer, so it can never be the wrong passage
CONTEXTS = (PASSAGE, OTHER_PASSAGE, GOLD_LEAK_PASSAGE)
ITEM = {"id": "qa_hi_000", "lang": "hi", "ref_text": "ताजमहल किस शहर में स्थित है ?", "context": PASSAGE,
        "answers": ["आगरा"], "answerable": True}
ENGLISH_ITEM = {"id": "qa_en_000", "lang": "en", "ref_text": "Who built the Taj Mahal?",
                "context": "Shah Jahan built the Taj Mahal in Agra.", "answers": ["Shah Jahan"], "answerable": True}


def test_conditions_are_known():
    assert TEXT_CONDITIONS == ("scramble", "drop_head", "drop_tail", "swap_anchor", "wrong_passage")


def test_scramble_reorders_words_keeps_punctuation_last_and_task_unchanged():
    scrambled = perturb(ITEM, "scramble", CONTEXTS)
    assert scrambled.question.endswith("?")
    assert sorted(scrambled.question.removesuffix("?").split()) == sorted("ताजमहल किस शहर में स्थित है".split())
    assert scrambled.question.removesuffix("?").split() != "ताजमहल किस शहर में स्थित है".split()
    assert (scrambled.context, scrambled.answers, scrambled.answerable) == (PASSAGE, ("आगरा",), True)
    assert perturb(ITEM, "scramble", CONTEXTS) == scrambled  # seeded by id and condition


def test_scramble_never_returns_the_original_order():
    short_items = [{**ITEM, "id": f"qa_hi_{index:03d}", "ref_text": "राम कहाँ गया?"} for index in range(30)]
    assert all(perturb(item, "scramble", CONTEXTS).question != "राम कहाँ गया?" for item in short_items)


def test_drop_head_and_tail_clip_one_word():
    assert perturb(ITEM, "drop_head", CONTEXTS).question == "किस शहर में स्थित है?"
    assert perturb(ITEM, "drop_tail", CONTEXTS).question == "ताजमहल किस शहर में स्थित?"
    assert perturb(ENGLISH_ITEM, "drop_head", CONTEXTS).question == "built the Taj Mahal?"
    assert perturb(ITEM, "drop_tail", CONTEXTS).answerable


def test_swap_anchor_replaces_the_longest_passage_word_and_makes_it_unanswerable():
    swapped = perturb(ITEM, "swap_anchor", CONTEXTS)
    original_words, swapped_words = ITEM["ref_text"].removesuffix(" ?").split(), swapped.question.removesuffix("?").split()
    changed = [(old, new) for old, new in zip(original_words, swapped_words) if old != new]
    assert len(swapped_words) == len(original_words) and len(changed) == 1
    old_word, new_word = changed[0]
    assert old_word == "ताजमहल"  # 5 aksharas, the longest question word found in the passage
    # the only other-passage word of at least 5 aksharas: a content word, not a short function word
    assert new_word == "कुतुबुद्दीन" and new_word not in words(normalize(PASSAGE, "hi"))
    assert (swapped.context, swapped.answers, swapped.answerable) == (PASSAGE, (), False)


def test_wrong_passage_skips_own_and_gold_holding_passages():
    mismatched = perturb(ITEM, "wrong_passage", CONTEXTS)
    assert (mismatched.question, mismatched.context) == (ITEM["ref_text"], OTHER_PASSAGE)
    assert (mismatched.answers, mismatched.answerable) == ((), False)
    with pytest.raises(ValueError):
        perturb(ITEM, "wrong_passage", (PASSAGE, GOLD_LEAK_PASSAGE))
