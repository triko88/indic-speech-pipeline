"""Linguistic conditions: deterministic perturbations of the gold question, answered and scored like `oracle`.

Syntactic / structural: scramble (word order), drop_head / drop_tail (first or last word clipped, as by a late or
early endpointer; in SOV languages the tail holds the verb and question particle). Semantic: swap_anchor (the
question's longest passage-anchored word replaced by a word from another passage, a false premise) and wrong_passage
(a passage without the gold answer). Both semantic conditions make the item unanswerable: any answer is a hallucination.
"""

import random
from dataclasses import dataclass, replace

import regex

from indic_s2t.textnorm import aksharas, normalize, words

TEXT_CONDITIONS = ("scramble", "drop_head", "drop_tail", "swap_anchor", "wrong_passage")
SEMANTIC_CONDITIONS = ("swap_anchor", "wrong_passage")  # unanswerable by construction
TRAILING_PUNCTUATION = regex.compile(r"\s*[\p{P}\p{S}]+$")
LETTERS_ONLY = regex.compile(r"[\p{L}\p{M}]+")
MIN_REPLACEMENT_AKSHARAS = 3


@dataclass(frozen=True)
class QaInput:
    question: str
    context: str
    answers: tuple[str, ...]
    answerable: bool


def split_trailing_punctuation(question):
    """"ताजमहल कहाँ है ?" -> (["ताजमहल", "कहाँ", "है"], "?")"""
    punctuation_match = TRAILING_PUNCTUATION.search(question)
    if punctuation_match is None:
        return question.split(), ""
    return question[:punctuation_match.start()].split(), punctuation_match.group().strip()


def scrambled(question_words, rng):
    shuffled_words = rng.sample(question_words, len(question_words))
    return shuffled_words if shuffled_words != question_words else question_words[1:] + question_words[:1]


def anchor_index(question_words, context, lang):
    """The longest question word (in aksharas) that occurs in the passage; the longest word when none does."""
    passage_words = set(words(normalize(context, lang)))
    word_lengths = [len(aksharas(normalize(word, lang))) for word in question_words]
    anchored_indices = [index for index, word in enumerate(question_words) if normalize(word, lang) in passage_words]
    return max(anchored_indices or range(len(question_words)), key=lambda index: word_lengths[index])


def replacement_words(context, language_contexts, lang, min_aksharas):
    """Words of the other passages that are absent from this one: letters only, at least `min_aksharas` long."""
    passage_words = set(words(normalize(context, lang)))
    return sorted({word for other_context in language_contexts if other_context != context
                   for word in words(normalize(other_context, lang))
                   if LETTERS_ONLY.fullmatch(word) and len(aksharas(word)) >= min_aksharas
                   and word not in passage_words})


def swap_candidates(anchor_word, context, language_contexts, lang):
    """Words at least as long as the anchor (content words, not short function words), else any of 3+ aksharas."""
    anchor_length = len(aksharas(normalize(anchor_word, lang)))
    return (replacement_words(context, language_contexts, lang, max(anchor_length, MIN_REPLACEMENT_AKSHARAS))
            or replacement_words(context, language_contexts, lang, MIN_REPLACEMENT_AKSHARAS))


def passages_without_gold(context, gold_answers, language_contexts, lang):
    normalized_golds = [normalize(gold, lang) for gold in gold_answers if normalize(gold, lang)]
    return sorted({other_context for other_context in language_contexts if other_context != context
                   and not any(gold in normalize(other_context, lang) for gold in normalized_golds)})


def perturb(item, condition, language_contexts):
    """QA input for `condition`; `language_contexts` holds every QA passage of the item's language.
    Seeded by (item id, condition), so a smoke run perturbs an item exactly as the full run does."""
    rng = random.Random(f"{item['id']}|{condition}")
    lang, context = item["lang"], item["context"]
    question_words, trailing_punctuation = split_trailing_punctuation(item["ref_text"])
    unchanged_task = QaInput(item["ref_text"], context, tuple(item["answers"]), item["answerable"])

    def asked(perturbed_words):
        return " ".join(perturbed_words) + trailing_punctuation

    if condition == "scramble":
        return replace(unchanged_task, question=asked(scrambled(question_words, rng)))
    if condition == "drop_head":
        return replace(unchanged_task, question=asked(question_words[1:]))
    if condition == "drop_tail":
        return replace(unchanged_task, question=asked(question_words[:-1]))
    if condition == "swap_anchor":
        # ponytail: the swapped word may not carry the question's meaning, leaving it answerable; the judge and the
        # human audit catch those. Swap a POS-tagged noun instead if they turn out common.
        swap_index = anchor_index(question_words, context, lang)
        replacement = rng.choice(swap_candidates(question_words[swap_index], context, language_contexts, lang))
        swapped_words = [*question_words[:swap_index], replacement, *question_words[swap_index + 1:]]
        return QaInput(asked(swapped_words), context, (), False)
    if condition == "wrong_passage":
        other_passages = passages_without_gold(context, item["answers"], language_contexts, lang)
        if not other_passages:
            raise ValueError(f"{item['id']}: no other {lang} passage lacks the gold answers {item['answers']}")
        return QaInput(item["ref_text"], rng.choice(other_passages), (), False)
    raise ValueError(f"unknown text condition {condition!r}; expected one of {TEXT_CONDITIONS}")
