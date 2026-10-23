"""Rule-based response labels, QA rates, and the LLM-judge prompt/parse/mapping onto the same labels."""

import json
import math
from collections import Counter
from dataclasses import dataclass

import regex

from indic_s2t.metrics import wilson_ci
from indic_s2t.textnorm import aksharas, normalize, script_letter_share, words

NO_ANSWER = "NO_ANSWER"
# ponytail: fixed phrase lists; a response that merely contains one of these is treated as an abstention.
ABSTAIN_PHRASES = {
    "hi": ("no answer", "उत्तर नहीं", "जवाब नहीं", "जानकारी नहीं", "उल्लेख नहीं", "नहीं दिया गया", "पता नहीं"),
    "ta": ("no answer", "பதில் இல்லை", "தகவல் இல்லை", "குறிப்பிடப்படவில்லை", "தெரியவில்லை", "கொடுக்கப்படவில்லை"),
}
MIN_GOLD_AKSHARAS = 2
F1_THRESHOLD = 0.5
SUPPORT_THRESHOLD = 0.6
SUPPORT_PREFIX_SHARE = 0.7
VERBOSE_TOKENS = 25
SCRIPT_SHARE_THRESHOLD = 0.5
LANGUAGE_NAMES = {"hi": "Hindi", "ta": "Tamil"}


@dataclass(frozen=True)
class Verdict:
    label: str  # CORRECT | CORRECT_ABSTAIN | FALSE_ABSTAIN | WRONG_GROUNDED | HALLUCINATED
    abstained: bool
    verbose: bool
    off_script: bool


def is_abstention(response, lang):
    normalized_response = normalize(response, lang)
    return NO_ANSWER in response.upper() or any(
        normalize(phrase, lang) in normalized_response for phrase in ABSTAIN_PHRASES[lang])


def token_f1(normalized_gold, normalized_response):
    gold_tokens, response_tokens = words(normalized_gold), words(normalized_response)
    overlap = sum((Counter(gold_tokens) & Counter(response_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(response_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def passage_support(response, passage, lang):
    """Share of response tokens whose first ⌈70%⌉ aksharas occur in the passage (tolerates agglutinative suffixes)."""
    response_tokens = words(normalize(response, lang))
    normalized_passage = normalize(passage, lang)

    def token_prefix(token):
        token_aksharas = aksharas(token)
        return "".join(token_aksharas[:math.ceil(SUPPORT_PREFIX_SHARE * len(token_aksharas))])

    if not response_tokens:
        return 0.0
    return sum(token_prefix(token) in normalized_passage for token in response_tokens) / len(response_tokens)


def is_correct(response, gold_answers, lang):
    normalized_response = normalize(response, lang)
    normalized_golds = [normalize(gold, lang) for gold in gold_answers]
    return any(
        (len(aksharas(gold)) >= MIN_GOLD_AKSHARAS and gold in normalized_response)
        or token_f1(gold, normalized_response) >= F1_THRESHOLD
        for gold in normalized_golds if gold)


def label_response(response, gold_answers, context, answerable, lang):
    abstained = is_abstention(response, lang)
    if not answerable:
        label = "CORRECT_ABSTAIN" if abstained else "HALLUCINATED"
    elif abstained:
        label = "FALSE_ABSTAIN"
    elif is_correct(response, gold_answers, lang):
        label = "CORRECT"
    else:
        label = "WRONG_GROUNDED" if passage_support(response, context, lang) >= SUPPORT_THRESHOLD else "HALLUCINATED"
    return Verdict(
        label=label,
        abstained=abstained,
        verbose=len(words(normalize(response, lang))) > VERBOSE_TOKENS,
        off_script=not abstained and script_letter_share(response, lang) < SCRIPT_SHARE_THRESHOLD,
    )


def qa_rates(labels_and_answerable):
    """Rates over (label, answerable) pairs. Correct = CORRECT or CORRECT_ABSTAIN."""
    labels = [label for label, _ in labels_and_answerable]
    answerable_labels = [label for label, answerable in labels_and_answerable if answerable]
    unanswerable_labels = [label for label, answerable in labels_and_answerable if not answerable]
    non_abstained_labels = [label for label in labels if label not in ("CORRECT_ABSTAIN", "FALSE_ABSTAIN")]
    correct_count = sum(label in ("CORRECT", "CORRECT_ABSTAIN") for label in labels)
    hallucinated_count = labels.count("HALLUCINATED")

    def share(count, total):
        return count / total if total else float("nan")

    return {
        "n": len(labels),
        "correctness": share(correct_count, len(labels)),
        "correctness_ci": wilson_ci(correct_count, len(labels)),
        "answerable_accuracy": share(answerable_labels.count("CORRECT"), len(answerable_labels)),
        "abstention_recall": share(unanswerable_labels.count("CORRECT_ABSTAIN"), len(unanswerable_labels)),
        "false_abstain_rate": share(answerable_labels.count("FALSE_ABSTAIN"), len(answerable_labels)),
        "hallucination_rate": share(hallucinated_count, len(labels)),
        "hallucination_ci": wilson_ci(hallucinated_count, len(labels)),
        "hallucination_rate_non_abstained": share(non_abstained_labels.count("HALLUCINATED"), len(non_abstained_labels)),
        "hallucination_non_abstained_ci": wilson_ci(non_abstained_labels.count("HALLUCINATED"), len(non_abstained_labels)),
    }


JUDGE_SYSTEM_PROMPT = (
    "You grade answers to reading-comprehension questions in {language}. "
    "You are given a passage, a question, the gold answers (empty if the passage does not answer the question) "
    "and a model response. Decide:\n"
    '- "abstained": true if the response declines to answer (e.g. NO_ANSWER, "not in the passage").\n'
    '- "correct": true if the response gives the same answer as one of the gold answers '
    "(inflected forms, extra particles or case suffixes are fine).\n"
    '- "supported": true if the response content is stated in or directly entailed by the passage.\n'
    'Reply with a single JSON object: {{"abstained": <bool>, "correct": <bool>, "supported": <bool>}}'
)
JUDGE_STRICT_REMINDER = (
    "Your previous output was not valid. Output ONLY the JSON object with exactly the three boolean keys "
    "abstained, correct, supported. No prose, no code fences."
)


def judge_messages(context, question, gold_answers, response, lang, strict=False):
    user_content = (f"Passage:\n{context}\n\nQuestion:\n{question}\n\n"
                    f"Gold answers: {json.dumps(list(gold_answers), ensure_ascii=False)}\n\nResponse:\n{response}")
    if strict:
        user_content += f"\n\n{JUDGE_STRICT_REMINDER}"
    return ({"role": "system", "content": JUDGE_SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[lang])},
            {"role": "user", "content": user_content})


def parse_judge(judge_output):
    """First JSON object in the output with exactly-boolean abstained/correct/supported, else None."""
    for object_match in regex.finditer(r"\{[^{}]*\}", judge_output):
        try:
            parsed = json.loads(object_match.group())
        except json.JSONDecodeError:
            continue
        if all(isinstance(parsed.get(key), bool) for key in ("abstained", "correct", "supported")):
            return {key: parsed[key] for key in ("abstained", "correct", "supported")}
    return None


def judge_label(parsed_judgement, answerable):
    if parsed_judgement is None:
        return "JUDGE_ERROR"
    if not answerable:
        return "CORRECT_ABSTAIN" if parsed_judgement["abstained"] else "HALLUCINATED"
    if parsed_judgement["abstained"]:
        return "FALSE_ABSTAIN"
    if parsed_judgement["correct"]:
        return "CORRECT"
    return "WRONG_GROUNDED" if parsed_judgement["supported"] else "HALLUCINATED"
