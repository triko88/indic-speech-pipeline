import jiwer
import pytest

from indic_s2t import metrics
from indic_s2t.textnorm import normalize

HINDI_PAIRS = (
    ("भारत की राजधानी नई दिल्ली है।", "भारत की राजधानी दिल्ली है"),
    ("गंगा नदी हिमालय से निकलती है", "गंगा नदी हिमालय से निकलती हैं और बहती है"),
    ("ताजमहल आगरा में है", "ताज महल आगरा में"),
)
TAMIL_PAIRS = (
    ("தமிழ்நாட்டின் தலைநகரம் சென்னை", "தமிழ்நாட்டின் தலைநகர் சென்னையில்"),
    ("காவிரி ஆறு கர்நாடகாவில் உற்பத்தியாகிறது.", "காவேரி ஆறு கர்நாடகாவில் உற்பத்தியாகிறது"),
    ("திருவள்ளுவர் திருக்குறளை எழுதினார்", "திருவள்ளுவர் திருக்குறள் எழுதினார்"),
)


def normalized_pairs(raw_pairs, lang):
    return tuple((normalize(reference, lang), normalize(hypothesis, lang)) for reference, hypothesis in raw_pairs)


@pytest.mark.parametrize("raw_pairs,lang", [(HINDI_PAIRS, "hi"), (TAMIL_PAIRS, "ta")])
def test_wer_and_cer_match_jiwer(raw_pairs, lang):
    pairs = normalized_pairs(raw_pairs, lang)
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    assert metrics.corpus_wer(pairs) == pytest.approx(jiwer.wer(references, hypotheses))
    assert metrics.corpus_cer(pairs) == pytest.approx(jiwer.cer(references, hypotheses))


def test_sub_del_ins_counts_on_toy_example():
    assert metrics.edit_counts(("a", "b", "c", "d"), ("a", "x", "c", "d", "e")) == metrics.EditCounts(sub=1, dele=0, ins=1, ref_len=4)
    assert metrics.edit_counts(("a", "b", "c"), ("a", "c")) == metrics.EditCounts(sub=0, dele=1, ins=0, ref_len=3)
    assert metrics.edit_counts((), ("a",)) == metrics.EditCounts(sub=0, dele=0, ins=1, ref_len=0)


def test_sub_del_ins_match_jiwer_on_hindi():
    pairs = normalized_pairs(HINDI_PAIRS, "hi")
    jiwer_output = jiwer.process_words([reference for reference, _ in pairs], [hypothesis for _, hypothesis in pairs])
    totals = [metrics.edit_counts(tuple(reference.split()), tuple(hypothesis.split())) for reference, hypothesis in pairs]
    assert sum(counts.sub + counts.dele + counts.ins for counts in totals) == (
        jiwer_output.substitutions + jiwer_output.deletions + jiwer_output.insertions)


def test_akshara_cer_counts_grapheme_clusters():
    # one akshara wrong (ली -> ला); the cluster count depends on the regex Unicode version, so derive it
    from indic_s2t.textnorm import aksharas
    reference, hypothesis = "दिल्ली", "दिल्ला"
    expected = 1 / len(aksharas(reference))
    assert metrics.corpus_akshara_cer(((reference, hypothesis),)) == pytest.approx(expected)


def test_exact_match_rate():
    assert metrics.exact_match_rate((("a b", "a b"), ("a", "b"))) == 0.5


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    pairs = normalized_pairs(HINDI_PAIRS + HINDI_PAIRS, "hi")
    low, high = metrics.bootstrap_wer_ci(pairs, n_resamples=2000, seed=0)
    assert low <= metrics.corpus_wer(pairs) <= high
    assert (low, high) == metrics.bootstrap_wer_ci(pairs, n_resamples=2000, seed=0)


def test_wilson_hand_computed():
    low, high = metrics.wilson_ci(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert metrics.wilson_ci(0, 0) == (0.0, 1.0)


def test_cohen_kappa_hand_computed():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: po = 0.7, pe = 0.5 -> kappa = 0.4
    labels_a = ("yes",) * 25 + ("no",) * 25
    labels_b = ("yes",) * 20 + ("no",) * 5 + ("yes",) * 10 + ("no",) * 15
    assert metrics.cohen_kappa(labels_a, labels_b) == pytest.approx(0.4)
    assert metrics.cohen_kappa(("x", "x"), ("x", "x")) == 1.0


def test_confusion_matrix():
    label_order, matrix = metrics.confusion(("a", "a", "b"), ("a", "b", "b"))
    assert label_order == ("a", "b")
    assert matrix == ((1, 1), (0, 1))


def test_mcnemar_exact_hand_computed():
    assert metrics.mcnemar_exact(1, 6) == pytest.approx(0.125)  # 2 * (C(7,0)+C(7,1)) / 2^7
    assert metrics.mcnemar_exact(5, 5) == 1.0
    assert metrics.mcnemar_exact(0, 0) == 1.0
