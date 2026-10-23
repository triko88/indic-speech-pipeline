"""ASR and agreement metrics, implemented here (jiwer is only used in tests to cross-check)."""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from indic_s2t.textnorm import aksharas, words


@dataclass(frozen=True)
class EditCounts:
    sub: int
    dele: int
    ins: int
    ref_len: int

    @property
    def errors(self):
        return self.sub + self.dele + self.ins


def edit_counts(reference_tokens, hypothesis_tokens):
    """Levenshtein alignment; each DP cell holds (cost, sub, del, ins) so S/D/I come out with the minimum cost."""
    previous_row = [(column, 0, 0, column) for column in range(len(hypothesis_tokens) + 1)]
    for row, reference_token in enumerate(reference_tokens, start=1):
        current_row = [(row, 0, row, 0)]
        for column, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            diagonal = previous_row[column - 1]
            if reference_token == hypothesis_token:
                match_or_sub = diagonal
            else:
                match_or_sub = (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3])
            deletion = previous_row[column]
            insertion = current_row[column - 1]
            current_row.append(min(match_or_sub,
                                   (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                                   (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
                                   key=lambda cell: cell[0]))
        previous_row = current_row
    _, substitutions, deletions, insertions = previous_row[-1]
    return EditCounts(substitutions, deletions, insertions, len(reference_tokens))


def corpus_error_rate(pairs, tokenize):
    """Σ(S+D+I) / Σ reference tokens over (reference, hypothesis) pairs of normalised strings."""
    all_counts = [edit_counts(tokenize(reference), tokenize(hypothesis)) for reference, hypothesis in pairs]
    return sum(counts.errors for counts in all_counts) / sum(counts.ref_len for counts in all_counts)


def corpus_wer(pairs):
    return corpus_error_rate(pairs, words)


def corpus_cer(pairs):
    return corpus_error_rate(pairs, tuple)  # code points, spaces kept


def corpus_akshara_cer(pairs):
    return corpus_error_rate(pairs, aksharas)  # grapheme clusters, spaces kept


def exact_match_rate(pairs):
    return sum(reference == hypothesis for reference, hypothesis in pairs) / len(pairs)


def bootstrap_wer_ci(pairs, n_resamples=2000, seed=0, confidence=0.95):
    """Percentile bootstrap over utterances. ponytail: ignores speaker clustering, so CIs are optimistic."""
    all_counts = [edit_counts(words(reference), words(hypothesis)) for reference, hypothesis in pairs]
    errors = np.array([counts.errors for counts in all_counts])
    reference_lengths = np.array([counts.ref_len for counts in all_counts])
    resample_indices = np.random.default_rng(seed).integers(0, len(pairs), size=(n_resamples, len(pairs)))
    resampled_wers = errors[resample_indices].sum(axis=1) / reference_lengths[resample_indices].sum(axis=1)
    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(resampled_wers, tail)), float(np.percentile(resampled_wers, 100 - tail))


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - half_width, center + half_width


def cohen_kappa(labels_a, labels_b):
    total = len(labels_a)
    observed_agreement = sum(label_a == label_b for label_a, label_b in zip(labels_a, labels_b)) / total
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    expected_agreement = sum(counts_a[label] * counts_b[label] for label in counts_a) / total**2
    if expected_agreement == 1:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def confusion(labels_a, labels_b):
    """Rows = labels_a, columns = labels_b, over the sorted union of labels."""
    label_order = tuple(sorted(set(labels_a) | set(labels_b)))
    pair_counts = Counter(zip(labels_a, labels_b))
    return label_order, tuple(tuple(pair_counts[(row_label, column_label)] for column_label in label_order)
                              for row_label in label_order)


def mcnemar_exact(only_first_correct, only_second_correct):
    """Two-sided exact McNemar p-value on the discordant counts (binomial, p = 0.5)."""
    discordant = only_first_correct + only_second_correct
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, count) for count in range(min(only_first_correct, only_second_correct) + 1))
    return min(1.0, 2 * tail / 2**discordant)
