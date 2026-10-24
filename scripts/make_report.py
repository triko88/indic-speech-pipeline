"""Turn a run directory into report.md (plus a blind human-audit sheet). Every number is computed from runs/ files.

    PYTHONPATH=. uv run python scripts/make_report.py --run runs/main [--variance runs/var_1 runs/var_2 runs/var_3]
                                                       [--audit_filled runs/main/audit_sheet.csv]
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from indic_s2t import metrics
from indic_s2t.scoring import label_response, qa_rates
from indic_s2t.textnorm import normalize, words

ORACLE = "oracle"
CORRECT_LABELS = ("CORRECT", "CORRECT_ABSTAIN")
RULE_LABELS = ("CORRECT", "CORRECT_ABSTAIN", "FALSE_ABSTAIN", "WRONG_GROUNDED", "HALLUCINATED")
FAILURE_TYPES = ("FALSE_ABSTAIN", "WRONG_GROUNDED", "HALLUCINATED")
WER_BUCKETS = ("0", "(0,20%]", "(20,50%]", ">50%")
AUDIT_SIZE = 40
QUESTIONS_PATH = Path("data/qa_questions.jsonl")
EXAMPLES_PER_TYPE = 3

METRIC_DEFINITIONS = """\
**Normalisation** (identical for reference and hypothesis): Unicode NFC; strip ZWJ, ZWNJ, ZWSP, BOM; indic-nlp-library
normaliser with nasal unification (`nasals_mode=to_nasal_consonants`: anusvara before a stop becomes the
homorganic nasal + virama, so हिंदू = हिन्दू; a no-op for Tamil); Indic digits to ASCII; replace `\\p{P}\\p{S}` (incl. danda and double danda) with spaces; collapse whitespace;
lowercase. Nukta stripping is reported as a separate column (`WER no-nukta`).

**ASR.** Own Levenshtein alignment with S/D/I counts. Corpus WER = Σ(S+D+I) / Σ reference words. CER over Unicode code
points, spaces kept. Akshara-CER over grapheme clusters (`regex` `\\X`; a Devanagari conjunct is one cluster, a Tamil
pulli consonant is its own). Exact match = normalised hypothesis equals normalised reference. WER 95% CI: percentile
bootstrap over utterances, 2000 resamples (ignores speaker clustering).

**Response labels** (rule-based, mutually exclusive). Answerable question: `FALSE_ABSTAIN` if the response contains
`NO_ANSWER` or a per-language abstain phrase; else `CORRECT` if a normalised gold answer of at least 2 aksharas is a
substring of the normalised response, or token-F1 ≥ 0.5; else passage support = share of response tokens whose first
⌈70%⌉ aksharas occur in the normalised passage; support ≥ 0.6 → `WRONG_GROUNDED`, else `HALLUCINATED`. Unanswerable
question: `CORRECT_ABSTAIN` if abstained, else `HALLUCINATED`. Flags: `verbose` (> 25 tokens), `off_script`
(non-abstained response with < 50% of letters in the language's Unicode block).

**Rates.** Correctness = (CORRECT + CORRECT_ABSTAIN) / all, Wilson 95% CI. Answerable accuracy = CORRECT / answerable.
Abstention recall = CORRECT_ABSTAIN / unanswerable. False-abstain rate = FALSE_ABSTAIN / answerable. Hallucination rate
= HALLUCINATED / all, and / non-abstained responses, Wilson 95% CI.

**Error attribution** (per language × condition, against the gold-text `oracle` run of the same items): both correct;
ASR-induced (oracle correct, ASR wrong); LM failure (both wrong); ASR-only correct (oracle wrong, ASR correct).
Δ = ASR correctness − oracle correctness in percentage points. p-values: exact two-sided McNemar on the discordant pairs.

**Latency.** 3 warm-ups per stage excluded; `time.perf_counter`. ASR: seconds per utterance, RTF = Σ ASR time / Σ audio
time, and p95 of per-utterance RTF. LLM: client TTFT and total time; server prompt and decode tokens/s from llama-server
`timings`. E2E = ASR time + LLM total time; speech-end → first token = ASR time + TTFT. Stages run in separate processes,
so E2E is the sum of separately measured stages.
"""


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def is_correct(verdict):
    return verdict.label in CORRECT_LABELS


def rule_verdicts(llm_rows):
    return {(row["id"], row["condition"]): label_response(row["response"], row["answers"], row["context"],
                                                          row["answerable"], row["lang"])
            for row in llm_rows}


def percentile(values, quantile):
    present_values = [value for value in values if value is not None]
    return float(np.percentile(present_values, quantile)) if present_values else None


def asr_table(asr_rows):
    groups = defaultdict(list)
    for row in asr_rows:
        groups[(row["track"], row["lang"], row["condition"])].append(row)
    table = []
    for (track, lang, condition), group_rows in sorted(groups.items()):
        pairs = tuple((normalize(row["ref_text"], lang), normalize(row["hyp"], lang)) for row in group_rows)
        no_nukta_pairs = tuple((normalize(row["ref_text"], lang, strip_nukta=True),
                                normalize(row["hyp"], lang, strip_nukta=True)) for row in group_rows)
        table.append({"track": track, "lang": lang, "condition": condition, "n": len(pairs),
                      "wer": metrics.corpus_wer(pairs), "wer_ci": metrics.bootstrap_wer_ci(pairs),
                      "wer_no_nukta": metrics.corpus_wer(no_nukta_pairs), "cer": metrics.corpus_cer(pairs),
                      "akshara_cer": metrics.corpus_akshara_cer(pairs), "exact_match": metrics.exact_match_rate(pairs)})
    return table


def qa_table(llm_rows, verdicts):
    groups = defaultdict(list)
    for row in llm_rows:
        groups[(row["lang"], row["condition"])].append(row)
    table = []
    for (lang, condition), group_rows in sorted(groups.items()):
        group_verdicts = [verdicts[(row["id"], row["condition"])] for row in group_rows]
        table.append({"lang": lang, "condition": condition,
                      **qa_rates(tuple((verdict.label, row["answerable"]) for verdict, row in zip(group_verdicts, group_rows))),
                      "verbose_rate": sum(verdict.verbose for verdict in group_verdicts) / len(group_rows),
                      "off_script_rate": sum(verdict.off_script for verdict in group_verdicts) / len(group_rows),
                      "reasoning_leaks": sum(bool(row.get("reasoning_text")) for row in group_rows)})
    return table


def correctness_by_id(llm_rows, verdicts, lang, condition):
    return {row["id"]: is_correct(verdicts[(row["id"], row["condition"])])
            for row in llm_rows if row["lang"] == lang and row["condition"] == condition}


def attribution_table(llm_rows, verdicts):
    lang_conditions = sorted({(row["lang"], row["condition"]) for row in llm_rows if row["condition"] != ORACLE})
    table = []
    for lang, condition in lang_conditions:
        condition_correct = correctness_by_id(llm_rows, verdicts, lang, condition)
        oracle_correct = correctness_by_id(llm_rows, verdicts, lang, ORACLE)
        clean_correct = correctness_by_id(llm_rows, verdicts, lang, "clean")
        shared_ids = sorted(set(condition_correct) & set(oracle_correct))
        pairs = [(oracle_correct[item_id], condition_correct[item_id]) for item_id in shared_ids]
        asr_induced = sum(oracle and not asr for oracle, asr in pairs)
        asr_only_correct = sum(asr and not oracle for oracle, asr in pairs)
        p_vs_clean = None
        if condition != "clean" and clean_correct:
            clean_ids = sorted(set(condition_correct) & set(clean_correct))
            p_vs_clean = metrics.mcnemar_exact(
                sum(clean_correct[item_id] and not condition_correct[item_id] for item_id in clean_ids),
                sum(condition_correct[item_id] and not clean_correct[item_id] for item_id in clean_ids))
        table.append({"lang": lang, "condition": condition, "n": len(pairs),
                      "both_correct": sum(oracle and asr for oracle, asr in pairs), "asr_induced": asr_induced,
                      "lm_failure": sum(not oracle and not asr for oracle, asr in pairs),
                      "asr_only_correct": asr_only_correct,
                      "correctness": sum(asr for _, asr in pairs) / len(pairs) if pairs else None,
                      "oracle_correctness": sum(oracle for oracle, _ in pairs) / len(pairs) if pairs else None,
                      "delta_pp": 100 * (sum(asr for _, asr in pairs) - sum(oracle for oracle, _ in pairs)) / len(pairs)
                      if pairs else None,
                      "p_vs_oracle": metrics.mcnemar_exact(asr_only_correct, asr_induced), "p_vs_clean": p_vs_clean})
    return table


def question_wer(row):
    counts = metrics.edit_counts(words(normalize(row["ref_question"], row["lang"])),
                                 words(normalize(row["question"], row["lang"])))
    return counts.errors / counts.ref_len if counts.ref_len else 0.0


def wer_bucket(question_error_rate):
    if question_error_rate == 0:
        return "0"
    if question_error_rate <= 0.2:
        return "(0,20%]"
    return "(20,50%]" if question_error_rate <= 0.5 else ">50%"


def wer_bucket_table(llm_rows, verdicts):
    bucket_correct = defaultdict(list)
    for row in llm_rows:
        if row["condition"] != ORACLE:
            bucket_correct[wer_bucket(question_wer(row))].append(is_correct(verdicts[(row["id"], row["condition"])]))
    return [{"bucket": bucket, "n": len(bucket_correct[bucket]),
             "correctness": sum(bucket_correct[bucket]) / len(bucket_correct[bucket]) if bucket_correct[bucket] else None,
             "correctness_ci": metrics.wilson_ci(sum(bucket_correct[bucket]), len(bucket_correct[bucket]))}
            for bucket in WER_BUCKETS]


def tokens_per_second(token_count, milliseconds):
    return token_count / milliseconds * 1000 if token_count and milliseconds else None


def throttled_flags(prompt_speeds):
    """True where server-side prompt speed fell below half the run's p90 (unthrottled) speed: the GPU clocked down.
    The reference is p90, not the median, because a hot run can spend most of its requests throttled."""
    present_speeds = [speed for speed in prompt_speeds if speed is not None]
    reference_speed = percentile(present_speeds, 90)
    return [speed is not None and speed < 0.5 * reference_speed for speed in prompt_speeds]


def gpu_values(llm_rows, field):
    return [row["gpu"][field] for row in llm_rows if row.get("gpu") and row["gpu"].get(field) is not None]


def latency_table(asr_rows, llm_rows):
    asr_by_key = {(row["id"], row["condition"]): row for row in asr_rows}
    table = []
    for lang in sorted({row["lang"] for row in (*asr_rows, *llm_rows)}):
        lang_asr = [row for row in asr_rows if row["lang"] == lang]
        lang_llm = [row for row in llm_rows if row["lang"] == lang]
        joined = [(asr_by_key[(row["id"], row["condition"])], row) for row in lang_llm
                  if row["condition"] != ORACLE and (row["id"], row["condition"]) in asr_by_key]
        prompt_speeds = [tokens_per_second(row["timings"].get("prompt_n"), row["timings"].get("prompt_ms")) for row in lang_llm]
        decode_speeds = [tokens_per_second(row["timings"].get("predicted_n"), row["timings"].get("predicted_ms"))
                         for row in lang_llm]
        table.append({
            "lang": lang,
            "asr_s_p50": percentile([row["asr_s"] for row in lang_asr], 50),
            "asr_s_p95": percentile([row["asr_s"] for row in lang_asr], 95),
            "asr_rtf": sum(row["asr_s"] for row in lang_asr) / sum(row["audio_s"] for row in lang_asr) if lang_asr else None,
            "asr_rtf_p95": percentile([row["asr_s"] / row["audio_s"] for row in lang_asr], 95),
            "ttft_p50": percentile([row["ttft_s"] for row in lang_llm], 50),
            "ttft_p95": percentile([row["ttft_s"] for row in lang_llm], 95),
            "llm_total_p50": percentile([row["total_s"] for row in lang_llm], 50),
            "llm_total_p95": percentile([row["total_s"] for row in lang_llm], 95),
            "prompt_tok_s_p50": percentile(prompt_speeds, 50),
            "decode_tok_s_p50": percentile(decode_speeds, 50),
            "prompt_tokens_p50": percentile([row["timings"].get("prompt_n") for row in lang_llm], 50),
            "output_tokens_p50": percentile([row["timings"].get("predicted_n") for row in lang_llm], 50),
            "speech_end_to_first_token_p50": percentile([asr["asr_s"] + row["ttft_s"] for asr, row in joined
                                                         if row["ttft_s"] is not None], 50),
            "speech_end_to_first_token_p95": percentile([asr["asr_s"] + row["ttft_s"] for asr, row in joined
                                                         if row["ttft_s"] is not None], 95),
            "e2e_p50": percentile([asr["asr_s"] + row["total_s"] for asr, row in joined], 50),
            "e2e_p95": percentile([asr["asr_s"] + row["total_s"] for asr, row in joined], 95),
            "throttled_requests": sum(throttled_flags(prompt_speeds)),
            "ttft_p50_unthrottled": percentile([row["ttft_s"] for row, throttled in zip(lang_llm, throttled_flags(prompt_speeds))
                                                if not throttled], 50),
            "gpu_sclk_mhz_p50": percentile(gpu_values(lang_llm, "sclk_mhz"), 50),
            "gpu_junction_c_max": max(gpu_values(lang_llm, "junction_c"), default=None),
        })
    return table


def variance_table(variance_dirs):
    tracked_metrics = ("asr_s_p50", "ttft_p50", "llm_total_p50", "speech_end_to_first_token_p50", "e2e_p50")
    per_run = [{row["lang"]: row for row in latency_table(read_jsonl(run_dir / "asr.jsonl"), read_jsonl(run_dir / "llm.jsonl"))}
               for run_dir in variance_dirs]
    table = []
    for lang in sorted({lang for run in per_run for lang in run}):
        for metric_name in tracked_metrics:
            run_values = [run[lang][metric_name] for run in per_run if lang in run and run[lang][metric_name] is not None]
            if run_values:
                mean_value = sum(run_values) / len(run_values)
                table.append({"lang": lang, "metric": metric_name, "runs": run_values, "mean": mean_value,
                              "spread_pct": 100 * (max(run_values) - min(run_values)) / mean_value if mean_value else None})
    return table


def judge_agreement(verdicts, judge_rows):
    judge_by_key = {(row["id"], row["condition"]): row["judge_label"] for row in judge_rows}
    judged_keys = [key for key in verdicts if key in judge_by_key and judge_by_key[key] != "JUDGE_ERROR"]
    rule_labels = [verdicts[key].label for key in judged_keys]
    judge_labels = [judge_by_key[key] for key in judged_keys]
    label_order, matrix = metrics.confusion(rule_labels, judge_labels)
    return {"n": len(judged_keys), "judge_errors": sum(label == "JUDGE_ERROR" for label in judge_by_key.values()),
            "kappa_5way": metrics.cohen_kappa(rule_labels, judge_labels) if judged_keys else None,
            "kappa_correct": metrics.cohen_kappa([label in CORRECT_LABELS for label in rule_labels],
                                                 [label in CORRECT_LABELS for label in judge_labels]) if judged_keys else None,
            "kappa_hallucinated": metrics.cohen_kappa([label == "HALLUCINATED" for label in rule_labels],
                                                      [label == "HALLUCINATED" for label in judge_labels]) if judged_keys else None,
            "confusion_labels": label_order, "confusion": matrix}


AUDIT_FIELDS = ("id", "condition", "lang", "answerable", "ref_question", "question", "gold_answers", "response",
                "context", "human_label")


def write_audit_sheet(llm_rows, verdicts, sheet_path, size=AUDIT_SIZE, seed=0):
    """Stratified by rule label (round-robin across labels), shuffled, and blind: no rule or judge labels shown."""
    rng = random.Random(seed)
    rows_by_label = defaultdict(list)
    for row in llm_rows:
        rows_by_label[verdicts[(row["id"], row["condition"])].label].append(row)
    for label_rows in rows_by_label.values():
        rng.shuffle(label_rows)
    sampled_rows = []
    while len(sampled_rows) < size and any(rows_by_label.values()):
        for label in sorted(rows_by_label):
            if rows_by_label[label] and len(sampled_rows) < size:
                sampled_rows.append(rows_by_label[label].pop())
    rng.shuffle(sampled_rows)
    with Path(sheet_path).open("w", encoding="utf-8", newline="") as sheet_file:
        writer = csv.DictWriter(sheet_file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in sampled_rows:
            writer.writerow({"id": row["id"], "condition": row["condition"], "lang": row["lang"],
                             "answerable": row["answerable"], "ref_question": row["ref_question"],
                             "question": row["question"], "gold_answers": " | ".join(row["answers"]),
                             "response": row["response"], "context": row["context"], "human_label": ""})


def audit_agreement(sheet_path, verdicts, judge_rows):
    with Path(sheet_path).open(encoding="utf-8") as sheet_file:
        filled_rows = [row for row in csv.DictReader(sheet_file) if row["human_label"].strip()]
    judge_by_key = {(row["id"], row["condition"]): row["judge_label"] for row in judge_rows}
    human_labels = [row["human_label"].strip().upper() for row in filled_rows]
    rule_labels = [verdicts[(row["id"], row["condition"])].label for row in filled_rows]
    judged = [(human_label, judge_by_key[(row["id"], row["condition"])])
              for human_label, row in zip(human_labels, filled_rows) if (row["id"], row["condition"]) in judge_by_key]
    return {"n": len(filled_rows),
            "kappa_human_rule": metrics.cohen_kappa(human_labels, rule_labels) if filled_rows else None,
            "kappa_human_judge": metrics.cohen_kappa(*zip(*judged)) if judged else None}


def qualitative_examples(llm_rows, verdicts, per_type=EXAMPLES_PER_TYPE):
    oracle_correct = {(row["lang"], row["id"]): is_correct(verdicts[(row["id"], ORACLE)])
                      for row in llm_rows if row["condition"] == ORACLE}
    examples = {}
    for failure_type in (*FAILURE_TYPES, "ASR_INDUCED"):
        if failure_type == "ASR_INDUCED":
            matching_rows = [row for row in llm_rows if row["condition"] != ORACLE
                             and oracle_correct.get((row["lang"], row["id"]))
                             and not is_correct(verdicts[(row["id"], row["condition"])])]
        else:
            matching_rows = [row for row in llm_rows if verdicts[(row["id"], row["condition"])].label == failure_type]
        examples[failure_type] = matching_rows[:per_type]
    return examples


def formatted(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, float) for part in value):
        return f"[{value[0]:.{digits}f}, {value[1]:.{digits}f}]"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.{digits}f}"
    return str(value)


def markdown_table(table, columns):
    if not table:
        return "_no data_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(formatted(row.get(column)) for column in columns) + " |" for row in table]
    return "\n".join(lines) + "\n"


def setup_section(run_dir):
    stage_infos = {path.stem.removeprefix("sysinfo_"): json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted(run_dir.glob("sysinfo_*.json"))}
    if not stage_infos:
        return "_no sysinfo files in the run directory_\n"
    lines = ["Hardware: Ryzen 7 6800H (8C/16T), 16 GB RAM; RX 6650M 8 GB (Vulkan, RADV) for the LLM; ASR on CPU.", ""]
    for stage_name, stage_info in stage_infos.items():
        environment = stage_info.get("environment", {})
        lines.append(f"- **{stage_name}**: llama.cpp {environment.get('llama_cpp')}; power {environment.get('power')}; "
                     f"dGPU VRAM used {environment.get('dgpu_vram_used_mib')} MiB; peak RSS self "
                     f"{formatted(stage_info.get('peak_rss_mb_self'), 0)} MB, children "
                     f"{formatted(stage_info.get('peak_rss_mb_children'), 0)} MB")
        for detail_key in ("asr_info", "torch_threads", "server", "gguf", "reasoning_leaks", "asr_aware_prompt"):
            if detail_key in stage_info:
                lines.append(f"  - {detail_key}: {stage_info[detail_key]}")
        lines.append(f"  - packages: {environment.get('packages')}")
    return "\n".join(lines) + "\n"


def selection_section(selection_dir):
    if not list(Path(selection_dir).glob("*.json")):
        return "_LLM selection not run_\n"
    from scripts.select_llm import load_results, summary_markdown
    results = load_results(selection_dir, QUESTIONS_PATH)
    return summary_markdown(results).replace("# LLM selection\n", "", 1).replace("## ", "### ")


def examples_section(examples):
    lines = []
    for failure_type, example_rows in examples.items():
        lines.append(f"**{failure_type}**\n")
        for row in example_rows:
            lines.append(f"- `{row['id']}` ({row['condition']}): gold question «{row['ref_question']}» → heard "
                         f"«{row['question']}»; gold {row['answers']}; response «{row['response']}»")
        lines.append("" if example_rows else "_none_\n")
    return "\n".join(lines) + "\n"


LATENCY_COLUMNS = ("lang", "asr_s_p50", "asr_s_p95", "asr_rtf", "asr_rtf_p95", "ttft_p50", "ttft_p95", "llm_total_p50",
                   "llm_total_p95", "prompt_tok_s_p50", "decode_tok_s_p50", "prompt_tokens_p50", "output_tokens_p50",
                   "speech_end_to_first_token_p50", "speech_end_to_first_token_p95", "e2e_p50", "e2e_p95",
                   "throttled_requests", "ttft_p50_unthrottled", "gpu_sclk_mhz_p50", "gpu_junction_c_max")


def latency_section(run_dirs):
    """One table per run; a run without its own asr.jsonl (an LLM-only re-measurement) reuses the first run's."""
    first_asr_rows = read_jsonl(Path(run_dirs[0]) / "asr.jsonl")
    sections = []
    for latency_run_dir in run_dirs:
        asr_rows = read_jsonl(Path(latency_run_dir) / "asr.jsonl") or first_asr_rows
        sections.append(f"### Latency: {Path(latency_run_dir).name}\n\n"
                        + markdown_table(latency_table(asr_rows, read_jsonl(Path(latency_run_dir) / "llm.jsonl")),
                                         LATENCY_COLUMNS))
    return "\n".join(sections)


def build_report(run_dir, selection_dir, variance_dirs, audit_filled=None, latency_runs=()):
    run_dir = Path(run_dir)
    asr_rows, llm_rows, judge_rows = (read_jsonl(run_dir / f"{stage}.jsonl") for stage in ("asr", "llm", "judge"))
    verdicts = rule_verdicts(llm_rows)
    sheet_path = run_dir / "audit_sheet.csv"
    if llm_rows and not sheet_path.exists():  # never overwrite a sheet that may already hold human labels
        write_audit_sheet(llm_rows, verdicts, sheet_path)
    agreement = judge_agreement(verdicts, judge_rows) if judge_rows else None
    audit = audit_agreement(audit_filled, verdicts, judge_rows) if audit_filled else None
    confusion_text = ""
    if agreement and agreement["n"]:
        confusion_text = "Confusion (rows = rule, columns = judge):\n\n" + markdown_table(
            [{"rule": label, **dict(zip(agreement["confusion_labels"], counts))}
             for label, counts in zip(agreement["confusion_labels"], agreement["confusion"])],
            ("rule", *agreement["confusion_labels"]))
    sections = [
        "# Indic speech → text-response evaluation (hi, ta)\n",
        "## Setup\n", setup_section(run_dir),
        "## Metric definitions\n", METRIC_DEFINITIONS,
        "## LLM selection and tokenizer fertility\n", selection_section(selection_dir),
        "## ASR\n", markdown_table(asr_table(asr_rows), ("track", "lang", "condition", "n", "wer", "wer_ci", "wer_no_nukta",
                                                       "cer", "akshara_cer", "exact_match")),
        "## Spoken QA\n", markdown_table(qa_table(llm_rows, verdicts), (
            "lang", "condition", "n", "correctness", "correctness_ci", "answerable_accuracy", "abstention_recall",
            "false_abstain_rate", "hallucination_rate", "hallucination_ci", "hallucination_rate_non_abstained",
            "verbose_rate", "off_script_rate", "reasoning_leaks")),
        "## Error attribution\n", markdown_table(attribution_table(llm_rows, verdicts), (
            "lang", "condition", "n", "both_correct", "asr_induced", "lm_failure", "asr_only_correct", "correctness",
            "oracle_correctness", "delta_pp", "p_vs_oracle", "p_vs_clean")),
        "## Correctness by question WER\n", markdown_table(wer_bucket_table(llm_rows, verdicts),
                                                           ("bucket", "n", "correctness", "correctness_ci")),
        "## Latency\n",
        "Stages run in separate processes, so E2E is the sum of separately measured stages. Seconds unless noted. "
        "`throttled_requests`: server-side prompt speed below half the run's p90 (unthrottled) speed.\n\n",
        latency_section((run_dir, *latency_runs)),
        "## Variance\n", markdown_table(variance_table(variance_dirs), ("lang", "metric", "runs", "mean", "spread_pct"))
        if variance_dirs else "_variance runs not provided_\n",
        "## Judge agreement\n",
        "A small, 4-bit, local judge from another model family has limited validity: treat κ as a sanity check on the "
        "rules, not as ground truth.\n\n",
        markdown_table([agreement], ("n", "judge_errors", "kappa_5way", "kappa_correct", "kappa_hallucinated"))
        if agreement else "_judge stage not run_\n", confusion_text,
        "## Human audit\n",
        f"Blind sheet: `{sheet_path}` ({AUDIT_SIZE} rows stratified by rule label; fill `human_label` with one of "
        f"{', '.join(RULE_LABELS)}, then rerun with `--audit_filled`).\n\n",
        markdown_table([audit], ("n", "kappa_human_rule", "kappa_human_judge")) if audit else "_not filled yet_\n",
        "## Qualitative examples\n", examples_section(qualitative_examples(llm_rows, verdicts)),
    ]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--selection", type=Path, default=Path("runs/llm_selection"))
    parser.add_argument("--variance", type=Path, nargs="*", default=())
    parser.add_argument("--audit_filled", type=Path)
    parser.add_argument("--latency_runs", type=Path, nargs="*", default=(), help="extra runs shown in the latency section")
    args = parser.parse_args()
    report_path = args.run / "report.md"
    report_path.write_text(build_report(args.run, args.selection, args.variance, args.audit_filled, args.latency_runs), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
