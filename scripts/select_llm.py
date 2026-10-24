"""Benchmark one LLM candidate per invocation (own process, own server), or --summarize all into a recommendation.

  select_llm.py --gguf X.gguf --name A --family gemma --questions data/qa_questions.jsonl --out runs/llm_selection
  select_llm.py --summarize --out runs/llm_selection        # -> runs/llm_selection.md
Pass server extras as one string: --extra_args="--n-cpu-moe 20".
"""

import argparse
import json
import shlex
import statistics
from pathlib import Path

from indic_s2t import llamacpp_server, llm, sysinfo

ANSWERER_CANDIDATES = ("A", "B")
CORRECT_LABELS = ("CORRECT", "CORRECT_ABSTAIN")
LANGS = ("hi", "ta")
CTX_SIZE = 4096
ANSWER_MAX_TOKENS = 96
FERTILITY_PASSAGES_PER_LANG = 20
PILOT_PER_LANG = 10
WARMUP_COUNT = 3


def fertility(token_counts, char_counts):
    return sum(token_counts) / sum(char_counts)


def fertility_passages(questions, lang, limit):
    unique_contexts = dict.fromkeys(question["context"] for question in questions if question["lang"] == lang)
    return tuple(unique_contexts)[:limit]


def pilot_questions(questions, per_lang):
    selected_ids = {question["id"] for lang in LANGS
                    for question in [question for question in questions if question["lang"] == lang][:per_lang]}
    return tuple(question for question in questions if question["id"] in selected_ids)


def median_or_none(values):
    present_values = [value for value in values if value is not None]
    return statistics.median(present_values) if present_values else None


def accuracy(pilot_items):
    return sum(item["label"] in CORRECT_LABELS for item in pilot_items) / len(pilot_items) if pilot_items else None


def pilot_summary(pilot_items):
    return {
        "n": len(pilot_items),
        "accuracy": accuracy(pilot_items),
        "accuracy_by_lang": {lang: accuracy([item for item in pilot_items if item["lang"] == lang]) for lang in LANGS},
        "off_script_rate": sum(item["off_script"] for item in pilot_items) / len(pilot_items),
        "reasoning_leak_count": sum(bool(item["reasoning_text"]) for item in pilot_items),
        "median_ttft_s": median_or_none(item["ttft_s"] for item in pilot_items),
        "median_latency_s": median_or_none(item["total_s"] for item in pilot_items),
        "median_prompt_tok_s": median_or_none(item["timings"].get("prompt_per_second") for item in pilot_items),
        "median_decode_tok_s": median_or_none(item["timings"].get("predicted_per_second") for item in pilot_items),
    }


def relabel(result, questions_by_id):
    """Re-score stored pilot responses with the current rules, so summaries never show labels from older rules."""
    from indic_s2t import scoring
    relabeled_items = []
    for item in result["pilot_items"]:
        question = questions_by_id[item["id"]]
        verdict = scoring.label_response(item["response"], question["answers"], question["context"],
                                         question["answerable"], question["lang"])
        relabeled_items.append({**item, "label": verdict.label, "off_script": verdict.off_script,
                                "verbose": verdict.verbose})
    return {**result, "pilot_items": relabeled_items, "pilot": pilot_summary(relabeled_items)}


def load_results(out_dir, questions_path):
    questions_by_id = {question["id"]: question for question in read_jsonl(questions_path)}
    return tuple(relabel(json.loads(result_path.read_text(encoding="utf-8")), questions_by_id)
                 for result_path in sorted(Path(out_dir).glob("*.json")))


def recommend(results):
    """(answerer, judge): answerer is the best A/B by accuracy then latency; judge is the best other-family model."""
    ranked = sorted(results, key=lambda result: (-result["pilot"]["accuracy"], result["pilot"]["median_latency_s"]))
    answerer = next((result for result in ranked if result["name"] in ANSWERER_CANDIDATES), None)
    if answerer is None:
        raise ValueError("no A/B candidate results to pick an answerer from")
    judge = next((result["name"] for result in ranked
                  if result["name"] != answerer["name"] and result["family"] != answerer["family"]), None)
    return answerer["name"], judge


def formatted(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def summary_markdown(results):
    answerer, judge = recommend(results)
    header = ("| Candidate | Family | Load s | VRAM +MiB | Max prompt tok (fits) | Pilot acc | acc hi / ta | Off-script "
              "| Reasoning leaks | Median TTFT s | Median latency s | Prompt tok/s | Decode tok/s |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for result in sorted(results, key=lambda result: result["name"]):
        server, pilot, context_fit = result.get("server", {}), result["pilot"], result.get("context_fit", {})
        by_lang = pilot.get("accuracy_by_lang", {})
        rows.append(
            f"| {result['name']} | {result['family']} | {formatted(server.get('load_seconds'), 1)} "
            f"| {formatted(server.get('vram_delta_mib'))} "
            f"| {formatted(context_fit.get('max_prompt_tokens'))} ({formatted(context_fit.get('fits'))}) "
            f"| {formatted(pilot['accuracy'])} | {formatted(by_lang.get('hi'))} / {formatted(by_lang.get('ta'))} "
            f"| {formatted(pilot.get('off_script_rate'))} | {formatted(pilot.get('reasoning_leak_count'))} "
            f"| {formatted(pilot.get('median_ttft_s'))} | {formatted(pilot['median_latency_s'])} "
            f"| {formatted(pilot.get('median_prompt_tok_s'), 0)} | {formatted(pilot.get('median_decode_tok_s'), 1)} |")
    fertility_rows = [f"| {result['name']} | {formatted(result.get('fertility', {}).get('hi'), 3)} "
                      f"| {formatted(result.get('fertility', {}).get('ta'), 3)} |"
                      for result in sorted(results, key=lambda result: result["name"])]
    return "\n".join([
        "# LLM selection", "", header, *rows, "",
        "## Tokenizer fertility (tokens per character, lower is better)", "",
        "| Candidate | hi | ta |", "|---|---|---|", *fertility_rows, "",
        "## Recommendation", "",
        f"Answerer: **{answerer}** (best pilot accuracy among {'/'.join(ANSWERER_CANDIDATES)}, "
        f"tie-break lower median latency).",
        f"Judge: **{judge or 'none available'}** (best pilot accuracy among the rest, different family from the answerer).",
        ""])


def read_jsonl(path):
    return tuple(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def run_pilot_item(base_url, question):
    from indic_s2t import scoring  # written in parallel; only needed for real runs
    completion = llm.stream_chat(base_url, llm.qa_messages(question["ref_text"], question["context"], question["lang"]),
                                 ANSWER_MAX_TOKENS)
    verdict = scoring.label_response(completion.text, question["answers"], question["context"],
                                     question["answerable"], question["lang"])
    return {"id": question["id"], "lang": question["lang"], "answerable": question["answerable"],
            "response": completion.text, "reasoning_text": completion.reasoning_text, "label": verdict.label,
            "off_script": verdict.off_script, "verbose": verdict.verbose, "ttft_s": completion.ttft_s,
            "total_s": completion.total_s, "timings": completion.timings}


def context_fit(base_url, questions):
    # ponytail: rendered prompt has no BOS for some templates (server adds it), so counts may be 1 low.
    prompt_tokens = [len(llm.tokenize(base_url, llm.apply_template(
        base_url, llm.qa_messages(question["ref_text"], question["context"], question["lang"]))))
        for question in questions]
    longest_index = max(range(len(questions)), key=prompt_tokens.__getitem__)
    return {"method": "apply-template + tokenize", "max_prompt_tokens": prompt_tokens[longest_index],
            "longest_id": questions[longest_index]["id"], "answer_max_tokens": ANSWER_MAX_TOKENS, "ctx": CTX_SIZE,
            "fits": prompt_tokens[longest_index] + ANSWER_MAX_TOKENS <= CTX_SIZE}


def evaluate_candidate(args):
    questions = read_jsonl(args.questions)
    vk_device = sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(args.server_bin))).device_id
    args.out.mkdir(parents=True, exist_ok=True)
    vram_before_mib = sysinfo.vram_used_mib(sysinfo.dgpu_card())
    with llamacpp_server.running_server(args.server_bin, args.gguf, vk_device, ctx=CTX_SIZE, port=args.port,
                                        extra_args=tuple(shlex.split(args.extra_args)),
                                        log_path=args.out / f"{args.name}.server.log") as server_info:
        base_url = server_info.base_url
        passages_by_lang = {lang: fertility_passages(questions, lang, FERTILITY_PASSAGES_PER_LANG) for lang in LANGS}
        fertility_by_lang = {lang: fertility([len(llm.tokenize(base_url, passage)) for passage in passages],
                                             [len(passage) for passage in passages])
                             for lang, passages in passages_by_lang.items()}
        fit = context_fit(base_url, questions)
        selected_questions = pilot_questions(questions, PILOT_PER_LANG)
        for warmup_question in selected_questions[:WARMUP_COUNT]:
            run_pilot_item(base_url, warmup_question)
        pilot_items = [run_pilot_item(base_url, question) for question in selected_questions]
    result = {
        "name": args.name, "family": args.family, "gguf": str(args.gguf), "extra_args": args.extra_args,
        "server": {"vk_device": vk_device, "load_seconds": server_info.load_seconds,
                   "offload_line": server_info.offload_line, "device_line": server_info.device_line,
                   "vram_used_mib": server_info.vram_used_mib,
                   "vram_delta_mib": server_info.vram_used_mib - vram_before_mib,
                   "log_path": str(server_info.log_path)},
        "fertility": fertility_by_lang, "context_fit": fit, "pilot": pilot_summary(pilot_items),
        "pilot_items": pilot_items, "environment": sysinfo.snapshot(),
    }
    result_path = args.out / f"{args.name}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {result_path}: accuracy {result['pilot']['accuracy']}, fits {fit['fits']}")


def summarize(out_dir, questions_path):
    results = load_results(out_dir, questions_path)
    summary_path = out_dir.with_suffix(".md")
    summary_path.write_text(summary_markdown(results), encoding="utf-8")
    print(f"wrote {summary_path}: answerer/judge = {recommend(results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--name", choices=("A", "B", "C"))
    parser.add_argument("--family", choices=("gemma", "qwen", "sarvam"))
    parser.add_argument("--questions", type=Path, default=Path("data/qa_questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/llm_selection"))
    parser.add_argument("--server_bin", default="third_party/llama.cpp/build/bin/llama-server")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--extra_args", default="", help='one string, e.g. --extra_args="--n-cpu-moe 20"')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.out, args.questions)
    elif not (args.gguf and args.name and args.family):
        parser.error("--gguf, --name and --family are required unless --summarize")
    else:
        evaluate_candidate(args)


if __name__ == "__main__":
    main()
