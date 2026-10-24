"""Evaluate ASR -> LLM -> judge. Each stage runs in its own child process, so only one model is ever loaded.

    PYTHONPATH=. uv run python scripts/run_eval.py --llm_gguf A.gguf --judge_gguf B.gguf --out runs/main

Outputs (append-only, resumable, keyed by (id, condition)): <out>/asr.jsonl, llm.jsonl, judge.jsonl,
plus sysinfo_<stage>.json. Metrics are computed later by make_report.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("asr", "llm", "judge")
ORACLE = "oracle"
JUDGE_MAX_TOKENS = 64
LLM_PORT = 8090


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_keys(path):
    return {(row["id"], row["condition"]) for row in read_jsonl(path)}


def prepare_output(path, force):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if force:
        Path(path).write_text("", encoding="utf-8")


def select_items(items, limit):
    """First `limit` items per (track, lang), keeping file order; all items when limit is None."""
    if limit is None:
        return tuple(items)
    seen_counts = {}
    selected = []
    for item in items:
        group = (item["track"], item["lang"])
        if seen_counts.get(group, 0) < limit:
            seen_counts[group] = seen_counts.get(group, 0) + 1
            selected.append(item)
    return tuple(selected)


def work_keys(stage, items, conditions):
    """ASR covers both tracks; LLM and judge cover QA items under every condition plus the gold-text oracle."""
    if stage == "asr":
        return tuple((item["id"], condition) for condition in conditions for item in items)
    qa_items = [item for item in items if item["track"] == "qa"]
    return tuple((item["id"], condition) for condition in (*conditions, ORACLE) for item in qa_items)


def pending_keys(keys, out_path):
    already_done = done_keys(out_path)
    return tuple(key for key in keys if key not in already_done)


def run_pending(keys, process, out_path, warmup):
    """Process keys not yet in out_path, appending one row each. Warm-ups re-run the first pending key, unrecorded."""
    keys_to_run = pending_keys(keys, out_path)
    for _ in range(warmup if keys_to_run else 0):
        process(keys_to_run[0])
    for position, key in enumerate(keys_to_run, start=1):
        append_jsonl(out_path, process(key))
        print(f"  {position}/{len(keys_to_run)} {key[0]} {key[1]}", flush=True)
    return len(keys_to_run)


def question_for(item, condition, asr_hypotheses):
    return item["ref_text"] if condition == ORACLE else asr_hypotheses[(item["id"], condition)]


def judge_with_retry(call_judge, answerable):
    """Ask once; on malformed output ask again with a stricter reminder; then give up as JUDGE_ERROR."""
    from indic_s2t.scoring import judge_label, parse_judge
    raw_outputs = []
    parsed_judgement = None
    for strict in (False, True):
        raw_outputs.append(call_judge(strict))
        parsed_judgement = parse_judge(raw_outputs[-1])
        if parsed_judgement is not None:
            break
    return {"judge_outputs": raw_outputs, "judgement": parsed_judgement, "attempts": len(raw_outputs),
            "judge_label": judge_label(parsed_judgement, answerable)}


def run_stages(stages, forwarded_args, runner=subprocess.run):
    for stage in stages:
        print(f"== stage {stage}", flush=True)
        runner([sys.executable, __file__, *forwarded_args, "--_stage", stage], check=True)


def stage_environment(stage_name, args, extra):
    from indic_s2t import sysinfo
    return {"stage": stage_name, "args": {key: str(value) for key, value in vars(args).items()},
            "peak_rss_mb_self": sysinfo.peak_rss_mb(), "peak_rss_mb_children": sysinfo.peak_rss_mb(include_children=True),
            "environment": sysinfo.snapshot(), **extra}


def write_stage_info(args, stage_name, extra):
    info_path = args.out / f"sysinfo_{stage_name}.json"
    info_path.write_text(json.dumps(stage_environment(stage_name, args, extra), ensure_ascii=False, indent=2),
                         encoding="utf-8")


def load_items(args):
    return select_items(read_jsonl(args.asr_track) + read_jsonl(args.qa_track), args.limit)


def stage_work_keys(args):
    if args.child_stage == "judge":
        return tuple((row["id"], row["condition"]) for row in read_jsonl(args.out / "llm.jsonl"))
    return work_keys(args.child_stage, load_items(args), args.conditions)


def asr_stage(args):
    import dataclasses

    import numpy as np
    import torch

    from indic_s2t.asr import load_asr, transcribe
    from indic_s2t.audio import BabbleSource, apply_condition, load_16k_mono

    items_by_id = {item["id"]: item for item in load_items(args)}
    babble_pools = {}
    for babble_row in read_jsonl(args.babble_pool):
        babble_pools.setdefault(babble_row["lang"], []).append(
            BabbleSource(babble_row["utt_id"], babble_row["speaker_id"], load_16k_mono(babble_row["audio"])))
    model, asr_info = load_asr(args.asr_threads)
    print(asr_info, f"torch threads {torch.get_num_threads()}", flush=True)
    clean_waves = {}

    def process(key):
        item_id, condition = key
        item = items_by_id[item_id]
        if item_id not in clean_waves:
            clean_waves.clear()  # ponytail: keep one clean wave cached; items repeat across conditions, not ids
            clean_waves[item_id] = load_16k_mono(item["audio"])
        degraded_wave = apply_condition(clean_waves[item_id], condition, item_id,
                                        tuple(babble_pools.get(item["lang"], ())),
                                        item.get("meta", {}).get("speaker_id"))
        start = time.perf_counter()
        hypothesis = transcribe(model, degraded_wave, item["lang"], args.asr_decoding)
        asr_seconds = time.perf_counter() - start
        return {"id": item_id, "condition": condition, "track": item["track"], "lang": item["lang"],
                "ref_text": item["ref_text"], "hyp": hypothesis, "asr_s": asr_seconds,
                "audio_s": len(degraded_wave) / 16000, "decoding": args.asr_decoding,
                "peak_abs": float(np.abs(degraded_wave).max())}

    out_path = args.out / "asr.jsonl"
    count = run_pending(work_keys("asr", tuple(items_by_id.values()), args.conditions), process, out_path, args.warmup)
    print(f"asr: wrote {count} rows to {out_path}")
    write_stage_info(args, "asr", {"asr_info": dataclasses.asdict(asr_info), "torch_threads": torch.get_num_threads()})


def server_for(args, gguf, stage_name):
    from indic_s2t import llamacpp_server, sysinfo
    vk_device = args.vk_device or sysinfo.pick_dgpu(
        sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(args.server_bin))).device_id
    return llamacpp_server.running_server(args.server_bin, gguf, vk_device, ngl=args.ngl, ctx=args.ctx, port=LLM_PORT,
                                          log_path=args.out / f"{stage_name}.server.log")


def server_info_dict(server_info):
    return {"load_seconds": server_info.load_seconds, "offload_line": server_info.offload_line,
            "device_line": server_info.device_line, "vram_used_mib": server_info.vram_used_mib}


def llm_stage(args):
    from indic_s2t import llm, sysinfo

    items_by_id = {item["id"]: item for item in load_items(args)}
    asr_hypotheses = {(row["id"], row["condition"]): row["hyp"] for row in read_jsonl(args.out / "asr.jsonl")}
    asr_aware = not args.no_asr_aware_prompt

    with server_for(args, args.llm_gguf, "llm") as server_info:
        def process(key):
            item_id, condition = key
            item = items_by_id[item_id]
            question = question_for(item, condition, asr_hypotheses)
            completion = llm.stream_chat(server_info.base_url, llm.qa_messages(question, item["context"], item["lang"],
                                                                                asr_aware), args.max_new_tokens)
            return {"id": item_id, "condition": condition, "lang": item["lang"], "question": question,
                    "ref_question": item["ref_text"], "context": item["context"], "answers": item["answers"],
                    "answerable": item["answerable"], "response": completion.text,
                    "reasoning_text": completion.reasoning_text, "ttft_s": completion.ttft_s,
                    "total_s": completion.total_s, "timings": completion.timings,
                    "gpu": sysinfo.dgpu_state(dgpu_card_dir)}  # sampled right after the request, to spot throttling

        dgpu_card_dir = sysinfo.dgpu_card()
        out_path = args.out / "llm.jsonl"
        count = run_pending(work_keys("llm", tuple(items_by_id.values()), args.conditions), process, out_path,
                            args.warmup)
    print(f"llm: wrote {count} rows to {out_path}")
    reasoning_leaks = sum(bool(row["reasoning_text"]) for row in read_jsonl(out_path))
    if reasoning_leaks:
        print(f"WARNING: {reasoning_leaks} responses carry reasoning_content (thinking not disabled)", flush=True)
    write_stage_info(args, "llm", {"server": server_info_dict(server_info), "gguf": str(args.llm_gguf),
                                   "reasoning_leaks": reasoning_leaks, "asr_aware_prompt": asr_aware})


def judge_stage(args):
    from indic_s2t import llm
    from indic_s2t.scoring import judge_messages

    llm_rows = {(row["id"], row["condition"]): row for row in read_jsonl(args.out / "llm.jsonl")}
    with server_for(args, args.judge_gguf, "judge") as server_info:
        def process(key):
            llm_row = llm_rows[key]

            def call_judge(strict):
                return llm.stream_chat(server_info.base_url, judge_messages(
                    llm_row["context"], llm_row["ref_question"], llm_row["answers"], llm_row["response"],
                    llm_row["lang"], strict=strict), JUDGE_MAX_TOKENS).text

            return {"id": key[0], "condition": key[1], **judge_with_retry(call_judge, llm_row["answerable"])}

        out_path = args.out / "judge.jsonl"
        count = run_pending(tuple(llm_rows), process, out_path, args.warmup)
    print(f"judge: wrote {count} rows to {out_path}")
    write_stage_info(args, "judge", {"server": server_info_dict(server_info), "gguf": str(args.judge_gguf)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--conditions", nargs="+", default=["clean", "babble_10db", "telephony"])
    parser.add_argument("--asr_decoding", choices=("rnnt", "ctc"), default="rnnt")
    parser.add_argument("--asr_threads", type=int, default=8)
    parser.add_argument("--llm_gguf", type=Path)
    parser.add_argument("--judge_gguf", type=Path)
    parser.add_argument("--server_bin", default="third_party/llama.cpp/build/bin/llama-server")
    parser.add_argument("--vk_device", help="default: the Vulkan device named RX 6650M")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--no_asr_aware_prompt", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, help="first N items per (track, language)")
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--force", action="store_true", help="truncate the outputs of the stages being run")
    parser.add_argument("--asr_track", type=Path, default=Path("data/asr_track.jsonl"))
    parser.add_argument("--qa_track", type=Path, default=Path("data/qa_track.jsonl"))
    parser.add_argument("--babble_pool", type=Path, default=Path("data/babble_pool.jsonl"))
    parser.add_argument("--_stage", dest="child_stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_stage == "llm" and not args.llm_gguf:
        parser.error("--llm_gguf is required for the llm stage")
    if args.child_stage == "judge" and not args.judge_gguf:
        parser.error("--judge_gguf is required for the judge stage")
    return args


def main():
    args = parse_args()
    if args.child_stage is None:
        forwarded_args = [token for token in sys.argv[1:] if token != "--force"]
        if args.force:
            for stage in args.stages:
                prepare_output(args.out / f"{stage}.jsonl", force=True)
        run_stages(args.stages, forwarded_args)
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are local; the ASR loader would otherwise hit the network
    stage_output = args.out / f"{args.child_stage}.jsonl"
    prepare_output(stage_output, force=False)
    if not pending_keys(stage_work_keys(args), stage_output):
        print(f"{args.child_stage}: nothing pending in {stage_output}, model not loaded", flush=True)
        return
    {"asr": asr_stage, "llm": llm_stage, "judge": judge_stage}[args.child_stage](args)


if __name__ == "__main__":
    main()
