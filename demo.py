"""Speak a question, get a text answer: IndicConformer (CPU) -> llama-server (Vulkan dGPU).

    PYTHONPATH=. uv run python demo.py --wav question.wav --lang hi --gguf model.gguf [--context passage.txt]

Interactive tool, not a measurement: the ASR model stays in RAM while the server holds the LLM in VRAM.
"""

import argparse
import os
import time
from pathlib import Path

from indic_s2t import llm
from indic_s2t.languages import LANGUAGES

SERVER_BIN = "third_party/llama.cpp/build/bin/llama-server"


def open_question_messages(question, lang):
    instructions = (f"Answer the question briefly in {LANGUAGES[lang].name}. The question was produced by automatic "
                    "speech recognition and may contain recognition errors, so interpret it charitably.")
    return ({"role": "system", "content": instructions}, {"role": "user", "content": question})


def answer_wav(asr_model, base_url, wav_path, lang, context=None, max_new_tokens=96):
    from indic_s2t.asr import transcribe
    from indic_s2t.audio import load_16k_mono
    wave = load_16k_mono(wav_path)
    asr_start = time.perf_counter()
    transcript = transcribe(asr_model, wave, lang)
    asr_seconds = time.perf_counter() - asr_start
    messages = llm.qa_messages(transcript, context, lang) if context else open_question_messages(transcript, lang)
    completion = llm.stream_chat(base_url, messages, max_new_tokens)
    return {"transcript": transcript, "response": completion.text, "audio_s": len(wave) / 16000,
            "asr_s": asr_seconds, "ttft_s": completion.ttft_s, "llm_total_s": completion.total_s,
            "speech_end_to_first_token_s": asr_seconds + (completion.ttft_s or 0.0),
            "e2e_s": asr_seconds + completion.total_s, "timings": completion.timings}


def dgpu_device(server_bin):
    from indic_s2t import sysinfo
    return sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(server_bin))).device_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGES))
    parser.add_argument("--context", type=Path, help="text file with a passage to answer from")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--server_bin", default=SERVER_BIN)
    parser.add_argument("--asr_threads", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from indic_s2t.asr import load_asr_models
    from indic_s2t.llamacpp_server import running_server
    model_id = LANGUAGES[args.lang].asr_model
    asr_model, asr_info = load_asr_models((model_id,), args.asr_threads)[model_id]
    print(asr_info)
    context = args.context.read_text(encoding="utf-8") if args.context else None
    with running_server(args.server_bin, args.gguf, dgpu_device(args.server_bin)) as server_info:
        print(f"server: {server_info.device_line} | {server_info.offload_line} | load {server_info.load_seconds:.1f} s")
        answer = answer_wav(asr_model, server_info.base_url, args.wav, args.lang, context)
    print(f"transcript: {answer['transcript']}\nresponse:   {answer['response']}")
    print(f"audio {answer['audio_s']:.2f} s | ASR {answer['asr_s']:.2f} s | TTFT {answer['ttft_s']:.2f} s | "
          f"LLM total {answer['llm_total_s']:.2f} s | speech-end -> first token "
          f"{answer['speech_end_to_first_token_s']:.2f} s | E2E {answer['e2e_s']:.2f} s")


if __name__ == "__main__":
    main()
