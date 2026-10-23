"""Extractive QA prompt and a streaming client for llama-server's OpenAI-compatible chat endpoint."""

import json
import time
from dataclasses import dataclass

import httpx

NO_ANSWER = "NO_ANSWER"
LANGUAGE_NAMES = {"hi": "Hindi", "ta": "Tamil"}


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning_text: str
    ttft_s: float | None  # first non-empty content or reasoning delta; None if the model produced nothing
    total_s: float
    timings: dict  # llama-server `timings` from the final chunk: prompt_n, prompt_ms, predicted_n, ...


def qa_messages(question, context, lang, asr_aware=True):
    """Identical template for oracle and ASR questions: only `question` changes."""
    language_name = LANGUAGE_NAMES[lang]
    instructions = (
        f"Answer the question using ONLY the passage. Answer in {language_name}, the same language as the question, "
        f"with a short span of a few words. Do not explain. "
        f"If the passage does not contain the answer, reply exactly {NO_ANSWER}.")
    if asr_aware:
        instructions += (" The question was produced by automatic speech recognition and may contain recognition "
                         "errors, so interpret it charitably.")
    return ({"role": "system", "content": instructions},
            {"role": "user", "content": f"Passage:\n{context}\n\nQuestion: {question}"})


def chat_request_body(messages, max_tokens):
    return {"messages": list(messages), "stream": True, "temperature": 0, "cache_prompt": False,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}


def parse_sse_events(timed_lines, request_start):
    """Fold (perf_counter arrival time, raw SSE line) pairs into a Completion."""
    text_parts, reasoning_parts = [], []
    first_token_time, last_time, timings = None, request_start, {}
    for arrival_time, raw_line in timed_lines:
        last_time = arrival_time
        if raw_line.startswith("error:"):
            raise RuntimeError(f"llama-server stream error: {raw_line}")
        if not raw_line.startswith("data:") or raw_line == "data: [DONE]":
            continue
        chunk = json.loads(raw_line.removeprefix("data:"))
        if "error" in chunk:
            raise RuntimeError(f"llama-server stream error: {chunk['error']}")
        timings = chunk.get("timings", timings)
        for choice in chunk.get("choices", ()):
            content = choice.get("delta", {}).get("content") or ""
            reasoning = choice.get("delta", {}).get("reasoning_content") or ""
            text_parts.append(content)
            reasoning_parts.append(reasoning)
            if first_token_time is None and (content or reasoning):
                first_token_time = arrival_time
    return Completion(
        text="".join(text_parts), reasoning_text="".join(reasoning_parts),
        ttft_s=None if first_token_time is None else first_token_time - request_start,
        total_s=last_time - request_start, timings=timings)


def stream_chat(base_url, messages, max_tokens, timeout_s=600):
    request_start = time.perf_counter()
    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=chat_request_body(messages, max_tokens),
                      timeout=timeout_s) as response:
        response.raise_for_status()
        timed_lines = tuple((time.perf_counter(), line) for line in response.iter_lines())
    return parse_sse_events(timed_lines, request_start)


def tokenize(base_url, text):
    response = httpx.post(f"{base_url}/tokenize", json={"content": text}, timeout=60)
    response.raise_for_status()
    return tuple(response.json()["tokens"])


def apply_template(base_url, messages):
    """The exact prompt string the server feeds the model for these chat messages."""
    response = httpx.post(f"{base_url}/apply-template", json={"messages": list(messages)}, timeout=60)
    response.raise_for_status()
    return response.json()["prompt"]
