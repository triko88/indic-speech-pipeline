import json
from pathlib import Path

import httpx
import pytest

from indic_s2t import llm

FIXTURES = Path(__file__).parent / "fixtures"
PASSAGE = "ताजमहल आगरा में यमुना नदी के किनारे स्थित है।"


def recorded_sse_lines():
    """s3_sse_stream.txt: `<seconds since request start>\\t<raw SSE line>`, recorded from the 1B probe."""
    fixture_lines = (FIXTURES / "s3_sse_stream.txt").read_text(encoding="utf-8").splitlines()
    return tuple((float(offset), raw_line) for offset, raw_line in (line.split("\t", 1) for line in fixture_lines))


def sse_line(delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}, ensure_ascii=False)


def test_request_body_is_deterministic_uncached_and_streaming():
    request_body = llm.chat_request_body(llm.qa_messages("प्रश्न?", PASSAGE, "hi"), max_tokens=96)
    assert request_body["stream"] is True
    assert request_body["temperature"] == 0
    assert request_body["cache_prompt"] is False
    assert request_body["max_tokens"] == 96
    assert request_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert request_body["stream_options"] == {"include_usage": True}
    json.dumps(request_body, ensure_ascii=False)


def test_oracle_and_asr_prompts_differ_only_in_question():
    oracle_messages = llm.qa_messages("ताजमहल किस शहर में है?", PASSAGE, "hi")
    asr_messages = llm.qa_messages("ताज महल किस सहर में है", PASSAGE, "hi")
    assert oracle_messages[0] == asr_messages[0]
    assert oracle_messages[1]["content"].replace("ताजमहल किस शहर में है?", "") == \
        asr_messages[1]["content"].replace("ताज महल किस सहर में है", "")


def test_system_prompt_requires_no_answer_and_language():
    system_prompt = llm.qa_messages("Where is it?", "A passage.", "en")[0]["content"]
    assert "exactly NO_ANSWER" in system_prompt
    assert "ONLY" in system_prompt
    assert "English" in system_prompt
    assert "speech recognition" in system_prompt


def test_asr_note_is_optional():
    plain_prompt = llm.qa_messages("प्रश्न?", PASSAGE, "hi", asr_aware=False)[0]["content"]
    assert "speech recognition" not in plain_prompt
    assert "Hindi" in plain_prompt


def test_messages_are_immutable_tuple_with_passage_and_question():
    messages = llm.qa_messages("प्रश्न?", PASSAGE, "hi")
    assert isinstance(messages, tuple)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert PASSAGE in messages[1]["content"] and "प्रश्न?" in messages[1]["content"]


def test_parse_recorded_probe_stream():
    completion = llm.parse_sse_events(recorded_sse_lines(), request_start=0.0)
    assert completion.text == "NO_ANSWER"
    assert completion.reasoning_text == ""
    assert completion.ttft_s == pytest.approx(0.234261)  # first chunk is role-only with null content
    assert completion.total_s == pytest.approx(0.248821)
    assert completion.timings["prompt_n"] == 102
    assert completion.timings["predicted_n"] == 4
    assert completion.timings["predicted_per_second"] == pytest.approx(206.384, rel=1e-3)


def test_parse_collects_reasoning_separately_and_counts_it_for_ttft():
    timed_lines = (
        (10.5, sse_line({"role": "assistant", "content": None})),
        (10.6, sse_line({"reasoning_content": "सोच"})),
        (10.7, sse_line({"reasoning_content": "रहा"})),
        (10.9, sse_line({"content": "आगरा"})),
        (11.0, 'data: {"choices":[],"timings":{"prompt_n":5,"predicted_n":3}}'),
        (11.1, "data: [DONE]"),
    )
    completion = llm.parse_sse_events(timed_lines, request_start=10.0)
    assert completion.text == "आगरा"
    assert completion.reasoning_text == "सोचरहा"
    assert completion.ttft_s == pytest.approx(0.6)
    assert completion.total_s == pytest.approx(1.1)
    assert completion.timings == {"prompt_n": 5, "predicted_n": 3}


def test_parse_raises_on_server_error_event():
    with pytest.raises(RuntimeError, match="context"):
        llm.parse_sse_events(((1.0, 'error: {"message":"context size exceeded"}'),), request_start=0.0)


def test_parse_raises_on_error_payload():
    with pytest.raises(RuntimeError, match="exceeds"):
        llm.parse_sse_events(((1.0, 'data: {"error":{"message":"prompt exceeds context"}}'),), request_start=0.0)


def test_tokenize_returns_tuple_from_recorded_response(monkeypatch):
    recorded_body = (FIXTURES / "s3_tokenize_response.json").read_text(encoding="utf-8")

    def fake_post(url, json, timeout):
        assert url == "http://server/tokenize" and json == {"content": "ताजमहल आगरा में है"}
        return httpx.Response(200, text=recorded_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm.tokenize("http://server", "ताजमहल आगरा में है") == (
        80338, 100675, 124794, 92911, 100320, 103210, 24810, 92317, 100271, 85410, 100266)
