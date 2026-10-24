import numpy as np
import soundfile

import demo
from indic_s2t import llm


def fake_stream_chat(recorded_messages):
    def stream_chat(base_url, messages, max_tokens):
        recorded_messages.append(messages)
        return llm.Completion(text="आगरा", reasoning_text="", ttft_s=0.25, total_s=0.75, timings={})
    return stream_chat


def test_answer_wav_uses_passage_prompt_and_sums_stage_times(tmp_path, monkeypatch):
    wav_path = tmp_path / "question.wav"
    soundfile.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)
    recorded_messages = []
    monkeypatch.setattr(llm, "stream_chat", fake_stream_chat(recorded_messages))
    monkeypatch.setattr("indic_s2t.asr.transcribe", lambda model, wave, lang: "ताजमहल कहाँ है")

    answer = demo.answer_wav(asr_model=None, base_url="http://x", wav_path=wav_path, lang="hi", context="ताजमहल आगरा में है")
    assert answer["transcript"] == "ताजमहल कहाँ है" and answer["response"] == "आगरा"
    assert answer["audio_s"] == 1.0
    assert answer["e2e_s"] == answer["asr_s"] + 0.75
    assert answer["speech_end_to_first_token_s"] == answer["asr_s"] + 0.25
    assert llm.NO_ANSWER in recorded_messages[0][0]["content"]  # passage mode: extractive prompt

    demo.answer_wav(asr_model=None, base_url="http://x", wav_path=wav_path, lang="sa")
    assert llm.NO_ANSWER not in recorded_messages[1][0]["content"]  # no passage: open question prompt
    assert "Sanskrit" in recorded_messages[1][0]["content"]
