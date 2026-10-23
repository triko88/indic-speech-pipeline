"""Stub AI pipeline: audio -> transcript -> text response."""

SUPPORTED_LANGUAGES = ("Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia")


def transcribe(audio_path: str, language: str) -> str:
    # ponytail: stub, swap in a real ASR model (e.g. IndicConformer / Whisper)
    return f"[{language} transcript of {audio_path}]"


def respond(transcript: str, language: str) -> str:
    # ponytail: stub, swap in a real LLM call
    return f"[{language} response to: {transcript}]"


def run(audio_path: str | None, language: str) -> tuple[str, str]:
    if not audio_path:
        raise ValueError("No audio provided")
    transcript = transcribe(audio_path, language)
    return transcript, respond(transcript, language)


if __name__ == "__main__":
    sample_transcript, sample_response = run("sample.wav", "Hindi")
    assert "sample.wav" in sample_transcript and sample_transcript in sample_response
    print("ok")
