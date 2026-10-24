"""Gradio UI over demo.answer_wav: ASR and llama-server are started once and stopped at exit.

    PYTHONPATH=. uv run python app.py --gguf model.gguf   # http://127.0.0.1:7860
"""

import argparse
import atexit
import os
from contextlib import ExitStack

import gradio as gr

import demo
from indic_s2t.languages import LANGUAGES


def build_interface(asr_models, base_url):
    """asr_models: model id -> loaded model, covering every language's `asr_model`."""
    def process_audio(audio_path, language, context):
        if not audio_path:
            raise gr.Error("No audio provided")
        answer = demo.answer_wav(asr_models[LANGUAGES[language].asr_model], base_url, audio_path, language,
                                 context.strip() or None)
        timing = (f"ASR {answer['asr_s']:.2f} s · TTFT {answer['ttft_s']:.2f} s · "
                  f"speech-end → first token {answer['speech_end_to_first_token_s']:.2f} s · E2E {answer['e2e_s']:.2f} s")
        return answer["transcript"], answer["response"], timing

    return gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
            gr.Dropdown([(language.name, code) for code, language in LANGUAGES.items()], value=next(iter(LANGUAGES)),
                        label="Language"),
            gr.Textbox(label="Passage (optional: answer only from this text)", lines=4),
        ],
        outputs=[gr.Textbox(label="Transcript"), gr.Textbox(label="Response"), gr.Textbox(label="Timings")],
        title="Indic Speech Pipeline",
        flagging_mode="never",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--server_bin", default=demo.SERVER_BIN)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from indic_s2t.asr import load_asr_models
    from indic_s2t.llamacpp_server import running_server
    asr_models = {model_id: model for model_id, (model, _) in
                  load_asr_models(tuple(language.asr_model for language in LANGUAGES.values())).items()}
    resources = ExitStack()
    atexit.register(resources.close)  # always stop llama-server, even on Ctrl-C
    server_info = resources.enter_context(running_server(args.server_bin, args.gguf, demo.dgpu_device(args.server_bin)))
    build_interface(asr_models, server_info.base_url).launch()


if __name__ == "__main__":
    main()
