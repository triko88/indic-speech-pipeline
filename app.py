import gradio as gr

import pipeline


def process_audio(audio_path, language):
    try:
        return pipeline.run(audio_path, language)
    except ValueError as invalid_input:
        raise gr.Error(str(invalid_input))


demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
        gr.Dropdown(pipeline.SUPPORTED_LANGUAGES, value="Hindi", label="Language"),
    ],
    outputs=[gr.Textbox(label="Transcript"), gr.Textbox(label="Response")],
    title="Indic Speech Pipeline",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
