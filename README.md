# indic-speech-pipeline
A simple pipeline to process audio in Indic languages and generate text response

## Run

```sh
uv sync
uv run app.py   # http://127.0.0.1:7860
```

`pipeline.py` holds the stub backend (`transcribe` → `respond`); `uv run pipeline.py` runs its self-check.
