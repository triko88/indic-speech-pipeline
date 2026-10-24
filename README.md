# indic-speech-pipeline

Spoken question (Hindi, English or Sanskrit) → text answer, built and evaluated on one laptop: IndicConformer-600M
(ONNX Runtime) or Whisper ASR on the CPU, a 4-bit LLM in `llama-server` on the RX 6650M over Vulkan, and a judge LLM from
another model family.

## Languages

Every language-specific value lives in `languages.toml`, one table per language: its name, Unicode block, ASR model,
data sources, abstain phrases and TTS voices. Add a table to try a language, comment one out to drop it, then
`make data`. The current set:

| Lang | ASR model | ASR track | QA track |
|---|---|---|---|
| hi | IndicConformer-600M | Kathbath test_unknown | IndicQA |
| en | openai/whisper-small | FLEURS en_us test | SQuAD v2 dev |
| sa | IndicConformer-600M | Kathbath test_unknown | none yet (ASR only; add a `squad_url`) |

## System

```
 wav (16 kHz mono)
   │   acoustic condition, on the fly: clean | babble_10db | telephony | white_{snr}db, chained with "+"
   ▼
 IndicConformer-600M  ── ONNX Runtime, CPU, 8 intra-op threads, RNNT (or CTC) decoding     [process 1: ASR]
   or Whisper         ── transformers, CPU, greedy (per language's asr_model)
   │ transcript
   ▼
 llama-server (llama.cpp, Vulkan, RX 6650M, all layers offloaded, 1 slot, ctx 4096)       [process 2: LLM]
   │ extractive QA prompt (passage + "may contain ASR errors"), temperature 0, thinking off
   ▼
 answer ──► rule labels (CORRECT / CORRECT_ABSTAIN / FALSE_ABSTAIN / WRONG_GROUNDED / HALLUCINATED)
        ──► judge LLM, other model family, same labels                                     [process 3: judge]
        ──► blind human audit sheet (40 rows)
```

Each stage runs in its own process and exits before the next starts, so only one model is ever loaded. The gold-text
`oracle` run of every QA item separates ASR-induced failures from LM failures. Linguistic conditions skip ASR the same
way: the gold question is perturbed syntactically (`scramble`, `drop_head`, `drop_tail`) or semantically
(`swap_anchor`, `wrong_passage`, both unanswerable by construction) and scored against `oracle`
(`indic_s2t/perturb.py`; `--text_conditions` with no names turns them off).

## Setup (Arch / Omarchy)

```sh
sudo pacman -S --needed base-devel git cmake ninja vulkan-radeon vulkan-icd-loader vulkan-headers vulkan-tools \
    shaderc ffmpeg uv
git submodule update --init                       # llama.cpp, pinned to release tag b11844
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build --target llama-server
uv sync                                           # main env: Python 3.12, CPU-only torch
uv sync --project tts                             # TTS env: Python 3.11, parler-tts
uv run hf auth login                              # accept the gates for ai4bharat/indic-conformer-600m-multilingual and ai4bharat/Kathbath first
make check                                        # Vulkan devices, dGPU proof, HF access, disk, RAM, swap, power
```

`uv.lock` and `tts/uv.lock` pin everything; `make requirements` exports `requirements.txt` / `requirements-tts.txt`.

## Run

```sh
make test        # unit tests (no models)
make test-hw     # model/GPU tests, one process per file
make data        # ASR models, ASR track + babble (Kathbath/FLEURS), QA questions (IndicQA/SQuAD), examples
make tts         # Parler-TTS audio for the QA questions (resumable, prints ETA)
make select      # LLM candidates A and B, one at a time -> runs/llm_selection.md
make smoke       # 5 items per track x language, clean only
make eval ANSWERER=A JUDGE=B
make variance ANSWERER=A JUDGE=B
make report      # runs/main/report.md + runs/main/audit_sheet.csv
PYTHONPATH=. uv run python demo.py --wav question.wav --lang hi --gguf <model.gguf> [--context passage.txt]
PYTHONPATH=. uv run python app.py --gguf <model.gguf>
```

## Metric definitions

See the "Metric definitions" section of `runs/main/report.md`, generated from `scripts/make_report.py`
(`METRIC_DEFINITIONS`); it is copied here once the results are in.

## Results

Pending the full evaluation.

## Findings and key areas of improvement

Written after the results.
