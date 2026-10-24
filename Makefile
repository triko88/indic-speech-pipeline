PY := PYTHONPATH=. uv run python
PY_TTS := PYTHONPATH=. uv run --project tts python
SERVER_BIN := third_party/llama.cpp/build/bin/llama-server

.PHONY: check test test-hw requirements data tts select smoke eval variance report

check:
	$(PY) scripts/setup_check.py --server_bin $(SERVER_BIN)

test:
	$(PY) -m pytest -q -m "not hw"

# One process per file, so a model loaded by one test file is gone before the next boots another.
test-hw:
	for test_file in $$(grep -l "mark.hw" tests/*.py); do HF_HUB_OFFLINE=1 $(PY) -m pytest -q -m hw -x $$test_file || exit 1; done

# uv.lock / tts/uv.lock are the source of truth; these pinned exports exist for pip users.
requirements:
	uv export --quiet --no-hashes --no-dev -o requirements.txt
	uv export --quiet --no-hashes --project tts -o requirements-tts.txt

data:
	$(PY) scripts/prepare_data.py --stage asr
	$(PY) scripts/prepare_data.py --stage qa
	$(PY) scripts/prepare_data.py --stage examples

tts:
	mkdir -p runs
	$(PY_TTS) scripts/prepare_data.py --stage tts 2>&1 | tee -a runs/tts.log

# Candidate GGUFs (downloaded/cached on first use). Text only: the mmproj files are never fetched or passed.
GGUF_A = $(shell uv run hf download google/gemma-4-E4B-it-qat-q4_0-gguf gemma-4-E4B_q4_0-it.gguf -q)
GGUF_B = $(shell uv run hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf -q)
FAMILY_A := gemma
FAMILY_B := qwen
ANSWERER ?= A
JUDGE ?= B
EVAL_ARGS = --llm_gguf $(GGUF_$(ANSWERER)) --judge_gguf $(GGUF_$(JUDGE)) --server_bin $(SERVER_BIN)

select:
	$(PY) scripts/select_llm.py --name A --family $(FAMILY_A) --gguf $(GGUF_A) --server_bin $(SERVER_BIN)
	$(PY) scripts/select_llm.py --name B --family $(FAMILY_B) --gguf $(GGUF_B) --server_bin $(SERVER_BIN)
	$(PY) scripts/select_llm.py --summarize

smoke:
	$(PY) scripts/run_eval.py $(EVAL_ARGS) --limit 5 --conditions clean --out runs/smoke

eval:
	$(PY) scripts/run_eval.py $(EVAL_ARGS) --out runs/main

# Same 20-item clean subset (5 per track x language), three fresh runs.
variance:
	for run_index in 1 2 3; do $(PY) scripts/run_eval.py $(EVAL_ARGS) --stages asr llm --limit 5 --conditions clean --out runs/var_$$run_index --force || exit 1; done

report:
	$(PY) scripts/make_report.py --run runs/main --variance runs/var_1 runs/var_2 runs/var_3
