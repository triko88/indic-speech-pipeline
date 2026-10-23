PY := PYTHONPATH=. uv run python
PY_TTS := PYTHONPATH=. uv run --project tts python
SERVER_BIN := third_party/llama.cpp/build/bin/llama-server

.PHONY: check test test-hw requirements data tts select smoke eval report

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
