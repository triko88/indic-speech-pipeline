PY := PYTHONPATH=. uv run python
PY_TTS := PYTHONPATH=. uv run --project tts python
SERVER_BIN := third_party/llama.cpp/build/bin/llama-server

.PHONY: check test test-hw requirements data tts select smoke eval report

check:
	$(PY) scripts/setup_check.py --server_bin $(SERVER_BIN)

test:
	$(PY) -m pytest -q -m "not hw"

test-hw:
	$(PY) -m pytest -q -m hw -x

# uv.lock / tts/uv.lock are the source of truth; these pinned exports exist for pip users.
requirements:
	uv export --quiet --no-hashes --no-dev -o requirements.txt
	uv export --quiet --no-hashes --project tts -o requirements-tts.txt
