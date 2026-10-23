import os
import socket
import stat
import sys
from pathlib import Path

import httpx
import psutil
import pytest

from indic_s2t import llamacpp_server

FIXTURES = Path(__file__).parent / "fixtures"
GOOD_LOG = (FIXTURES / "server_log_good.txt").read_text(encoding="utf-8")
HELP_TEXT = (FIXTURES / "s3_server_help.txt").read_text(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = REPO_ROOT / "third_party/llama.cpp/build/bin/llama-server"

# Real WARN lines from llama.cpp src/llama-context.cpp (resolve_fused_ops) when an op has no backend support.
FUSED_OP_FALLBACK_LINES = (
    "0.01.779.519 W resolve_fused_ops: layer 0 is assigned to device Vulkan1 but Flash Attention "
    "is assigned to device CPU (usually due to missing support)\n"
    "0.01.779.520 W resolve_fused_ops: Flash Attention not supported, set to disabled\n")


def test_check_offload_passes_on_good_log():
    offload_report = llamacpp_server.check_offload(GOOD_LOG, "Vulkan1")
    assert "offloaded 17/17 layers" in offload_report.offload_line
    assert "using device Vulkan1" in offload_report.device_line
    assert offload_report.graph_splits == 2
    assert offload_report.warnings == ()


def test_check_offload_ignores_cpu_mapped_token_embeddings():
    assert "CPU_Mapped model buffer size" in GOOD_LOG
    llamacpp_server.check_offload(GOOD_LOG, "Vulkan1")


def test_check_offload_fails_on_partial_layer_offload():
    partial_log = GOOD_LOG.replace("offloaded 17/17 layers", "offloaded 15/17 layers")
    with pytest.raises(llamacpp_server.OffloadError, match="15/17"):
        llamacpp_server.check_offload(partial_log, "Vulkan1")


def test_check_offload_fails_on_extra_graph_splits_from_unsupported_op():
    # An op the Vulkan backend rejects runs on CPU, splitting the graph GPU->CPU->GPU in every layer.
    split_log = GOOD_LOG.replace("splits = 2,", "splits = 34 / 18,")
    with pytest.raises(llamacpp_server.OffloadError, match="splits = 34 / 18"):
        llamacpp_server.check_offload(split_log, "Vulkan1")


def test_check_offload_reports_fused_op_fallback_warnings():
    warned_log = GOOD_LOG.replace("0.01.779.519 I resolve_fused_ops: Flash Attention enabled\n", FUSED_OP_FALLBACK_LINES)
    offload_report = llamacpp_server.check_offload(warned_log, "Vulkan1")
    assert len(offload_report.warnings) == 2
    assert "usually due to missing support" in offload_report.warnings[0]


def test_check_offload_fails_on_kv_cache_on_cpu():
    cpu_kv_log = GOOD_LOG.replace("Vulkan1 KV buffer size", "CPU KV buffer size")
    with pytest.raises(llamacpp_server.OffloadError, match="CPU KV buffer"):
        llamacpp_server.check_offload(cpu_kv_log, "Vulkan1")


def test_check_offload_fails_when_device_is_igpu():
    igpu_log = GOOD_LOG.replace("using device Vulkan1 (AMD Radeon RX 6650M", "using device Vulkan0 (AMD Radeon 680M")
    with pytest.raises(llamacpp_server.OffloadError, match="Vulkan0"):
        llamacpp_server.check_offload(igpu_log, "Vulkan1")


def test_check_offload_fails_without_load_lines():
    with pytest.raises(llamacpp_server.OffloadError):
        llamacpp_server.check_offload("srv  llama_server: model loaded\n", "Vulkan1")


def test_check_offload_allows_cpu_moe_splits_when_unbounded():
    split_log = GOOD_LOG.replace("splits = 2,", "splits = 34 / 18,")
    assert llamacpp_server.check_offload(split_log, "Vulkan1", max_graph_splits=None).graph_splits == 34


@pytest.mark.parametrize("flag", ["-m", "-dev", "-ngl", "-c", "-np", "--jinja", "--reasoning-format",
                                  "--no-mmproj", "-lv", "--port", "--n-cpu-moe"])
def test_flag_supported_on_real_help(flag):
    assert llamacpp_server.flag_supported(HELP_TEXT, flag)


@pytest.mark.parametrize("flag", ["--made-up-flag", "-d", "--device-draftx"])
def test_flag_not_supported_on_real_help(flag):
    assert not llamacpp_server.flag_supported(HELP_TEXT, flag)


def test_help_lists_chosen_reasoning_format():
    assert f"- {llamacpp_server.REASONING_FORMAT}: puts thoughts in `message.reasoning_content`" in HELP_TEXT


def test_server_command_contains_contract_flags():
    command = llamacpp_server.server_command("bin", "model.gguf", "Vulkan1", 99, 4096, 8090, ("--n-cpu-moe", "8"))
    assert command[:3] == ("bin", "-m", "model.gguf")
    assert command[command.index("-dev") + 1] == "Vulkan1"
    assert command[command.index("--reasoning-format") + 1] == llamacpp_server.REASONING_FORMAT
    assert command[-2:] == ("--n-cpu-moe", "8")
    assert {"--jinja", "--no-mmproj"} <= set(command)


FAKE_SERVER_SOURCE = '''
import http.server, os, sys
from pathlib import Path
if "--help" in sys.argv:
    print(Path(os.environ["FAKE_HELP"]).read_text(encoding="utf-8"))
    sys.exit(0)
Path(os.environ["FAKE_PID_FILE"]).write_text(str(os.getpid()))
print(Path(os.environ["FAKE_LOG"]).read_text(encoding="utf-8"), flush=True)
port = int(sys.argv[sys.argv.index("--port") + 1])
class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), HealthHandler).serve_forever()
'''


def free_port():
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    fake_bin = tmp_path / "fake-llama-server"
    fake_bin.write_text(f"#!{sys.executable}\n{FAKE_SERVER_SOURCE}", encoding="utf-8")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC)
    fake_log = tmp_path / "fake_log.txt"
    fake_log.write_text(GOOD_LOG, encoding="utf-8")
    monkeypatch.setenv("FAKE_HELP", str(FIXTURES / "s3_server_help.txt"))
    monkeypatch.setenv("FAKE_LOG", str(fake_log))
    monkeypatch.setenv("FAKE_PID_FILE", str(tmp_path / "pid"))
    return fake_bin, fake_log, tmp_path / "pid"


def test_running_server_kills_process_on_exception(fake_server, tmp_path):
    fake_bin, _, pid_file = fake_server
    with pytest.raises(RuntimeError, match="boom"):
        with llamacpp_server.running_server(fake_bin, "model.gguf", "Vulkan1", port=free_port(),
                                            log_path=tmp_path / "server.log") as server_info:
            assert server_info.load_seconds > 0
            assert "17/17" in server_info.offload_line
            assert httpx.get(f"{server_info.base_url}/health").status_code == 200
            raise RuntimeError("boom")
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_running_server_raises_offload_error_and_kills(fake_server, tmp_path):
    fake_bin, fake_log, pid_file = fake_server
    fake_log.write_text(GOOD_LOG.replace("offloaded 17/17", "offloaded 9/17"), encoding="utf-8")
    with pytest.raises(llamacpp_server.OffloadError, match="9/17"):
        with llamacpp_server.running_server(fake_bin, "model.gguf", "Vulkan1", port=free_port(),
                                            log_path=tmp_path / "server.log"):
            pytest.fail("must not yield on partial offload")
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_running_server_rejects_unknown_flag_before_starting(fake_server, tmp_path):
    fake_bin, _, pid_file = fake_server
    with pytest.raises(ValueError, match="--no-such-flag"):
        with llamacpp_server.running_server(fake_bin, "model.gguf", "Vulkan1", port=free_port(),
                                            extra_args=("--no-such-flag",), log_path=tmp_path / "server.log"):
            pass
    assert not pid_file.exists()


@pytest.mark.hw
def test_real_probe_boots_on_dgpu_and_is_killed(tmp_path):
    from huggingface_hub import hf_hub_download
    from indic_s2t import llm, sysinfo
    probe_path = hf_hub_download("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    vk_device = sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(SERVER_BIN))).device_id
    with llamacpp_server.running_server(SERVER_BIN, probe_path, vk_device, port=free_port(),
                                        log_path=tmp_path / "server.log") as server_info:
        assert vk_device in server_info.device_line
        assert len(llm.tokenize(server_info.base_url, "ताजमहल आगरा में है")) > 0
        server_pids = [process.pid for process in psutil.Process(os.getpid()).children()
                       if process.name() == "llama-server"]
        assert len(server_pids) == 1
    assert not psutil.pid_exists(server_pids[0])
