"""Run llama-server pinned to one Vulkan device and prove from its log that the whole model runs there."""

import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from indic_s2t import sysinfo

# deepseek = thoughts extracted into `reasoning_content`; `none` would leave them inline in `content`.
REASONING_FORMAT = "deepseek"
CPU_MOE_FLAGS = ("-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe")
# Full offload = 1 CPU split (token-embedding get_rows from CPU_Mapped) + 1 Vulkan split. An op the backend
# rejects (ggml_backend_vk_device_supports_op) is scheduled on CPU and adds splits in every layer.
MAX_GRAPH_SPLITS = 2
LOG_TAIL_LINES = 40


class OffloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class OffloadReport:
    offload_line: str
    device_line: str
    graph_splits: int
    warnings: tuple[str, ...]  # fused ops auto-disabled for lack of support (still run unfused on the GPU)


@dataclass(frozen=True)
class ServerInfo:
    base_url: str
    load_seconds: float
    offload_line: str
    device_line: str
    log_path: Path
    vram_used_mib: int


def matching_lines(pattern, log_text):
    return tuple(line.strip() for line in log_text.splitlines() if re.search(pattern, line))


def check_offload(log_text, vk_device, max_graph_splits=MAX_GRAPH_SPLITS):
    """Pure check of a `-lv 4` startup log. Raises OffloadError listing every problem line."""
    problems = []
    offload_lines = matching_lines(r"offloaded \d+/\d+ layers to GPU", log_text)
    offload_counts = [re.search(r"offloaded (\d+)/(\d+)", line).groups() for line in offload_lines]
    if not offload_lines:
        problems.append("no `offloaded N/N layers to GPU` line (was the server started with -lv 4?)")
    problems += [line for line, (offloaded, total) in zip(offload_lines, offload_counts) if offloaded != total]

    device_lines = matching_lines(r"using device \S+", log_text)
    if not device_lines:
        problems.append("no `using device` line")
    problems += [line for line in device_lines if not re.search(rf"using device {re.escape(vk_device)}\b", line)]

    # llama-context.cpp: "graph: nodes = X, splits = PP" or "graph (pp bs=.., tg bs=..): ... splits = PP / TG"
    split_lines = matching_lines(r"graph(?: \([^)]*\))?: nodes = .*splits = ", log_text)
    split_counts = [int(count) for line in split_lines
                    for count in re.search(r"splits = (\d+(?: / \d+)?)", line).group(1).split(" / ")]
    if not split_lines:
        problems.append("no `graph: ... splits = N` line")
    elif max_graph_splits is not None and max(split_counts) > max_graph_splits:
        problems += [f"{line}  (> {max_graph_splits} splits: some ops run on CPU)" for line in split_lines]

    problems += matching_lines(r"\bCPU KV buffer size", log_text)
    if problems:
        raise OffloadError(f"model not fully on {vk_device}:\n" + "\n".join(problems))
    return OffloadReport(
        offload_line=offload_lines[-1], device_line=device_lines[-1], graph_splits=max(split_counts),
        warnings=matching_lines(r"usually due to missing support|not supported, set to disabled", log_text))


def flag_supported(help_text, flag):
    """True if `flag` is declared on an option line (column 0) of `llama-server --help`."""
    declared_flags = {declared for line in help_text.splitlines() if line.startswith("-")
                      for declared in re.findall(r"(?<![\w-])(-{1,2}[A-Za-z][\w-]*)", line)}
    return flag in declared_flags


def server_command(server_bin, gguf, vk_device, ngl, ctx, port, extra_args):
    return (str(server_bin), "-m", str(gguf), "-dev", vk_device, "-ngl", str(ngl), "-c", str(ctx), "-np", "1",
            "--jinja", "--reasoning-format", REASONING_FORMAT, "--no-mmproj", "--cache-ram", "0",  # 8 GB host cache otherwise
            "-lv", "4", "--port", str(port), *extra_args)


def log_tail(log_path):
    return "\n".join(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:])


def wait_until_healthy(server_process, base_url, log_path, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {server_process.returncode}:\n{log_tail(log_path)}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"llama-server not healthy after {timeout_s} s:\n{log_tail(log_path)}")


@contextmanager
def running_server(server_bin, gguf, vk_device, ngl=99, ctx=4096, port=8090, extra_args=(), log_path=None,
                   ready_timeout_s=300, max_graph_splits=MAX_GRAPH_SPLITS):
    command = server_command(server_bin, gguf, vk_device, ngl, ctx, port, tuple(extra_args))
    help_text = subprocess.run([str(server_bin), "--help"], capture_output=True, text=True, timeout=60).stdout
    unsupported_flags = [token for token in command[1:]
                         if re.match(r"-{1,2}[A-Za-z]", token) and not flag_supported(help_text, token)]
    if unsupported_flags:
        raise ValueError(f"{server_bin} does not accept {unsupported_flags}")
    if any(flag in command for flag in CPU_MOE_FLAGS):
        max_graph_splits = None  # experts deliberately on CPU: splits are expected
    log_path = Path(log_path or tempfile.mkstemp(prefix="llama-server-", suffix=".log")[1])
    base_url = f"http://127.0.0.1:{port}"
    with log_path.open("w", encoding="utf-8") as server_log:
        load_start = time.perf_counter()
        server_process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            wait_until_healthy(server_process, base_url, log_path, ready_timeout_s)
            load_seconds = time.perf_counter() - load_start
            offload_report = check_offload(log_path.read_text(encoding="utf-8", errors="replace"), vk_device,
                                           max_graph_splits)
            yield ServerInfo(base_url, load_seconds, offload_report.offload_line, offload_report.device_line,
                             log_path, sysinfo.vram_used_mib(sysinfo.dgpu_card()))
        finally:
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
