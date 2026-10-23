"""Verify the machine before anything heavy runs. Exits non-zero if any check FAILs (WARN does not fail)."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import psutil

from indic_s2t import sysinfo

GATED_REPOS = (("ai4bharat/indic-conformer-600m-multilingual", "model"), ("ai4bharat/Kathbath", "dataset"))
OPEN_REPOS = (("ai4bharat/indic-parler-tts", "model"), ("ai4bharat/IndicQA", "dataset"))
PROBE_GGUF = ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
MIN_FREE_DISK_GB = 25
MIN_AVAILABLE_RAM_GB = 8


def check_vulkan():
    summary = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True).stdout
    gpu_blocks = re.findall(r"deviceName\s*=\s*(.+)\n(?:.*\n)*?\s*driverName\s*=\s*(\S+)", summary)
    detail = "; ".join(f"{device_name.strip()} [{driver_name}]" for device_name, driver_name in gpu_blocks)
    amdvlk_icds = list(Path("/usr/share/vulkan/icd.d").glob("amd_icd*.json"))
    if amdvlk_icds:
        return "FAIL", f"{detail}; AMDVLK also installed ({amdvlk_icds}), conflicts with RADV"
    all_radv = len(gpu_blocks) >= 2 and all(driver_name == "radv" for _, driver_name in gpu_blocks)
    has_dgpu = any(sysinfo.DGPU_NAME in device_name for device_name, _ in gpu_blocks)
    return ("OK" if all_radv and has_dgpu else "FAIL"), detail


def check_tools():
    missing_tools = [tool for tool in ("ffmpeg", "glslc", "git", "cmake") if not shutil.which(tool)]
    return ("FAIL", f"missing: {missing_tools}") if missing_tools else ("OK", "ffmpeg glslc git cmake")


def check_llama_cpp(server_bin):
    if not Path(server_bin).exists():
        return "FAIL", f"{server_bin} not built"
    dgpu = sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(server_bin)))
    version_info = sysinfo.llama_cpp_version()
    return "OK", f"{version_info['tag']} ({version_info['commit'][:10]}); dGPU = {dgpu.device_id}: {dgpu.description}"


def probe_dgpu_offload(server_bin, port=8091):
    """Boot a 1B GGUF on the dGPU and prove it: log names the device, all layers offloaded, VRAM on the dGPU card rises."""
    from huggingface_hub import hf_hub_download
    probe_path = hf_hub_download(*PROBE_GGUF)
    dgpu = sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(sysinfo.list_vulkan_devices(server_bin)))
    card_dir = sysinfo.dgpu_card()
    vram_before_mib = sysinfo.vram_used_mib(card_dir)
    with tempfile.NamedTemporaryFile("w+", suffix=".log", encoding="utf-8") as server_log:
        server_process = subprocess.Popen(
            [server_bin, "-m", probe_path, "-dev", dgpu.device_id, "-ngl", "99", "-c", "2048", "-np", "1",
             "--no-mmproj", "-lv", "4", "--port", str(port)],
            stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and server_process.poll() is None:
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            vram_loaded_mib = sysinfo.vram_used_mib(card_dir)
        finally:
            server_process.terminate()
            server_process.wait(timeout=30)
        log_text = Path(server_log.name).read_text(encoding="utf-8", errors="replace")
    proof_lines = [line.strip() for line in log_text.splitlines()
                   if re.search(r"offloaded \d+/\d+ layers|using device|Vulkan\d.*model buffer size", line)]
    offload_match = re.search(r"offloaded (\d+)/(\d+) layers", log_text)
    fully_offloaded = bool(offload_match) and offload_match.group(1) == offload_match.group(2)
    vram_delta_mib = vram_loaded_mib - vram_before_mib
    status = "OK" if fully_offloaded and dgpu.device_id in log_text and vram_delta_mib > 300 else "FAIL"
    return status, f"{card_dir.name} VRAM +{vram_delta_mib} MiB | " + " | ".join(proof_lines)


def check_hf_access():
    from huggingface_hub import auth_check, whoami
    try:
        user_name = whoami()["name"]
    except Exception:
        return "FAIL", "not logged in: run `.venv/bin/hf auth login`"
    problems = []
    for repo_id, repo_type in GATED_REPOS + OPEN_REPOS:
        try:
            auth_check(repo_id, repo_type=repo_type)
        except Exception as access_error:
            problems.append(f"{repo_id}: {type(access_error).__name__} (accept the gate at https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id})")
    return ("FAIL", f"user {user_name}; " + "; ".join(problems)) if problems else ("OK", f"user {user_name}; all repos accessible")


def check_disk():
    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    return ("OK" if free_gb >= MIN_FREE_DISK_GB else "FAIL"), f"{free_gb:.0f} GB free (need {MIN_FREE_DISK_GB}; 45 for Sarvam-30B)"


def check_memory():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    swap_devices = Path("/proc/swaps").read_text().splitlines()[1:]
    detail = (f"RAM {memory.total / 2**30:.1f} GiB, available {memory.available / 2**30:.1f} GiB; "
              f"swap {swap.total / 2**30:.1f} GiB ({', '.join(line.split()[0] for line in swap_devices) or 'none'})")
    if not swap_devices:
        return "WARN", detail + "; no swap/zram"
    return ("OK" if memory.available / 2**30 >= MIN_AVAILABLE_RAM_GB else "WARN"), detail


def check_power():
    power = sysinfo.power_state()
    return ("OK" if power["ac_online"] else "WARN"), str(power)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_bin", default="third_party/llama.cpp/build/bin/llama-server")
    parser.add_argument("--skip_probe", action="store_true", help="skip booting the 1B probe model on the dGPU")
    args = parser.parse_args()
    checks = [("vulkan", check_vulkan), ("tools", check_tools), ("llama.cpp", lambda: check_llama_cpp(args.server_bin)),
              ("dgpu_probe", (lambda: ("SKIP", "--skip_probe")) if args.skip_probe else (lambda: probe_dgpu_offload(args.server_bin))),
              ("hf_access", check_hf_access), ("disk", check_disk), ("memory", check_memory), ("power", check_power)]
    statuses = []
    for check_name, check_function in checks:
        try:
            status, detail = check_function()
        except Exception as check_error:
            status, detail = "FAIL", f"{type(check_error).__name__}: {check_error}"
        statuses.append(status)
        print(f"[{status:4}] {check_name:11} {detail}", flush=True)
    sys.exit(1 if "FAIL" in statuses else 0)


if __name__ == "__main__":
    main()
