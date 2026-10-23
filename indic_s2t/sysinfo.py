"""Hardware/run-environment facts, read from sysfs and tools. Logged next to every timed stage."""

import platform
import re
import resource
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DRM_ROOT = Path("/sys/class/drm")
POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")
LLAMA_CPP_DIR = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp"
DGPU_NAME = "6650M"
TRACKED_PACKAGES = ("torch", "transformers", "onnxruntime", "numpy", "datasets", "huggingface_hub", "indic-nlp-library")


@dataclass(frozen=True)
class VulkanDevice:
    device_id: str  # e.g. "Vulkan0", the value llama-server's --device takes
    description: str
    total_mib: int


def parse_vulkan_devices(list_devices_text):
    """Parse `llama-server --list-devices` output lines like `  Vulkan0: AMD Radeon RX 6650M (RADV NAVI23) (8176 MiB, 7900 MiB free)`."""
    device_pattern = re.compile(r"^\s*(Vulkan\d+): (.+?) \((\d+) MiB, \d+ MiB free\)\s*$", re.MULTILINE)
    return tuple(VulkanDevice(device_id, description, int(total_mib))
                 for device_id, description, total_mib in device_pattern.findall(list_devices_text))


def pick_dgpu(vulkan_devices, name_hint=DGPU_NAME):
    matches = [device for device in vulkan_devices if name_hint in device.description]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Vulkan device matching {name_hint!r}, got {vulkan_devices}")
    return matches[0]


def list_vulkan_devices(server_bin):
    completed = subprocess.run([server_bin, "--list-devices"], capture_output=True, text=True, timeout=60)
    return completed.stdout + completed.stderr


def dgpu_card(drm_root=DRM_ROOT):
    """The amdgpu card with the most VRAM is the dGPU (the 680M only has a small carve-out)."""
    cards = [card_dir for card_dir in drm_root.iterdir()
             if re.fullmatch(r"card\d+", card_dir.name) and (card_dir / "device" / "mem_info_vram_total").exists()]
    return max(cards, key=lambda card_dir: int((card_dir / "device" / "mem_info_vram_total").read_text()))


def vram_used_mib(card_dir):
    return int((card_dir / "device" / "mem_info_vram_used").read_text()) // (1024 * 1024)


def ac_online(power_supply_root=POWER_SUPPLY_ROOT):
    """True/False for mains power, None when no supply exposes an `online` file."""
    online_files = sorted(power_supply_root.glob("*/online")) if power_supply_root.exists() else []
    return any(online_file.read_text().strip() == "1" for online_file in online_files) if online_files else None


def read_text_or_none(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command_output_or_none(command):
    if not shutil.which(command[0]):
        return None
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return completed.stdout.strip() if completed.returncode == 0 else None


def power_state():
    # powerprofilesctl is masked on this machine; ACPI platform_profile is the real knob.
    return {
        "ac_online": ac_online(),
        "powerprofilesctl": command_output_or_none(["powerprofilesctl", "get"]),
        "platform_profile": read_text_or_none("/sys/firmware/acpi/platform_profile"),
        "cpu_governor": read_text_or_none("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_epp": read_text_or_none("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
    }


def llama_cpp_version(repo_dir=LLAMA_CPP_DIR):
    return {
        "tag": command_output_or_none(["git", "-C", str(repo_dir), "describe", "--tags", "--always"]),
        "commit": command_output_or_none(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]),
    }


def package_versions(package_names=TRACKED_PACKAGES):
    def version_or_none(package_name):
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None
    return {package_name: version_or_none(package_name) for package_name in package_names}


def peak_rss_mb(include_children=False):
    # ru_maxrss is the kernel's true peak (KiB on Linux); psutil only exposes current RSS here.
    who = resource.RUSAGE_CHILDREN if include_children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss / 1024


def snapshot():
    return {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "power": power_state(),
        "dgpu_vram_used_mib": vram_used_mib(dgpu_card()),
        "sensors": command_output_or_none(["sensors"]),
        "llama_cpp": llama_cpp_version(),
        "packages": package_versions(),
    }
