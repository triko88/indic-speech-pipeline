from pathlib import Path

import pytest

from indic_s2t import sysinfo

FIXTURES = Path(__file__).parent / "fixtures"


def make_card(drm_root, card_name, vram_total, vram_used):
    device_dir = drm_root / card_name / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "mem_info_vram_total").write_text(f"{vram_total}\n")
    (device_dir / "mem_info_vram_used").write_text(f"{vram_used}\n")


def test_parse_list_devices_finds_both_gpus():
    list_devices_text = (FIXTURES / "list_devices.txt").read_text(encoding="utf-8")
    vulkan_devices = sysinfo.parse_vulkan_devices(list_devices_text)
    assert len(vulkan_devices) == 2
    assert {device.device_id for device in vulkan_devices} == {"Vulkan0", "Vulkan1"}


def test_pick_dgpu_selects_rx_6650m_not_igpu():
    list_devices_text = (FIXTURES / "list_devices.txt").read_text(encoding="utf-8")
    dgpu = sysinfo.pick_dgpu(sysinfo.parse_vulkan_devices(list_devices_text))
    assert "6650M" in dgpu.description
    assert "680M" not in dgpu.description


def test_pick_dgpu_raises_when_missing():
    igpu_only = (sysinfo.VulkanDevice("Vulkan0", "AMD Radeon 680M (RADV REMBRANDT)", 512),)
    with pytest.raises(RuntimeError, match="6650M"):
        sysinfo.pick_dgpu(igpu_only)


def test_dgpu_card_is_the_one_with_most_vram(tmp_path):
    make_card(tmp_path, "card1", vram_total=8573157376, vram_used=1048576 * 300)
    make_card(tmp_path, "card2", vram_total=536870912, vram_used=1048576 * 100)
    (tmp_path / "card1-eDP-1").mkdir()  # connector dirs have no device/mem_info files
    make_card(tmp_path, "renderD128", vram_total=8573157376, vram_used=1048576 * 300)  # render node aliases card1
    dgpu_card_dir = sysinfo.dgpu_card(tmp_path)
    assert dgpu_card_dir.name == "card1"
    assert sysinfo.vram_used_mib(dgpu_card_dir) == 300


def test_ac_online_reads_mains_supply(tmp_path):
    (tmp_path / "ACAD").mkdir()
    (tmp_path / "ACAD" / "online").write_text("1\n")
    (tmp_path / "BAT0").mkdir()  # batteries have no "online" file
    assert sysinfo.ac_online(tmp_path) is True
    (tmp_path / "ACAD" / "online").write_text("0\n")
    assert sysinfo.ac_online(tmp_path) is False


def test_ac_online_unknown_without_supply(tmp_path):
    assert sysinfo.ac_online(tmp_path) is None
