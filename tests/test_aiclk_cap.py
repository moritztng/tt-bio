"""A chip listed as unable to hold its top clock is forced down before ttnn opens it.

qb2 chip 3 (ASIC 6DC6A57C892132D6) deadlocks a Protenix-v2 matmul within one call at 1350 MHz and runs clean
forced lower (spd-nocdeadlock2, 2026-10-14).
"""
import sys
import types

import pytest

from tt_bio import runtime

LIST = """# <asic id> <MHz> <reason>
6dc6a57c892132d6 1200 qb2 chip 3: PV2 pair-cond matmul deadlocks at 1350
"""
NODES = {0: 0, 3: 3}
ASIC = {0: "D7ADCC7E44A5908A", 3: "6DC6A57C892132D6"}


@pytest.fixture
def caps(tmp_path, monkeypatch):
    path = tmp_path / "AICLK_CAP"
    path.write_text(LIST)
    monkeypatch.setattr(runtime, "umd_index_to_dev_node", lambda: dict(NODES))
    monkeypatch.setattr(runtime, "tt_asic_id", ASIC.get)
    sent = []

    class PciChip:
        def __init__(self, pci_interface):
            self.node = pci_interface

        def arc_msg(self, msg, wait_for_done, arg0, arg1):
            sent.append((self.node, msg, arg0))
            return (0, 0)

    monkeypatch.setitem(sys.modules, "pyluwen", types.SimpleNamespace(PciChip=PciChip))
    return str(path), sent


def test_the_list_is_keyed_by_asic_id_case_insensitively(caps):
    assert runtime.aiclk_caps(caps[0]) == {"6DC6A57C892132D6": (1200, "qb2 chip 3: PV2 pair-cond matmul deadlocks at 1350")}


def test_only_the_listed_chip_is_forced(caps):
    path, sent = caps
    assert runtime.apply_aiclk_caps([0, 3], path) == {3: 1200}
    assert sent == [(3, 0x33, 1200)]


def test_no_list_touches_nothing(tmp_path, caps):
    assert runtime.apply_aiclk_caps([0, 3], str(tmp_path / "none")) == {}
    assert caps[1] == []


def test_a_listed_chip_that_cannot_be_capped_is_refused(caps, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyluwen", None)  # import fails
    with pytest.raises(RuntimeError, match="could not be capped"):
        runtime.apply_aiclk_caps([3], caps[0])
