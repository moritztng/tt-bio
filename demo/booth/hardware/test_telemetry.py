"""Tests for telemetry.py against fake sysfs trees and a fake tt-smi. No device is opened.

    python3 -m pytest demo/booth/hardware/test_telemetry.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import telemetry as tm  # noqa: E402

DEAD = "4294967295\n"


def chip(root: Path, node: int, bdf: str, aiclk="800\n", hb="1000\n", card_type="p300c\n",
         power="30000000\n", temp="51559\n"):
    pci = root / "pci" / bdf
    hw = pci / "hwmon" / f"hwmon{node + 2}"
    hw.mkdir(parents=True, exist_ok=True)
    for f, v in (("power1_input", power), ("temp1_input", temp), ("in0_input", "719\n"),
                 ("curr1_input", "42000\n")):
        (hw / f).write_text(v)
    d = root / "class" / f"tenstorrent!{node}"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "device").exists():
        (d / "device").symlink_to(pci)
    for f, v in (("tt_aiclk", aiclk), ("tt_heartbeat", hb), ("tt_card_type", card_type)):
        (d / f).write_text(v)
    return d


@pytest.fixture
def box(tmp_path):
    for n in range(4):
        chip(tmp_path, n, f"0000:0{n + 1}:00.0")
    return tmp_path


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(tm.time, "time", c)
    return c


def beat(root, node, hb, aiclk="1350\n"):
    d = root / "class" / f"tenstorrent!{node}"
    (d / "tt_heartbeat").write_text(f"{hb}\n")
    (d / "tt_aiclk").write_text(aiclk)


# ---- string AICLK --------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    ("800\n", 800), (" 800", 800), ("1350", 1350), ("1350.0", 1350), (" 1350.0 ", 1350),
    ("N/A", None), ("", None), (None, None), ("-1", None), ("0x546", None),
    ("4294967295", None), ("65535", None), ("  65535", None)])
def test_reading_parses_right_aligned_strings_and_rejects_sentinels(raw, want):
    assert tm.reading(raw) == want


def test_live_chip_reports_its_numbers(box, clock):
    m = tm.Monitor(root=box / "class", events=box / "ev.jsonl")
    for i in range(3):
        for n in range(4):
            beat(box, n, 1000 + 10 * i)
        m.sample()
        clock.t += 0.25
    s = m.snapshot()
    assert [c["card"] for c in s["chips"]] == [0, 1, 2, 3]
    c0 = s["chips"][0]
    assert (c0["state"], c0["aiclk_mhz"], c0["board"], c0["power_w"], c0["temp_c"]) == \
        ("busy", 1350, "p300c", 30.0, 51.559)


# ---- dead ARC ------------------------------------------------------------------------------------
def test_dead_arc_renders_as_resetting_not_as_a_number(box, clock):
    chip(box, 2, "0000:03:00.0", aiclk=DEAD, hb=DEAD, card_type="unknown\n")
    m = tm.Monitor(root=box / "class", events=box / "ev.jsonl")
    m.sample()
    c2 = m.snapshot()["chips"][2]
    assert c2["state"] == "resetting"
    assert c2["aiclk_mhz"] is None and c2["power_w"] is None and c2["temp_c"] is None
    assert "4294967295" not in json.dumps(c2)


def test_a_clocked_up_chip_with_no_demo_fold_is_busy(box, clock):
    m = tm.Monitor(root=box / "class", events=box / "ev.jsonl")
    beat(box, 0, 1, aiclk="800\n")
    beat(box, 1, 1, aiclk="1350\n")
    m.sample()
    assert [c["state"] for c in m.snapshot()["chips"][:2]] == ["idle", "busy"]


def test_frozen_heartbeat_is_resetting_even_with_a_plausible_clock(box, clock):
    m = tm.Monitor(root=box / "class", events=box / "ev.jsonl")
    for _ in range(12):  # 3 s, heartbeat frozen at 1000 on every chip but card 0
        beat(box, 0, int(clock.t * 10))
        m.sample()
        clock.t += 0.25
    states = [c["state"] for c in m.snapshot()["chips"]]
    assert states == ["busy", "resetting", "resetting", "resetting"]


def test_a_vanished_node_is_resetting_and_comes_back(box, clock):
    m = tm.Monitor(root=box / "class", events=box / "ev.jsonl")
    m.sample()
    gone = box / "class" / "tenstorrent!3"
    os.rename(gone, box / "hidden")  # tt-smi -r: the node disappears during the board reset
    m.sample()
    assert m.snapshot()["chips"][3]["state"] == "resetting"
    os.rename(box / "hidden", gone)
    beat(box, 3, 5000)
    m.sample()
    assert m.snapshot()["chips"][3]["state"] == "busy"


def test_card_index_is_pci_order_not_node_order(tmp_path, clock):
    for node, bus in ((0, "c1"), (1, "01"), (2, "41"), (3, "81")):
        chip(tmp_path, node, f"0000:{bus}:00.0")
    assert tm.Chips(tmp_path / "class").cards() == {0: 1, 1: 2, 2: 3, 3: 0}


# ---- tt-smi fallback: string AICLK, dead ARC and no orphan -----------------------------------------
def fake_tt_smi(tmp_path, body):
    p = tmp_path / "tt-smi"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return p


def test_tt_smi_snapshot_strings_and_its_dead_arc_mask(tmp_path):
    snap = {"device_info": [{"telemetry": {"aiclk": "1350"}}, {"telemetry": {"aiclk": " 800"}},
                            {"telemetry": {"aiclk": "65535"}}, {"telemetry": {}}]}
    (tmp_path / "snap.json").write_text(json.dumps(snap))
    exe = fake_tt_smi(tmp_path, f"cat {tmp_path / 'snap.json'}\n")
    assert tm.tt_smi_aiclk(cmd=[str(exe)]) == [1350, 800, None, None]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); it is reaped, not running
    return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"


def test_a_hung_tt_smi_and_its_children_are_reaped(tmp_path):
    pids = tmp_path / "pids"
    exe = fake_tt_smi(tmp_path, f"sleep 600 & echo $! >> {pids}\necho $$ >> {pids}\n"
                                "trap '' TERM\nwhile :; do sleep 1; done\n")
    t0 = time.monotonic()
    assert tm.tt_smi_aiclk(cmd=[str(exe)], timeout=0.5) is None
    assert time.monotonic() - t0 < 6
    time.sleep(0.2)
    left = [int(p) for p in pids.read_text().split()]
    assert len(left) == 2 and not any(alive(p) for p in left), left


def test_only_one_tt_smi_runs_at_a_time(tmp_path):
    tm._TT_SMI_LOCK.acquire()
    try:
        assert tm.tt_smi_aiclk(cmd=[str(fake_tt_smi(tmp_path, "exit 1\n"))]) is None
    finally:
        tm._TT_SMI_LOCK.release()


def test_sysfs_path_never_spawns_tt_smi(box, clock, monkeypatch):
    monkeypatch.setattr(tm, "tt_smi_aiclk", lambda *a, **k: pytest.fail("spawned tt-smi"))
    tm.Monitor(root=box / "class", events=box / "ev.jsonl").sample()


# ---- folds ---------------------------------------------------------------------------------------
def test_fold_ledger_quotes_the_clock_sampled_during_the_fold(box, clock):
    ev = box / "ev.jsonl"
    m = tm.Monitor(root=box / "class", events=ev)
    m.sample()
    tm.record_fold("start", 1, path=ev, model="esmfold2", name="GFP", residues=238)
    for i, mhz in enumerate((1350, 1350, 1300, 1350)):
        clock.t += 0.25
        beat(box, 1, 2000 + i, aiclk=f"{mhz}\n")
        m.sample()
    assert m.snapshot()["chips"][1]["state"] == "folding"
    assert m.snapshot()["chips"][1]["folding"]["name"] == "GFP"
    clock.t += 0.25
    tm.record_fold("done", 1, path=ev, seconds=1.0)
    with ev.open("a") as f:
        f.write('{"t":1,"event":"start","card":1')  # half-written line: ignored until complete
    beat(box, 1, 3000)
    m.sample()
    c1 = m.snapshot()["chips"][1]
    assert c1["state"] == "busy" and c1["folds_today"] == 1
    assert c1["last_fold"]["residues_per_s"] == 238.0
    assert c1["last_fold"]["aiclk_during"]["median"] == 1350
    assert c1["last_fold"]["aiclk_during"]["min"] == 1300


def test_failed_fold_is_not_counted(box, clock):
    ev = box / "ev.jsonl"
    m = tm.Monitor(root=box / "class", events=ev)
    tm.record_fold("start", 0, path=ev, model="esmfold2", residues=100)
    tm.record_fold("fail", 0, path=ev)
    m.sample()
    assert m.snapshot()["chips"][0]["folds_today"] == 0
