"""perf/pnr/cells_to_page.py puts only protocol-clean cells on the page, and the guard refuses its fake fixture."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "perf" / "pnr"))
import cells_to_page as c2p  # noqa: E402


def test_fake_fixture_builds_every_row_and_the_guard_refuses_it(tmp_path):
    out = tmp_path / "perf.json"
    subprocess.run([sys.executable, str(REPO / "perf/pnr/cells_to_page.py"), "--fake", "--out", str(out)],
                   check=True, capture_output=True)
    d = json.loads(out.read_text())
    assert "mol1" in [m["id"] for m in d["models"]]
    assert {"bindcraft2", "rfd4"} <= {m["id"] for m in d["design"]["models"]}
    g = subprocess.run([sys.executable, str(REPO / "scripts/site_publish_guard.py"), "--data", str(out)],
                       capture_output=True, text=True)
    assert g.returncode == 1 and "fake render fixture" in g.stderr


def test_fake_never_writes_the_published_file():
    r = subprocess.run([sys.executable, str(REPO / "perf/pnr/cells_to_page.py"), "--fake"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "never writes" in r.stderr


@pytest.mark.parametrize("break_it, msg", [
    (lambda tt, gpu: tt["mol1"].update(aiclk_min=1100), "void"),
    (lambda tt, gpu: tt["bindcraft2"].update(n=4), "n=4"),
    (lambda tt, gpu: tt["rfd4"].pop("batch"), "batch"),
    (lambda tt, gpu: gpu["b200"]["rfd4"]["arms"].update(
        quicker=dict(gpu["b200"]["rfd4"]["arms"]["fake-arm"], s_per_design=0.1)), "slower than quicker"),
])
def test_a_cell_off_protocol_is_refused(break_it, msg):
    tt, gpu = c2p.fake_cells()
    tt, gpu = copy.deepcopy(tt), copy.deepcopy(gpu)
    break_it(tt, gpu)
    with pytest.raises(SystemExit, match=msg):
        c2p.build(tt, gpu, {})


def test_a_faster_screen_rung_below_n5_does_not_outrank_the_chosen_arm():
    tt, gpu = copy.deepcopy(c2p.fake_cells())
    arms = gpu["b200"]["rfd4"]["arms"]
    arms["rung"] = dict(arms["fake-arm"], s_per_design=0.1, n=1)
    c2p.build(tt, gpu, {})


def test_rfd4_cells_carry_each_sides_batch():
    tt, gpu = c2p.fake_cells()
    rows = dict((r["id"], r) for _, r in c2p.build(tt, gpu, {}))
    assert {k: c["batch"] for k, c in rows["rfd4"]["cells"].items()} == \
        {"p150a": 20, "h200": 256, "b200": 256, "a100": 256}
    assert all(c["ref"] == "stock" for k, c in rows["mol1"]["cells"].items() if k != "p150a")


def test_a_row_subset_places_only_those_rows_and_prose_names_only_them(tmp_path):
    out = tmp_path / "perf.json"
    subprocess.run([sys.executable, str(REPO / "perf/pnr/cells_to_page.py"), "--fake", "--rows", "rfd4",
                    "--out", str(out)], check=True, capture_output=True)
    d = json.loads(out.read_text())
    ids = {m["id"] for m in d["design"]["models"]}
    assert "rfd4" in ids and "bindcraft2" not in ids and "mol1" not in [m["id"] for m in d["models"]]
    prose = " ".join(d["design"][k] for k in ("cond", "note", "methods"))
    assert "RFD4-Proteina" in prose and "BindCraft" not in prose
    assert "RFD4-Proteina" in d["scope"]["timed_region"]
