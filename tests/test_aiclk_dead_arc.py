"""A dead ARC answers `tt_aiclk` with 0xFFFFFFFF and does not raise. Every sampler drops it.

The chip still opens and runs, so a timing taken on it carries a clock field that reads
4294967295 MHz: shaped like a measurement, worth nothing (qb1 node 0, 2026-09-26). The one
decision lives in `tt_bio.runtime.aiclk_reading`, and this file checks that it is right and
that every AICLK sampler a gate or a test executes actually reaches it.
"""
from pathlib import Path

import pytest

from tt_bio import runtime
from tt_bio.train import provenance

REPO = Path(__file__).resolve().parents[1]

#: Every AICLK sampler that a gate, a training run or a test executes.
SAMPLERS = ("tt_bio/train/provenance.py", "perf/clocksample.py",
            "scripts/abb3_port/step_time.py", "perf/bcx_stack/stack.py")


def test_the_sentinel_is_not_a_clock_and_a_real_reading_passes_through():
    assert runtime.aiclk_reading(runtime.ARC_DEAD) is None
    assert runtime.ARC_DEAD == 4294967295
    # Negative control: idle, burst and a clamped clock are all real readings, unchanged.
    for mhz in (800, 1063, 1350, 500):
        assert runtime.aiclk_reading(mhz) == mhz


def _fake_sysfs(tmp_path, values):
    for node, raw in values.items():
        d = tmp_path / f"tenstorrent!{node}"
        d.mkdir()
        (d / "tt_aiclk").write_text(raw)
    return tmp_path


def test_provenance_reads_a_dead_arc_as_no_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_SYSFS",
                        _fake_sysfs(tmp_path, {0: "4294967295\n", 1: "1350\n", 2: "800\n"}))
    assert provenance.clocks() == {0: None, 1: 1350, 2: 800}


def test_a_dead_card_does_not_become_the_sampled_clock(tmp_path, monkeypatch):
    """`_Sampler` takes the max over the watched nodes, so one sentinel used to win every sample."""
    monkeypatch.setattr(provenance, "_SYSFS",
                        _fake_sysfs(tmp_path, {0: "4294967295\n", 1: "1350\n"}))
    monkeypatch.setattr(provenance, "open_nodes", lambda pid=None: [])
    s = provenance._Sampler(interval=0.01)
    s.start()
    while not s.samples:
        s._done.wait(0.01)
    assert s.stop()["max"] == 1350.0


@pytest.mark.parametrize("rel", SAMPLERS)
def test_every_sampler_reaches_the_one_decision(rel):
    src = (REPO / rel).read_text()
    assert "aiclk_reading(" in src, (
        f"{rel} samples AICLK without tt_bio.runtime.aiclk_reading, so a dead ARC reads as "
        f"a 4.3 GHz clock there")
