"""A dead ARC must not be bankable as a clock.

On 2026-09-26 `bcx-p10-devgap` ran a five-minute anchor on qb1 card 3 and wrote a
`round_events.json` in which every AICLK sample was 4294967295. Nothing raised, nothing
warned, and the file named its card correctly. The chip was fine four hours later after one
reset; the artifact was not.

Everything here runs off a fake sysfs tree, so it needs no card, no dead card and no host in
particular, and it runs in CI.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tt_bio import aiclk  # noqa: E402

SENTINEL = 4294967295
HELPER = ROOT / "perf" / "lib" / "aiclk.sh"


@pytest.fixture
def sysfs(tmp_path):
    """A `/sys/class/tenstorrent` with node 0's ARC dead and node 1 healthy."""
    for node, value in ((0, SENTINEL), (1, 1350)):
        d = tmp_path / f"tenstorrent!{node}"
        d.mkdir()
        (d / "tt_aiclk").write_text(f"{value}\n")
        # the same dead ARC answers these with the same sentinel
        (d / "tt_heartbeat").write_text(f"{value if node == 0 else 4211}\n")
    return tmp_path


def test_sentinel_is_the_value_the_dead_card_reported():
    assert aiclk.ARC_DEAD == SENTINEL
    assert not aiclk.sane(SENTINEL)


@pytest.mark.parametrize("raw,want", [
    ("4294967295\n", None),    # the dead ARC
    ("1350\n", 1350),
    ("800", 800),
    ("0\n", None),             # a clock of zero is not a clock either
    ("4294967295 0 0\n", None),
    ("", None),
    ("not a number", None),
    (f"{aiclk.MAX_PLAUSIBLE_MHZ + 1}\n", None),
    (f"{aiclk.MAX_PLAUSIBLE_MHZ}\n", aiclk.MAX_PLAUSIBLE_MHZ),
])
def test_parse(raw, want):
    assert aiclk.parse(raw) == want


def test_read_is_none_on_the_sentinel_and_the_number_on_a_live_node(sysfs):
    assert aiclk.read(0, sysfs=sysfs) is None
    assert aiclk.read(1, sysfs=sysfs) == 1350
    assert aiclk.read(7, sysfs=sysfs) is None      # no such node


def test_require_refuses_the_sentinel_and_names_it(sysfs):
    with pytest.raises(aiclk.DeadARC) as exc:
        aiclk.require(0, sysfs=sysfs)
    assert "ARC is dead" in str(exc.value)
    assert aiclk.require(1, sysfs=sysfs) == 1350


def test_require_refuses_an_absent_node(sysfs):
    with pytest.raises(aiclk.DeadARC):
        aiclk.require(7, sysfs=sysfs)


def test_the_sentinel_check_transfers_to_heartbeat_but_the_mhz_bound_does_not(sysfs):
    """What the sibling attributes share is the sentinel, not AICLK's plausible range.

    A heartbeat of 4211 is ordinary and a clock of 4211 MHz is not, so `sane` is the wrong
    door for them and `is_dead_arc` is the right one. Their readers are a follow-up.
    """
    assert aiclk.is_dead_arc((sysfs / "tenstorrent!0" / "tt_heartbeat").read_text())
    assert not aiclk.is_dead_arc((sysfs / "tenstorrent!1" / "tt_heartbeat").read_text())
    assert not aiclk.sane(4211)
    assert aiclk.is_dead_arc(f"{SENTINEL}\n") and not aiclk.is_dead_arc("1350")


# --------------------------------------------------------------------- the fleet sampler


def test_clocks_maps_the_sentinel_to_none_like_an_unreadable_node(sysfs, monkeypatch):
    from tt_bio.train import provenance

    monkeypatch.setattr(provenance, "_SYSFS", sysfs)
    assert provenance.clocks() == {0: None, 1: 1350}
    assert provenance.dead_arc_nodes() == [0]


def test_provenance_says_dead_arc_rather_than_no_card(sysfs, monkeypatch):
    """The old `why` claimed "this host has no Tenstorrent card visible", which on a
    dead-ARC host is a second wrong statement layered on the first."""
    from tt_bio.train import provenance

    monkeypatch.setattr(provenance, "_SYSFS", sysfs)
    monkeypatch.setattr(provenance, "open_nodes", lambda pid=None: [0])
    s = provenance._Sampler(interval=0.01)
    s.start()
    while not s.dead:
        pass
    got = s.stop()
    assert got["samples"] == 0
    assert got["dead_arc_nodes"] == [0]
    assert "dead ARC" in got["why"]
    assert "no Tenstorrent card" not in got["why"]


# ------------------------------------------------------------------- the stamping harness


def _meter(monkeypatch, tmp_path, dead_samples, live_samples=()):
    sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
    import meter

    monkeypatch.setattr(meter, "EVENTS", [])
    clock = meter.Clock.__new__(meter.Clock)
    clock.path = str(tmp_path / "tt_aiclk")
    clock.dead = dead_samples
    clock.samples = list(live_samples)
    monkeypatch.setattr(meter, "CLOCK", clock)
    return meter


def test_dump_refuses_a_sentinel_sample(monkeypatch, tmp_path):
    """The bar: a round_events.json whose clock field is the sentinel cannot be written."""
    meter = _meter(monkeypatch, tmp_path, 0, [(1.0, SENTINEL, 0.5)])
    out = tmp_path / "round_events.json"
    # BaseException, so BindCraft 2's `except Exception` around the compile cannot swallow it
    assert not issubclass(meter.DeadArcClock, Exception)
    with pytest.raises(meter.DeadArcClock):
        meter.dump(str(out), {"card": 3})
    assert not out.exists()


def test_dump_records_the_override_when_one_is_given(monkeypatch, tmp_path):
    meter = _meter(monkeypatch, tmp_path, 2, [(1.0, SENTINEL, 0.5), (2.0, 1350, 0.5)])
    monkeypatch.setenv(meter.ALLOW_DEAD, "1")
    out = tmp_path / "round_events.json"
    meter.dump(str(out), {"card": 3})
    got = json.loads(out.read_text())
    assert [s[1] for s in got["aiclk"]] == [1350]
    assert got["aiclk_dead_arc"] == {"dead_arc_reads": 3, "sentinel": SENTINEL,
                                     "override": True}
    assert got["stamp"]["aiclk_dead_arc"]["override"] is True


def test_dump_of_a_healthy_run_is_unchanged(monkeypatch, tmp_path):
    meter = _meter(monkeypatch, tmp_path, 0, [(1.0, 1350, 0.5)])
    out = tmp_path / "round_events.json"
    meter.dump(str(out), {"card": 3})
    got = json.loads(out.read_text())
    assert got["aiclk"] == [[1.0, 1350, 0.5]]
    assert "aiclk_dead_arc" not in got


def test_the_round_boundary_stops_the_run_when_the_arc_dies(monkeypatch, tmp_path):
    meter = _meter(monkeypatch, tmp_path, 4)
    monkeypatch.setattr(meter, "DUMP", None)
    with pytest.raises(meter.DeadArcClock):
        meter.Meter(rounds=10).on_sequence_gradients_enter()


def test_the_round_boundary_runs_on_through_a_healthy_clock(monkeypatch, tmp_path):
    meter = _meter(monkeypatch, tmp_path, 0, [(1.0, 1350, 0.5)])
    monkeypatch.setattr(meter, "DUMP", None)
    meter.Meter(rounds=10).on_sequence_gradients_enter()


# ------------------------------------------------------------------------ the shell door


def _sh(script):
    return subprocess.run(["sh", "-c", f'. "{HELPER}"\n{script}'],
                          capture_output=True, text=True)


def test_shell_helper_rejects_the_sentinel(sysfs):
    assert _sh(f'aiclk "{sysfs}/tenstorrent!0"').stdout.strip() == "DEAD"
    assert _sh(f'aiclk "{sysfs}/tenstorrent!1"').stdout.strip() == "1350"
    assert _sh(f'aiclk "{sysfs}/tenstorrent!9"').stdout.strip() == "NA"


def test_shell_helper_takes_a_node_dir_a_full_path_or_a_number(sysfs):
    assert _sh(f'aiclk "{sysfs}/tenstorrent!1/tt_aiclk"').stdout.strip() == "1350"
    assert _sh('aiclk 99999').stdout.strip() == "NA"


def test_shell_helper_exits_zero_so_it_is_safe_under_set_e(sysfs):
    r = _sh(f'set -e; v=$(aiclk "{sysfs}/tenstorrent!0"); echo "got $v"')
    assert r.returncode == 0
    assert r.stdout.strip() == "got DEAD"


def test_aiclk_alive_carries_the_status(sysfs):
    assert _sh(f'aiclk_alive "{sysfs}/tenstorrent!1"').returncode == 0
    assert _sh(f'aiclk_alive "{sysfs}/tenstorrent!0"').returncode == 1


def test_shell_bound_matches_the_python_one():
    """Two languages, one threshold. This test is the link between them."""
    m = re.search(r"^AICLK_MAX_PLAUSIBLE_MHZ=(\d+)$", HELPER.read_text(), re.M)
    assert m, "perf/lib/aiclk.sh no longer declares AICLK_MAX_PLAUSIBLE_MHZ"
    assert int(m.group(1)) == aiclk.MAX_PLAUSIBLE_MHZ


# --------------------------------------------------------------- no reader left behind


def test_no_shell_script_reads_tt_aiclk_without_the_helper():
    """The reason this row exists: one concern implemented 99 times, guarded once."""
    tracked = subprocess.run(["git", "ls-files", "--", "perf"], cwd=ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git work tree")
    offenders = []
    for rel in tracked.stdout.split():
        p = ROOT / rel
        if not rel.endswith(".sh") or p == HELPER or not p.exists():
            continue
        text = p.read_text()
        # a `cat`/`$(<)` straight at a tt_aiclk node, rather than a path handed to `aiclk`
        if re.search(r"(cat|<)\s+[\"']?[^\"'\s]*tt_aiclk", text):
            offenders.append(rel)
    assert offenders == [], f"direct tt_aiclk reads outside the helper: {offenders}"
