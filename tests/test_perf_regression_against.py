"""perf_regression.py --against: the reference tree measures itself first, then the gate reads
the baseline file that run wrote, not the committed one."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("perf_regression_t", REPO / "scripts" / "perf_regression.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_against_measures_the_reference_tree_then_gates_on_its_file(tmp_path):
    ref = tmp_path / "v0.13.1"
    (ref / "scripts").mkdir(parents=True)
    (ref / "scripts" / "perf_regression.py").write_text(
        "import json, os, sys, pathlib\n"
        "pathlib.Path('docs').mkdir(exist_ok=True)\n"
        "pathlib.Path('docs/perf_baselines.json').write_text(json.dumps("
        "{'argv': sys.argv[1:], 'pythonpath': os.environ['PYTHONPATH']}))\n")
    m = _module()
    m._measure_reference(Namespace(against=ref, model=["nesso1"]))
    assert m.BASELINE_FILE == ref / "docs" / "perf_baselines.json"
    got = json.loads(m.BASELINE_FILE.read_text())
    assert got["argv"][:2] == ["--update-baseline", "--allow-contended"]
    assert got["argv"][-2:] == ["--model", "nesso1"]
    assert got["pythonpath"].split(":")[0] == str(ref)
    assert json.loads((REPO / "docs" / "perf_baselines.json").read_text()) != got


def test_a_failed_reference_run_fails_the_gate(tmp_path):
    ref = tmp_path / "ref"
    (ref / "scripts").mkdir(parents=True)
    (ref / "scripts" / "perf_regression.py").write_text("import sys; sys.exit(3)\n")
    m = _module()
    try:
        m._measure_reference(Namespace(against=ref, model=None))
    except SystemExit as e:
        assert "exited 3" in str(e)
    else:
        raise AssertionError("a failed reference run must fail the gate")


class _Clock:
    def __init__(self, median):
        self.median = median

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return {0: {"min": self.median, "max": self.median, "median": self.median, "n": 3}}


def _fake_child(row):
    class Popen:
        def __init__(self, cmd, **kw):
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(row))
            self.pid = 0

        def wait(self, timeout=None):
            return 0
    return Popen


def test_a_measurement_carries_the_clock_and_a_clamped_one_cannot_seed(monkeypatch, tmp_path):
    m = _module()
    row = {"model": "boltz2", "hardware": "blackhole", "throughput": 0.8}
    monkeypatch.setattr(m.subprocess, "Popen", _fake_child(row))
    monkeypatch.setattr(m, "clock_during", lambda: _Clock(1350))
    fast = m._run_measure("boltz2")
    assert fast["aiclk_mhz"]["median"] == 1350 and not fast.get("clock_low")
    monkeypatch.setattr(m, "clock_during", lambda: _Clock(800))
    slow = m._run_measure("boltz2")
    assert slow["clock_low"]
    monkeypatch.setattr(m, "BASELINE_FILE", tmp_path / "b.json")
    try:
        m._update_baselines([slow], Namespace(note="x", allow_contended=False))
    except SystemExit as e:
        assert "boltz2 peak 800 MHz" in str(e)
    else:
        raise AssertionError("a baseline was seeded from an 800 MHz measurement")
    assert not (tmp_path / "b.json").exists()


def test_the_clock_floor_is_per_architecture(monkeypatch):
    """A busy Wormhole Galaxy chip runs at 1000 MHz, under Blackhole's floor; only a peak under
    Wormhole's own floor is clamped there."""
    m = _module()
    assert m.hardware_of("tt-galaxy-wh-l") == "wormhole_b0" and m.hardware_of("n300") == "wormhole_b0"
    assert m.hardware_of("p150a") == "blackhole" and m.hardware_of("p300c") == "blackhole"
    row = {"model": "boltzgen", "hardware": "wormhole_b0", "throughput": 0.01}
    monkeypatch.setattr(m.subprocess, "Popen", _fake_child(row))
    monkeypatch.setattr(m, "clock_during", lambda: _Clock(1000))
    assert not m._run_measure("boltzgen").get("clock_low")
    monkeypatch.setattr(m, "clock_during", lambda: _Clock(500))
    assert m._run_measure("boltzgen")["clock_low"]
