"""The fair harness without a card: run.py's climb and discard rules against a stub engine, and
verdict.py's admission rule and ratio against hand-made lines.

    python -m pytest perf/rfd4/fair/test_fair.py
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # perf/ for clocksample

try:
    import tt_bio.main  # noqa: F401
except ImportError:  # a CPU venv without the engine's deps: run.py only needs the descriptor hook
    sys.modules["tt_bio.main"] = types.SimpleNamespace(ensure_p300_mesh_descriptor=lambda: None)

import run  # noqa: E402
import verdict  # noqa: E402


class _Clock:
    """clocksample.during with fixed readings."""
    mhz = 1350

    def __init__(self, period=1.0):
        self.clocks, self.power, self.load = {0: [_Clock.mhz] * 3}, {0: [90.0, 110.0]}, [0.05]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        xs = self.clocks[0]
        return {0: {"min": min(xs), "max": max(xs), "median": xs[len(xs) // 2], "n": len(xs)}}


def _stub_rfd4(monkeypatch, tmp_path, *, dram_width, per_call_s):
    """A DesignEngine whose wall time is per_call_s(width) and whose DRAM fits `dram_width`."""
    import torch

    clock = {"t": 0.0}
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock["t"])

    class Engine:
        spec = types.SimpleNamespace(nsteps=60)

        def __init__(self, weights, recipe):
            assert recipe == "thinking_medium"
            self.max_parallel_samples, self.chunk_width = 1, 1

        def design(self, features, *, num_designs, nsteps, seed, rng):
            assert nsteps == 60 and rng == "per_design"
            self.chunk_width = min(self.max_parallel_samples, dram_width)
            clock["t"] += per_call_s(self.max_parallel_samples)
            return object()

    ex = types.SimpleNamespace(cif="x.cif", example_id="pdl1")
    rfd4 = types.SimpleNamespace(
        prepare_examples=lambda spec, d: [ex],
        featurize=lambda cif, seed: ({"padding_mask": torch.ones(1, 128)}, None),
        write_design=lambda *a, **k: None)
    engine = types.SimpleNamespace(DesignEngine=Engine)
    sampler = types.SimpleNamespace(recycle_depths=lambda spec, n: [3] * n)
    import tt_bio
    for name, mod in (("rfd4", rfd4), ("rfd4.engine", engine), ("rfd4.sampler", sampler)):
        monkeypatch.setitem(sys.modules, f"tt_bio.{name}", mod)
    monkeypatch.setattr(tt_bio, "rfd4", rfd4, raising=False)
    rfd4.engine, rfd4.sampler = engine, sampler
    monkeypatch.setattr(run.clocksample, "during", _Clock)
    monkeypatch.setattr(run, "board_type", lambda: "p150a")
    monkeypatch.setattr(run.Contention, "device_holders", staticmethod(lambda: set()))
    for f in ("_box_busy", "_own_busy"):  # an idle box, whatever the test host is doing
        monkeypatch.setattr(run.Contention, f, staticmethod(lambda: 0))


def _run(tmp_path, *extra):
    sys.argv = ["run.py", "--spec", "s.yaml", "--weights", "w", "--out", str(tmp_path), "--part", "p150a", *extra]
    run.main()
    return [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text().splitlines()]


def test_climb_stops_where_dram_pushes_back(monkeypatch, tmp_path):
    _stub_rfd4(monkeypatch, tmp_path, dram_width=4, per_call_s=lambda w: 10.0 + w)
    lines = _run(tmp_path)
    climb = next(r for r in lines if r["kind"] == "climb")
    assert [r["batch"] for r in climb["rungs"]] == [1, 2, 4, 8]
    assert climb["best_batch"] == 4 and climb["stopped"] == "pushback"
    measure = [r for r in lines if r.get("phase") == "measure"]
    assert len(measure) == 5 and {r["batch"] for r in measure} == {4}
    assert len({r["seed"] for r in measure}) == 5
    assert all("discard" not in r and r["trunk_passes"] == 180 for r in measure)
    assert all(r["foreign_cores_median"] is not None for r in measure)
    assert measure[0]["designs_per_s"] == round(4 / 14.0, 5)
    assert measure[0]["power_w"] == 100.0 and measure[0]["power_w_peak"] == 110.0
    summary = lines[-1]
    assert summary["kind"] == "summary" and summary["clean"] == 5


def test_climb_stops_after_two_flat_rungs_and_honours_the_ceiling(monkeypatch, tmp_path):
    _stub_rfd4(monkeypatch, tmp_path, dram_width=1024, per_call_s=lambda w: float(max(w, 4)))
    climb = next(r for r in _run(tmp_path) if r["kind"] == "climb")
    assert [r["batch"] for r in climb["rungs"]] == [1, 2, 4, 8, 16]
    assert climb["best_batch"] == 4  # 8 and 16 bought nothing

    _stub_rfd4(monkeypatch, tmp_path, dram_width=1024, per_call_s=lambda w: 10.0)
    climb = next(r for r in _run(tmp_path / "c", "--ceiling", "8") if r["kind"] == "climb")
    assert climb["best_batch"] == 8 and climb["stopped"] == "ceiling"

    _stub_rfd4(monkeypatch, tmp_path, dram_width=1024, per_call_s=lambda w: float(w))
    climb = next(r for r in _run(tmp_path / "t", "--climb-through", "8") if r["kind"] == "climb")
    assert [r["batch"] for r in climb["rungs"]] == [1, 2, 4, 8]  # flat from b2, climbed through b8


def test_under_clocked_batches_are_discarded(monkeypatch, tmp_path):
    _stub_rfd4(monkeypatch, tmp_path, dram_width=8, per_call_s=lambda w: 10.0)
    monkeypatch.setattr(_Clock, "mhz", 800)
    lines = _run(tmp_path, "--batch", "2")
    measure = [r for r in lines if r.get("phase") == "measure"]
    assert all(r["discard"].startswith("clock_artifact") for r in measure)
    assert lines[-1]["clean"] == 0 and lines[-1]["designs_per_s_median"] is None


def test_a_foreign_device_holder_aborts_before_the_engine_opens(monkeypatch, tmp_path):
    _stub_rfd4(monkeypatch, tmp_path, dram_width=8, per_call_s=lambda w: 10.0)
    monkeypatch.setattr(run.Contention, "device_holders", staticmethod(lambda: {"42:python"}))
    with pytest.raises(SystemExit):
        _run(tmp_path)
    assert json.loads((tmp_path / "runs.jsonl").read_text())["kind"] == "abort"


def _line(size, dps, sha, **kw):
    return {"kind": "warm", "phase": "measure", "size": size, "batch": 4, "designs": 4, "wall_s": 4 / dps,
            "designs_per_s": dps, "part": "p150a", "host": "pc", "card": "0", "sha": sha, "tree_dirty": False,
            "raw": f"r{size}", "t": "2026-10-22T00:00:00Z", "aiclk_mhz": 1350, "aiclk_min_mhz": 1340,
            "foreign_cores_median": 0.1, "host_load_per_core_median": 0.1, "seed": 1, **kw}


def test_verdict_admits_only_clean_p150a_main_lines_and_applies_section_7(tmp_path):
    repo = str(HERE.parents[2])
    main = subprocess.run(["git", "rev-parse", "origin/main"], cwd=repo, capture_output=True, text=True).stdout.strip()
    lines = [_line(128, 0.30 + i / 1000, main) for i in range(5)]
    lines += [_line(128, 9.0, main, discard="contended: x"), _line(128, 9.0, main, part="p300c"),
              _line(128, 9.0, main, tree_dirty=True), _line(256, 0.1, main)]  # 256: one batch, not five
    runs = tmp_path / "runs.jsonl"
    runs.write_text("".join(json.dumps(x) + "\n" for x in lines))
    gpu = tmp_path / "gpu.jsonl"
    gpu.write_text("".join(json.dumps({"kind": "warm", "size": 128, "designs": 8, "wall_s": 8 / 2.0,
                                       "arm": "a", "batch": 8}) + "\n" for _ in range(5)))
    tt, rejected = verdict.tt_side([runs], repo)
    assert set(tt) == {128} and tt[128]["designs_per_s"] == 0.302 and tt[128]["batches"] == 5
    assert rejected == {"contended": 1, "part p300c, not p150a": 1, "dirty tree": 1}
    g = verdict.gpu_side(gpu)
    assert g[128]["designs_per_s"] == 2.0
    assert verdict.ratio(0.302, 2.0) == pytest.approx(0.302 * 14.0538 / 2.0, rel=1e-4)
    assert verdict.ratio(0.2135 * 2.0, 2.0) == pytest.approx(3.0, abs=0.01)
