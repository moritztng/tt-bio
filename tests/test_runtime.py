"""Regression: detect_tenstorrent_devices must validate an explicit --device_ids against the
cards actually present, raising a clear error instead of passing a bad id straight through to a
deep, opaque ttnn device-open crash. Host-only; device discovery is monkeypatched (no hardware).
"""
from __future__ import annotations

import os

import pytest

from tt_bio import runtime


@pytest.fixture
def two_cards(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob",
                        lambda pat: ["/dev/tenstorrent/0", "/dev/tenstorrent/1"])
    # detect_tenstorrent_devices also filters by ambient TT_VISIBLE_DEVICES, so a suite run
    # from a shell pinned to one card would see through the fake two-card box and fail.
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    return None


def test_explicit_ids_selected_when_present(two_cards):
    assert runtime.detect_tenstorrent_devices("0,1", 0, max_workers=10) == [0, 1]
    assert runtime.detect_tenstorrent_devices("1", 0, max_workers=10) == [1]


def test_missing_id_raises_clear_error(two_cards):
    with pytest.raises(ValueError) as ei:
        runtime.detect_tenstorrent_devices("7", 0, max_workers=10)
    msg = str(ei.value)
    assert "7" in msg and "0, 1" in msg  # names the bad id and the available ones


def test_missing_id_when_no_cards(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob", lambda pat: [])
    with pytest.raises(ValueError) as ei:
        runtime.detect_tenstorrent_devices("0", 0, max_workers=10)
    assert "none detected" in str(ei.value)


def test_num_devices_and_default(two_cards):
    assert runtime.detect_tenstorrent_devices(None, 1, max_workers=10) == [0]
    assert runtime.detect_tenstorrent_devices(None, 0, max_workers=10) == [0, 1]
    assert runtime.detect_tenstorrent_devices(None, 0, max_workers=1) == [0]  # max_workers cap honored


def test_duplicate_stem_inputs_rejected(tmp_path):
    (tmp_path / "target.fasta").write_text(">A|protein\nMK\n")
    (tmp_path / "target.yaml").write_text("sequences: []\n")
    struct = tmp_path / "structures"
    struct.mkdir()
    with pytest.raises(ValueError, match="share a name stem"):
        runtime.discover_jobs(tmp_path, struct, "cif", override=True)


def test_unique_stems_discovered(tmp_path):
    (tmp_path / "a.fasta").write_text(">A|protein\nMK\n")
    (tmp_path / "b.yaml").write_text("sequences: []\n")
    struct = tmp_path / "structures"
    struct.mkdir()
    jobs = runtime.discover_jobs(tmp_path, struct, "cif", override=True)
    assert sorted(j.id for j in jobs) == ["a", "b"]


def test_host_thread_cap_splits_the_budget_across_cards():
    assert runtime.host_thread_cap(4, 32) == 8
    assert runtime.host_thread_cap(1, 32) == 32
    # never zero, however lopsided the split
    assert runtime.host_thread_cap(64, 4) == 1
    assert runtime.host_thread_cap(0, 8) == 8


def test_host_thread_cap_defaults_to_the_whole_host():
    assert runtime.host_thread_cap(2) == max(1, (os.cpu_count() or 1) // 2)


def test_host_thread_cap_env_caps_every_pool(monkeypatch):
    for var in runtime.HOST_THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    env = runtime.host_thread_cap_env(4, 32)
    assert env == {var: "8" for var in runtime.HOST_THREAD_VARS}


def test_host_thread_cap_env_leaves_an_operator_value_alone(monkeypatch):
    for var in runtime.HOST_THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    # no explicit budget: the operator knows best, only fill in what they left unset
    assert "OMP_NUM_THREADS" not in runtime.host_thread_cap_env(4)
    # explicit budget: the launcher knows how many siblings it started, so it wins
    assert runtime.host_thread_cap_env(4, 32)["OMP_NUM_THREADS"] == "8"


def test_host_thread_cap_env_parks_idle_threads_only_when_cores_are_scarce(monkeypatch):
    for var in runtime.HOST_THREAD_VARS + tuple(runtime.IDLE_THREADS_PARK):
        monkeypatch.delenv(var, raising=False)
    # 64 threads over 32 or 24 concurrent folds is 2 each: no core is left to absorb a spinning
    # pool thread, so the children park theirs. Measured 1.014x and 1.004x on folds per hour.
    assert runtime.IDLE_THREADS_PARK.items() <= runtime.host_thread_cap_env(32, 64).items()
    assert runtime.IDLE_THREADS_PARK.items() <= runtime.host_thread_cap_env(24, 64).items()
    # Width 20 is the boundary, and it is measured, not assumed: 3 threads each reads 0.990x, so
    # one share above the line the lever is a LOSS and has to stay off. Width 16 likewise (0.993x).
    assert not set(runtime.IDLE_THREADS_PARK) & set(runtime.host_thread_cap_env(20, 64))
    assert not set(runtime.IDLE_THREADS_PARK) & set(runtime.host_thread_cap_env(16, 64))
    assert not set(runtime.IDLE_THREADS_PARK) & set(runtime.host_thread_cap_env(1, 64))


def test_host_thread_cap_env_leaves_an_operator_wait_policy_alone(monkeypatch):
    for var in runtime.HOST_THREAD_VARS + tuple(runtime.IDLE_THREADS_PARK):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    env = runtime.host_thread_cap_env(32, 64)
    # all three, not just the one they named: GOMP_SPINCOUNT=0 would undo their ACTIVE anyway
    assert not set(runtime.IDLE_THREADS_PARK) & set(env)
    assert env["OMP_NUM_THREADS"] == "2"         # the thread cap is still ours


def test_host_thread_cap_env_is_byte_identical_above_the_line(monkeypatch):
    """A caller above the parking line gets exactly the env it got before parking existed.

    This is the perf-regression argument for the single-fold and low-concurrency paths, and it is
    mechanical rather than timed: a lone ``tt-bio predict`` sees n_workers == 1, so its share is
    every thread on the box and nothing about its environment moves. Timing it could only measure
    noise.
    """
    for var in runtime.HOST_THREAD_VARS + tuple(runtime.IDLE_THREADS_PARK):
        monkeypatch.delenv(var, raising=False)
    for n_workers, host_threads in ((1, 64), (1, None), (4, 64), (16, 64), (20, 64), (2, 8)):
        env = runtime.host_thread_cap_env(n_workers, host_threads)
        cap = runtime.host_thread_cap(n_workers, host_threads)
        if cap > runtime.IDLE_PARK_THREAD_SHARE:
            assert env == {var: str(cap) for var in runtime.HOST_THREAD_VARS}


def _boltzgen_child_envs(monkeypatch, tmp_path, n_devices, cores, operator_env=None):
    """Run BoltzGen's multi-device fan-out far enough to capture each child's environment.

    ``--debug`` is the branch with no progress view, so a fake Popen that reports success is
    enough to reach the end of the function. Nothing here opens a device.
    """
    import argparse
    import subprocess as _subprocess
    from tt_bio.boltzgen.cli import boltzgen as bg
    import tt_bio.main as bg_main

    for var in runtime.HOST_THREAD_VARS + tuple(runtime.IDLE_THREADS_PARK):
        monkeypatch.delenv(var, raising=False)
    for var, val in (operator_env or {}).items():
        monkeypatch.setenv(var, val)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cores)))
    monkeypatch.setattr(bg_main, "_detect_p300_devices", lambda: [])
    monkeypatch.setattr(bg.sys, "argv", ["boltzgen", "run", "--output", str(tmp_path)])

    captured = []

    class _Launched(Exception):
        """Every worker is up and its environment recorded; the merge past here is not our subject."""

    class _Proc:
        returncode = 0

        def wait(self):
            # _run_distributed launches all N workers before it waits on any, so the first wait
            # is the exact point where the captured environments are complete.
            raise _Launched

    def _fake_popen(cmd, env=None, **kwargs):
        captured.append(env)
        return _Proc()

    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)
    args = argparse.Namespace(output=tmp_path, num_designs=n_devices, device_ids=None, debug=True)
    with pytest.raises(_Launched):
        bg._run_distributed(args, list(range(n_devices)))
    assert len(captured) == n_devices
    return captured


def test_boltzgen_fanout_uses_the_one_thread_cap_builder(monkeypatch, tmp_path):
    """BoltzGen's per-card fan-out is a DP path and must get the same child env ``predict`` does.

    It used to build that env itself -- a second copy of ``cores // workers`` that never picked up
    idle-thread parking, so the one dispatch path that routinely starves the host (32 single-chip
    design workers on whglx's 64 threads, a 2-thread share) kept spinning its pool threads through
    every device sync. Same builder now, so the rule lands in one place.
    """
    for env in _boltzgen_child_envs(monkeypatch, tmp_path, n_devices=32, cores=64):
        assert env["OMP_NUM_THREADS"] == "2"
        assert runtime.IDLE_THREADS_PARK.items() <= env.items()


def test_boltzgen_fanout_leaves_the_thread_cap_unchanged_above_the_line(monkeypatch, tmp_path):
    """The cap itself must not move: 64 threads over 4 workers is still 16 each, and a fan-out
    with cores to spare still spins, exactly as it did before the builder was shared."""
    for env in _boltzgen_child_envs(monkeypatch, tmp_path, n_devices=4, cores=64):
        assert env["OMP_NUM_THREADS"] == "16"
        assert all(env[var] == "16" for var in runtime.HOST_THREAD_VARS)
        assert not set(runtime.IDLE_THREADS_PARK) & set(env)


def test_boltzgen_fanout_still_respects_an_explicit_operator_setting(monkeypatch, tmp_path):
    """The old code used ``setdefault`` so an operator's OMP_NUM_THREADS survived. Keep that."""
    for env in _boltzgen_child_envs(monkeypatch, tmp_path, n_devices=32, cores=64,
                                    operator_env={"OMP_NUM_THREADS": "7"}):
        assert env["OMP_NUM_THREADS"] == "7"
        # They capped threads, not the wait policy, so parking is still ours to set.
        assert runtime.IDLE_THREADS_PARK.items() <= env.items()
