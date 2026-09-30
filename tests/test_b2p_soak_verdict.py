"""`perf/b2p_soak/verdict.py`: the instrument that has to catch drift, tested against drift.

A soak that runs for hours and ends in "the series looked flat" has proved nothing. This tests
the reader against series built to be wrong in each of the four ways the charter names -- a
per-trajectory memory floor that climbs, a handle leak, a cache filling the disk, and a slowdown
that arrives at the end rather than at the start -- and against a healthy series that must come
back CLEAN. An instrument that says DRIFT CLEAN about a leak is worse than no instrument.

Card-free and dependency-free: `verdict.py` is stdlib only.
"""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GB = 1 << 30


def load():
    spec = importlib.util.spec_from_file_location(
        "b2p_verdict", ROOT / "perf" / "b2p_soak" / "verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verdict = load()


def series(tmp_path, *, trajectories=6, rounds=20, s_per_round=12.0, rss_gb=18.0,
           rss_per_trajectory_gb=0.0, fds=24, fds_per_trajectory=0, cache_gb=0.1,
           cache_per_trajectory_gb=0.0, disk_free_gb=3400.0, slow_last=1.0, arms=2):
    """A campaign's round stamps and drift samples, shaped like the real ones."""
    out = tmp_path
    (out / "project").mkdir(parents=True, exist_ok=True)
    now, round_rows, samples = 1_790_000_000.0, [], []
    for index in range(trajectories):
        pace = s_per_round * (slow_last if index >= trajectories - max(1, trajectories // 3) else 1.0)
        for step in range(1, rounds + 1):
            round_rows.append({"t": now, "slot": f"t{index % arms + 1}", "round": step,
                               "load1": 2.0})
            now += pace
            samples.append({
                "t": now, "alive": True, "procs": 3,
                "rss": int((rss_gb + rss_per_trajectory_gb * index) * GB),
                "hwm": int((rss_gb + rss_per_trajectory_gb * index + 1) * GB),
                "threads": 2400, "fds": fds + fds_per_trajectory * index, "maps": 15000,
                "project_bytes": 1000 * (index + 1),
                "cache_bytes": {"xla": int((cache_gb + cache_per_trajectory_gb * index) * GB)},
                "disk_free": int(disk_free_gb * GB), "mem_available": 300 * GB,
                "load1": 2.0, "aiclk": 1000,
            })
    (out / "project" / "rounds.json").write_text(json.dumps(round_rows))
    (out / "drift.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return out


def run(out):
    samples = verdict.read_samples(out / "drift.jsonl")
    windows = verdict.trajectory_windows(json.loads((out / "project" / "rounds.json").read_text()))
    return verdict.verdict(samples, windows)


def test_a_healthy_campaign_is_clean(tmp_path):
    drifted, read = run(series(tmp_path))
    assert drifted == [], drifted
    assert any("6 trajectories" in line for line in read)


def test_the_trajectory_boundaries_come_from_the_round_stamps(tmp_path):
    """`round` restarting at 1 is the only boundary the stamps carry, and an interleaved
    campaign's arms are interleaved in one file."""
    out = series(tmp_path, trajectories=6, rounds=20, arms=2)
    windows = verdict.trajectory_windows(
        json.loads((out / "project" / "rounds.json").read_text()))
    assert len(windows) == 6
    assert [w["rounds"] for w in windows] == [20] * 6
    assert {w["slot"] for w in windows} == {"t1", "t2"}
    assert [w["n"] for w in windows] == [1, 2, 3, 4, 5, 6]


def test_a_memory_floor_that_climbs_per_trajectory_is_caught(tmp_path):
    """bwx's measurement: 4.4 GB per trajectory after the first."""
    drifted, _read = run(series(tmp_path, rss_per_trajectory_gb=4.4))
    assert any("host memory climbs" in line for line in drifted), drifted
    assert any("4.40 GB per trajectory" in line for line in drifted)


def test_breathing_inside_the_budget_is_not_a_leak(tmp_path):
    """A trajectory that holds 0.2 GB more than the last one is not a campaign that will be
    OOM-killed, and an instrument that calls it drift will be ignored."""
    drifted, _read = run(series(tmp_path, rss_per_trajectory_gb=0.2))
    assert drifted == [], drifted


def test_a_handle_leak_is_caught(tmp_path):
    drifted, _read = run(series(tmp_path, fds=24, fds_per_trajectory=6))
    assert any("open file handles grew" in line for line in drifted), drifted


def test_the_startup_samples_are_not_the_baseline(tmp_path):
    """**The trap this instrument fell into on the real series.**

    Run against `long24` on dev Galaxy .108 it reported "threads grew 1 -> 2408 (2408.00x)",
    "mapped regions 400.29x" and "handles 8.00x" on a campaign that was not leaking anything: the
    first sample is taken before the campaign exists -- the launcher hands over a `timeout`
    wrapper holding ONE thread and three handles -- and on Blackhole the minutes after that are
    compile workers, 30 processes and 358 handles, which then go away. So the baseline is the
    first trajectory's window, not the first sample, and an instrument that cries leak on every
    healthy campaign would have been switched off before it ever caught one.
    """
    out = series(tmp_path, trajectories=6, fds=24, fds_per_trajectory=0)
    samples = verdict.read_samples(out / "drift.jsonl")
    wrapper = dict(samples[0], t=samples[0]["t"] - 120, fds=3, threads=1, maps=38, rss=1 << 20)
    compiling = dict(samples[0], t=samples[0]["t"] - 60, fds=358, threads=261, maps=8135)
    rounds = json.loads((out / "project" / "rounds.json").read_text())
    drifted, read = verdict.verdict([wrapper, compiling] + samples,
                                   verdict.trajectory_windows(rounds))
    assert drifted == [], drifted
    assert any("threads 2400 in trajectory 1 -> 2400 in trajectory 6" in line for line in read)


def test_a_cache_that_will_fill_the_disk_is_caught(tmp_path):
    drifted, _read = run(series(tmp_path, cache_per_trajectory_gb=6.0, disk_free_gb=40.0))
    assert any("fills" in line or "holds only" in line for line in drifted), drifted


def test_a_cache_the_disk_can_afford_is_not_drift(tmp_path):
    drifted, _read = run(series(tmp_path, cache_per_trajectory_gb=0.5, disk_free_gb=3400.0))
    assert drifted == [], drifted


def test_a_slowdown_arriving_late_is_caught(tmp_path):
    """The charter's case is "a slowdown arriving at trajectory 15", which a
    first-against-last comparison over a whole campaign can average away."""
    drifted, _read = run(series(tmp_path, trajectories=9, slow_last=1.4))
    assert any("slows down" in line for line in drifted), drifted


def test_a_campaign_that_holds_its_pace_is_clean(tmp_path):
    drifted, _read = run(series(tmp_path, trajectories=9, slow_last=1.05))
    assert drifted == [], drifted


def test_a_partial_last_line_from_a_killed_sampler_is_skipped(tmp_path):
    out = series(tmp_path)
    with (out / "drift.jsonl").open("a") as series_file:
        series_file.write('{"t": 1790000999.0, "rss": 1000')
    assert run(out)[0] == []


def test_too_short_a_series_says_so_rather_than_guessing(tmp_path):
    drifted, read = run(series(tmp_path, trajectories=2, rounds=5))
    assert drifted == []
    assert any("no memory-per-trajectory verdict" in line for line in read)
    assert any("no slowdown verdict" in line for line in read)


def test_one_trajectory_gives_no_handle_verdict(tmp_path):
    """One window is a baseline with nothing to compare it to, which is not a clean bill."""
    drifted, read = run(series(tmp_path, trajectories=1, rounds=10))
    assert drifted == []
    assert any("no open file handles verdict" in line for line in read)


def test_a_missing_series_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["verdict.py", str(tmp_path / "nothing")])
    assert verdict.main() == 2


def test_the_exit_code_is_the_verdict(tmp_path, monkeypatch, capsys):
    clean = series(tmp_path / "clean")
    monkeypatch.setattr("sys.argv", ["verdict.py", str(clean)])
    assert verdict.main() == 0
    assert "DRIFT CLEAN" in capsys.readouterr().out

    leaking = series(tmp_path / "leaking", rss_per_trajectory_gb=4.4)
    monkeypatch.setattr("sys.argv", ["verdict.py", str(leaking)])
    assert verdict.main() == 1
    assert "DRIFT FOUND" in capsys.readouterr().out


def test_it_works_with_no_round_stamps_at_all(tmp_path):
    """A campaign killed before its first dump has no rounds.json; the handle and cache
    verdicts still stand, and the ones that need boundaries say they do not."""
    out = series(tmp_path)
    (out / "project" / "rounds.json").unlink()
    samples = verdict.read_samples(out / "drift.jsonl")
    drifted, read = verdict.verdict(samples, [])
    assert drifted == []
    assert any("0 trajectories" in line for line in read)
    assert any("no threads verdict" in line for line in read)


def interleaved(tmp_path, *, finished=4, arms=3, rss_step_gb=0.0, stage_load_gb=0.0):
    """The shape a REAL interleaved campaign writes: `round` counts up per arm and never restarts,
    and each finished trajectory is a row in the table plus a folder whose newest file is when it
    finished. bh24 on qb1, 2026-09-30, had `round == 1` three times in five charged trajectories."""
    out = tmp_path
    project = out / "project"
    (project / "1_Trajectories").mkdir(parents=True, exist_ok=True)
    t0, rounds, samples, names = 1_790_000_000.0, [], [], []
    per_trajectory_s = 40 * 60
    total = per_trajectory_s * (finished + 1)
    for arm in range(arms):
        t, step = t0 + arm * 60, 1
        while t < t0 + total:
            rounds.append({"t": t, "slot": f"t{arm + 1}", "round": step})
            t, step = t + 18.3, step + 1
    for k in range(finished):
        name = f"traj_{k}"
        names.append(name)
        folder = project / "1_Trajectories" / name
        folder.mkdir()
        stamp = folder / "Trajectory.pdb"
        stamp.write_text("x")
        when = t0 + per_trajectory_s * (k + 1)
        import os
        os.utime(stamp, (when, when))
    with open(project / "1_Trajectories" / "!_Trajectories.csv", "w") as f:
        f.write("trajectory,design\n" + "".join(f"{i + 1},{n}\n" for i, n in enumerate(names)))
    for i in range(0, int(total), 30):
        t = t0 + i
        done = sum(1 for k in range(finished) if t >= t0 + per_trajectory_s * (k + 1))
        # a one-off step when the first trajectory reaches a new stage, and optionally a leak
        rss = 19.0 + (stage_load_gb if done >= 1 else 0.0) + rss_step_gb * done
        samples.append({"t": t, "alive": True, "rss": int(rss * GB), "fds": 25, "threads": 714,
                        "maps": 13300, "cache_bytes": {}, "disk_free": int(300 * GB)})
    (project / "rounds.json").write_text(json.dumps(rounds))
    (out / "drift.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return out


def run_real(out):
    samples = verdict.read_samples(out / "drift.jsonl")
    rounds = json.loads((out / "project" / "rounds.json").read_text())
    windows = verdict.boundary_windows(verdict.completion_times(out / "project"), rounds)
    return verdict.verdict(samples, windows)


def test_arms_starting_up_are_not_trajectory_boundaries(tmp_path):
    out = interleaved(tmp_path)
    rounds = json.loads((out / "project" / "rounds.json").read_text())
    assert len(verdict.trajectory_windows(rounds)) == 3            # one per arm: the old reading
    assert len(verdict.completion_times(out / "project")) == 4     # the campaign's own count


def test_a_new_stage_loading_once_is_not_a_leak(tmp_path):
    drift, _ = run_real(interleaved(tmp_path, stage_load_gb=0.9))
    assert not drift


def test_a_real_leak_on_an_interleaved_campaign_is_still_caught(tmp_path):
    drift, _ = run_real(interleaved(tmp_path, rss_step_gb=1.5))
    assert any("host memory climbs" in d for d in drift)


def test_the_window_pace_is_the_amortised_round(tmp_path):
    out = interleaved(tmp_path, arms=3)
    windows = verdict.boundary_windows(verdict.completion_times(out / "project"),
                                       json.loads((out / "project" / "rounds.json").read_text()))
    later = windows[1:]
    assert all(abs(w["s_per_round"] - 18.3 / 3) < 0.3 for w in later)


def test_no_trajectory_table_yet_means_no_completion_times(tmp_path):
    (tmp_path / "project").mkdir()
    assert verdict.completion_times(tmp_path / "project") == []


def test_arms_finishing_together_are_one_boundary_not_three(tmp_path):
    rounds = [{"t": 1_790_000_000.0 + i * 6.0, "slot": "t1", "round": i + 1} for i in range(600)]
    start = rounds[0]["t"]
    together = [start + 2400, start + 2640, start + 2670]           # bh24: 16:27:33, :31:34, :32:04
    windows = verdict.boundary_windows(together + [start + 3500], rounds)
    assert len(windows) == 2
    assert windows[0]["end"] == start + 2670


def test_the_real_bh24_shape_is_declined_not_called_a_leak(tmp_path):
    out = interleaved(tmp_path, finished=1, stage_load_gb=0.9)
    drift, read = run_real(out)
    assert not drift
    assert any("fewer than 3 trajectory boundaries" in line for line in read)
