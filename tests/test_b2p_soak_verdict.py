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
import os
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
    assert any("open file handles is still growing" in line for line in drifted), drifted


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


def test_too_few_trajectories_is_not_yet_judged_rather_than_clean(tmp_path, monkeypatch, capsys):
    short = series(tmp_path, trajectories=2, rounds=5)
    monkeypatch.setattr("sys.argv", ["verdict.py", str(short)])
    assert verdict.main() == 3
    said = capsys.readouterr().out
    assert "NOT YET JUDGED" in said and "DRIFT CLEAN" not in said


def staircase(tmp_path, *, steps, per_step=3000, base=13300, trajectories=9, still=False):
    """A campaign whose mapped-region count climbs in steps, as a stage's kernels load once.

    `steps` steps in the first part of the campaign and then a plateau, which is the real bh24
    shape: 13.3k, 19.3k, 22.1k, 25.7k over the first nine trajectories and flat for the eight
    after. With `still=True` the last third keeps climbing, which is the leak.
    """
    project = tmp_path / "project"
    (project / "1_Trajectories").mkdir(parents=True)
    t0, per = 1_790_000_000.0, 1200.0
    names = [f"d{k}" for k in range(trajectories)]
    for k, name in enumerate(names):
        folder = project / "1_Trajectories" / name
        folder.mkdir()
        stamp = folder / "Trajectory.pdb"
        stamp.write_text("x")
        os.utime(stamp, ((t0 + per * (k + 1)),) * 2)
    with open(project / "1_Trajectories" / "!_Trajectories.csv", "w") as f:
        f.write("trajectory,design\n" + "".join(f"{i + 1},{n}\n" for i, n in enumerate(names)))
    rounds, samples = [], []
    total = per * (trajectories + 1)
    for i in range(0, int(total), 30):
        t = t0 + i
        done = sum(1 for k in range(trajectories) if t >= t0 + per * (k + 1))
        climbed = min(done, steps) + (max(done - steps, 0) if still else 0)
        samples.append({"t": t, "alive": True, "rss": int(19.0 * GB), "fds": 25, "threads": 714,
                        "maps": base + per_step * climbed, "cache_bytes": {},
                        "disk_free": int(300 * GB)})
    for i in range(0, int(total), 6):
        rounds.append({"t": t0 + i, "slot": "t1", "round": i // 6 + 1})
    (project / "rounds.json").write_text(json.dumps(rounds))
    (tmp_path / "drift.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return tmp_path


def test_a_saturating_map_staircase_is_warmup_not_a_leak(tmp_path):
    """bh24's own shape: 13.3k -> 25.7k in three steps, then flat for eight trajectories.

    First-against-last reads 1.93x and calls a warmed-up campaign leaky. The steps are the refold,
    MPNN and validation stages loading their kernels once each, and the count then holds.
    """
    drift, read = run_real(staircase(tmp_path, steps=3))
    assert not any("mapped regions" in d for d in drift), drift
    assert any("mapped regions settled" in line for line in read), read
    assert any("vm.max_map_count" in line for line in read), read


def test_mapped_regions_still_climbing_at_the_end_are_a_leak(tmp_path):
    drift, _read = run_real(staircase(tmp_path, steps=3, still=True))
    assert any("mapped regions is still growing" in d for d in drift), drift
    assert any("reaches the limit in" in d for d in drift), drift


def test_the_map_limit_is_projected_from_the_late_rate_not_the_staircase(tmp_path):
    """The projection has to use the rate the series still climbs at, not its warmup average."""
    line = verdict.map_limit_line(25700, 1400, path="/proc/sys/vm/max_map_count")
    limit = verdict.map_limit()
    if limit:
        assert f"{(limit - 25700) / 1400:.0f} more trajectories" in line
    else:
        assert "publishes no vm.max_map_count" in line


def test_a_host_without_a_map_limit_says_so_rather_than_dividing(tmp_path):
    line = verdict.map_limit_line(25700, 1400, path=str(tmp_path / "not-here"))
    assert "publishes no vm.max_map_count" in line


def test_a_resumed_leg_that_rewrote_rounds_json_still_cuts_its_windows(tmp_path):
    """The box-lost drill's own shape, and it cost the Wormhole leg its memory verdict.

    A resumed campaign is handed the same project folder and the harness rewrote `rounds.json`
    from empty, so every round stamp was NEWER than all ten trajectory completions. The leading
    edge sat after the last boundary, not one window was cut, and a 4.21 h series with ten
    finished trajectories printed "0 trajectories" and no memory verdict at all.
    """
    out = staircase(tmp_path, steps=3)
    samples = verdict.read_samples(out / "drift.jsonl")
    boundaries = verdict.completion_times(out / "project")
    resumed = [{"t": samples[-1]["t"] + 60 + i * 6, "slot": "t1", "round": i + 1}
               for i in range(50)]
    assert verdict.boundary_windows(boundaries, resumed) == []          # the defect
    windows = verdict.boundary_windows(boundaries, resumed, samples)
    assert len(windows) >= 3
    _drift, read = verdict.verdict(samples, windows)
    assert not any("no memory-per-trajectory verdict" in line for line in read), read
    assert any("rss at trajectory boundaries" in line for line in read), read


def test_a_window_with_no_round_stamps_carries_no_pace_rather_than_vanishing(tmp_path, capsys,
                                                                            monkeypatch):
    out = staircase(tmp_path, steps=3)
    (out / "project" / "rounds.json").write_text(json.dumps([]))
    monkeypatch.setattr("sys.argv", ["verdict.py", str(out)])
    verdict.main()
    printed = capsys.readouterr().out
    assert "pace not recorded" in printed
    assert "0 trajectories" not in printed


def drained(*, windows=9, arms=3, tail_arms=2, tail=1, tail_pace=1.19, mid_short=None):
    """Round stamps from `arms` arms, one boundary per window; the last `tail` windows run on
    `tail_arms` arms at `tail_pace` times the amortised round, the shape of `bh24`'s end."""
    rounds, boundaries, now = [], [], 1_790_000_000.0
    for index in range(windows):
        late = index >= windows - tail
        live = tail_arms if late or index == mid_short else arms
        step = 6.28 * (tail_pace if late or index == mid_short else 1.0)
        for n in range(150):          # past MIN_WINDOW_S, as a real window is
            rounds.append({"t": now, "slot": f"t{n % live + 1}", "round": n + 1})
            now += step
        boundaries.append(now)
    return verdict.boundary_windows(boundaries, rounds)


def test_the_drain_at_a_campaigns_end_is_not_a_slowdown():
    """bh24's last window ran on two of three arms: amortised 6.28 -> 7.48 s while each arm got
    faster, 18.2 -> 8.5 s a round. That is the budget running out, not the campaign slowing."""
    windows = drained()
    assert [w["arms"] for w in windows] == [3] * 8 + [2]
    drifted, read = verdict.verdict([{"t": 0, "alive": False}], windows)
    assert not any("slows down" in line for line in drifted), drifted
    assert any("[9] ran on fewer than 3 arms" in line for line in read), read


def test_a_slowdown_with_every_arm_present_is_still_caught():
    drifted, _read = verdict.verdict([{"t": 0, "alive": False}],
                                     drained(tail=3, tail_arms=3, tail_pace=1.4))
    assert any("slows down" in line for line in drifted), drifted


def test_a_short_handed_window_mid_campaign_is_still_judged():
    """An arm in MPNN or validation stamps no rounds, so only the TAIL is a drain."""
    windows = drained(tail=0, mid_short=7, tail_pace=1.4)
    drifted, read = verdict.verdict([{"t": 0, "alive": False}], windows)
    assert not any("arms (the campaign draining)" in line for line in read), read
    assert any("slows down" in line for line in drifted), drifted


def killed(tmp_path, *, trajectories=9, watched=5, fds_per_trajectory=0, rss_per_trajectory_gb=0.0):
    """`long24`'s own shape: the box-lost drill kills the campaign mid-budget, the sampler stamps
    one dead tick, and a resumed leg keeps filling the SAME project folder afterwards.

    So the folder holds `trajectories` completions while the series only watched `watched` of
    them, and its last sample is the zeros a dead process reports.
    """
    project = tmp_path / "project"
    (project / "1_Trajectories").mkdir(parents=True)
    t0, per = 1_790_000_000.0, 1200.0
    names = [f"d{k}" for k in range(trajectories)]
    for k, name in enumerate(names):
        folder = project / "1_Trajectories" / name
        folder.mkdir()
        stamp = folder / "Trajectory.pdb"
        stamp.write_text("x")
        os.utime(stamp, ((t0 + per * (k + 1)),) * 2)
    with open(project / "1_Trajectories" / "!_Trajectories.csv", "w") as f:
        f.write("trajectory,design\n" + "".join(f"{i + 1},{n}\n" for i, n in enumerate(names)))
    killed_at = t0 + per * (watched + 0.5)
    rounds, samples = [], []
    for i in range(0, int(killed_at - t0), 30):
        t = t0 + i
        done = sum(1 for k in range(trajectories) if t >= t0 + per * (k + 1))
        samples.append({"t": t, "alive": True,
                        "rss": int((18.0 + rss_per_trajectory_gb * done) * GB),
                        "fds": 24 + fds_per_trajectory * done, "threads": 2400, "maps": 15000,
                        "cache_bytes": {}, "disk_free": int(300 * GB)})
    for i in range(0, int(killed_at - t0), 6):
        rounds.append({"t": t0 + i, "slot": "t1", "round": i // 6 + 1})
    samples.append({"t": killed_at, "alive": False, "procs": 1, "rss": 0, "hwm": 0,
                    "threads": 0, "fds": 0, "maps": 0, "cache_bytes": {},
                    "disk_free": int(300 * GB)})
    (project / "rounds.json").write_text(json.dumps(rounds))
    (tmp_path / "drift.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    return tmp_path


def test_a_dead_campaigns_zeros_are_not_a_reading(tmp_path):
    """A handle leak that ends in a kill must not read as handles settling.

    `long24` killed at trajectory 12: the sampler's next tick stamped `alive: false` with every
    counter at zero, the window holding the kill took their median with the live ones, and the
    verdict printed `open file handles 24 in trajectory 1 -> 12 in trajectory 10` and `12039
    mapped regions ... and not climbing` about a process that no longer existed.
    """
    drift, read = run_real(killed(tmp_path, fds_per_trajectory=8))
    assert any("open file handles" in d for d in drift), (drift, read)
    assert any("samples taken after the campaign died, dropped" in line for line in read), read
    assert not any("-> 0 in" in line or "-> 12 in" in line for line in read), read


def test_trajectories_that_finished_after_the_series_ended_are_not_judged(tmp_path):
    """The completion times come from the project folder, which a resumed leg keeps filling.

    `long24`'s series covered 9 trajectories and the folder held 16 by the time it was read, so
    the memory verdict divided its real rise by 15 boundaries it never watched and the handle
    verdict took medians of windows with no live sample in them at all.
    """
    out = killed(tmp_path, trajectories=9, watched=5, rss_per_trajectory_gb=1.0)
    _drift, read = run_real(out)
    assert any("finished after this series ended, not judged" in line for line in read), read
    judged = [line for line in read if "rss at trajectory boundaries" in line]
    assert judged and "+1.00 GB per trajectory" in judged[0], read


def test_a_campaign_whose_whole_series_is_dead_says_so(tmp_path):
    out = killed(tmp_path, watched=5)
    rows = [json.loads(line) for line in (out / "drift.jsonl").read_text().splitlines()]
    for row in rows:
        row["alive"] = False
    (out / "drift.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _drift, read = run_real(out)
    assert any("every sample was taken after the campaign died" in line for line in read), read
    assert not any("rss at trajectory boundaries" in line for line in read), read
    assert not any("open file handles" in line for line in read), read
