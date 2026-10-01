#!/usr/bin/env python3
"""The drift verdict: does this campaign get slower, leakier or fuller as it runs?

A soak is only worth its hours if someone reads the series afterwards, and "the graph looks
flat" is not a verdict. This reads what `drift.py` sampled and what the campaign stamped per
gradient round, cuts the series at trajectory boundaries, and answers the four questions the
charter asks, each with the number it turned on:

  memory     host RSS at each trajectory boundary. bwx measured 4.4 GB per trajectory after the
             first on a 4-trajectory campaign (21.89 GB over the campaign), which is the shape
             this has to catch: not a spike inside a trajectory, but a floor that never comes
             back down. Judged on the boundary floors, so the within-trajectory peak is ignored.
  handles    open file descriptors, threads and mapped regions, first sample against last.
  disk       cache growth per trajectory, projected against the free space on the filesystem.
  speed      amortised seconds per round per trajectory, late trajectories against the median.
             "A slowdown arriving at trajectory 15" is the case, so the comparison is the LAST
             third against the median of the rest rather than first-against-last.

Exit 0 clean, 1 when something drifted, 2 when the files are not there, 3 when there are too
few finished trajectories to judge memory yet (never CLEAN by default).

  verdict.py <out_dir>            # out_dir holds drift.jsonl and project/rounds.json
"""
import argparse
import csv
import json
import pathlib
import statistics
import sys

GB = 1 << 30

#: A trajectory may hold more than the last one; a campaign whose floor climbs by this much per
#: trajectory, sustained, is leaking rather than breathing. bwx's 4.4 GB/trajectory is 8.8x this.
MEMORY_PER_TRAJECTORY_GB = 0.5
#: Handles and threads are allowed to settle, not to grow with the work.
HANDLE_GROWTH = 1.25
#: ... and settling can take several trajectories. Mapped regions on Blackhole climb in three
#: steps -- 13.3k, then 19.3k, 22.1k, 25.7k, one per MPNN redesign of trajectories 4, 7 and 12 --
#: and then hold 25.7k through the MPNN runs of trajectories 17 and 22. First-against-last
#: reads that saturating staircase as 1.93x growth and calls a warmed-up campaign leaky, which is
#: the same mistake the RSS slope made. So growth is judged on the LAST THIRD of the windows: a
#: staircase that has stopped climbing is warmup, one still climbing there is a leak.
HANDLE_SETTLED = 1.05
#: A campaign that keeps mapping regions dies when the kernel's per-process limit is reached, not
#: when memory runs out: `mmap` returns ENOMEM at `vm.max_map_count` with gigabytes still free.
MAP_LIMIT_PATH = "/proc/sys/vm/max_map_count"
#: Trajectories a still-climbing series has to survive before the limit stops being a soak risk.
MAP_HEADROOM_TRAJECTORIES = 200
#: A late trajectory this much slower than the median is a slowdown, not noise.
SLOWDOWN = 1.15
#: Trajectories finishing closer together than this are one event, not independent boundaries.
#: Interleaved arms start together and so finish together: bh24's first three finished within
#: 4.5 minutes of each other, at the moment the refold stage first loaded (+2 700 mapped regions,
#: +0.9 GB), and reading those as three floors divided one stage load by two and called it a
#: 1.37 GB/trajectory leak. A trajectory takes 20-40 minutes on either board.
MIN_WINDOW_S = 600
#: A trajectory boundary lands between two ticks of the 30 s sampler, so a window ending within
#: one tick of the last sample was still watched. Anything later was not.
SERIES_SLACK_S = 60


def read_samples(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass          # a sampler killed mid-write leaves one partial line; skip it
    return sorted(rows, key=lambda r: r["t"])


def trajectory_windows(rounds: list[dict]) -> list[dict]:
    """One window per trajectory, from the per-round stamps.

    `round` restarts at 1 for each trajectory a slot runs, which is the only boundary the round
    stamps carry: the campaign's own trajectory numbers live in its tables, and an interleaved
    arm's rounds are interleaved with the others in one file.
    """
    by_slot: dict[str, list[dict]] = {}
    for row in sorted(rounds, key=lambda r: r["t"]):
        by_slot.setdefault(row.get("slot", ""), []).append(row)
    windows = []
    for slot, rows in by_slot.items():
        current: list[dict] = []
        for row in rows:
            if row.get("round") == 1 and current:
                windows.append((slot, current))
                current = []
            current.append(row)
        if current:
            windows.append((slot, current))
    out = []
    for index, (slot, rows) in enumerate(sorted(windows, key=lambda w: w[1][0]["t"]), 1):
        span = rows[-1]["t"] - rows[0]["t"]
        out.append({"n": index, "slot": slot, "start": rows[0]["t"], "end": rows[-1]["t"],
                    "rounds": len(rows),
                    "s_per_round": span / (len(rows) - 1) if len(rows) > 1 else None})
    return out


def completion_times(project: pathlib.Path) -> list[float]:
    """When each finished trajectory finished, from the campaign's own record.

    The round stamps cannot say this. On a real interleaved campaign `round` does NOT restart at
    1 when an arm moves on to its next trajectory -- measured on qb1's bh24 on 2026-09-30: 5
    trajectories charged, `round == 1` exactly three times, all in the first four minutes, one per
    arm -- so windows cut there are arm windows, and the first version of this read three arms
    starting up during compile as three trajectory boundaries and reported a 0.98 GB/trajectory
    leak. The campaign's trajectory table has one row per FINISHED trajectory, and each names its
    folder under `1_Trajectories/`; that folder's newest file is when it finished.
    """
    for table in (project / "1_Trajectories" / "!_Trajectories.csv", project / "trajectories.csv"):
        if table.is_file():
            break
    else:
        return []
    try:
        with open(table, newline="") as f:
            names = [row.get("design") or row.get("trajectory") or "" for row in csv.DictReader(f)]
    except (OSError, csv.Error):
        return []
    out = []
    for name in names:
        folder = project / "1_Trajectories" / name
        stamps = [child.stat().st_mtime for child in folder.iterdir()] if folder.is_dir() else []
        if stamps:
            out.append(max(stamps))
    return sorted(out)


def boundary_windows(boundaries: list[float], rounds: list[dict],
                     samples: list[dict] | None = None) -> list[dict]:
    """One window per finished trajectory, cut at the campaign's own completion times.

    With N arms in flight a window is the stretch between two completions, not one trajectory's
    life, so its pace is the AMORTISED round: the window's duration over every arm's rounds in it.

    The windows are cut against whichever series starts EARLIER, the round stamps or the drift
    samples, because the round stamps can be younger than the campaign. A resumed campaign shares
    its project folder, and the harness rewrites `rounds.json` from an empty list, so after the
    box-lost drill restarted `long24` every stamp was newer than all ten trajectory completions:
    the leading edge sat after the last boundary, no window was cut, and a 4.21 h drift series
    with ten finished trajectories reported "0 trajectories" and no memory verdict at all. The
    samples were intact the whole time. Memory and handles are read off the samples anyway; only
    the pace needs round stamps, so a window without them carries `rounds: 0` and no pace rather
    than deleting the window.

    Each window also records how many arms stamped a round in it. Once the budget is spent no arm
    is refilled, so the last windows run on fewer arms: `bh24`'s last had two of three, and its
    amortised round rose 6.28 -> 7.48 s while each remaining arm got FASTER, 18.2 -> 8.5 s a round,
    because it had the chip to itself. That window is the campaign draining, not slowing down.
    """
    stamps = sorted(r["t"] for r in rounds)
    starts = [series[0] for series in (stamps, sorted(r["t"] for r in samples or [])) if series]
    if not boundaries or not starts:
        return []
    edges = [min(starts)]
    for when in sorted(boundaries):
        if when - edges[-1] < MIN_WINDOW_S and len(edges) > 1:
            edges[-1] = when          # the same event as the last boundary: move it, do not add
        elif when - edges[-1] >= MIN_WINDOW_S:
            edges.append(when)
    out = []
    for index, (start, end) in enumerate(zip(edges, edges[1:]), 1):
        inside = sum(1 for t in stamps if start < t <= end)
        arms = len({r.get("slot", "") for r in rounds if start < r["t"] <= end})
        out.append({"n": index, "slot": "all", "start": start, "end": end, "rounds": inside,
                    "arms": arms,
                    "s_per_round": (end - start) / inside if inside > 1 else None})
    return out


def at(samples: list[dict], when: float, key: str):
    """The sampled value nearest in time to `when`."""
    usable = [r for r in samples if r.get(key) is not None]
    if not usable:
        return None
    return min(usable, key=lambda r: abs(r["t"] - when))[key]


def window_median(samples: list[dict], window: dict, key: str):
    """The median of `key` over one trajectory's window, or None if nothing was sampled in it."""
    inside = [row[key] for row in samples
              if window["start"] <= row["t"] <= window["end"] and row.get(key) is not None]
    return statistics.median(inside) if inside else None


def map_limit(path: str = MAP_LIMIT_PATH) -> int | None:
    """The kernel's per-process mapping limit, or None on a host that does not publish one."""
    try:
        return int(pathlib.Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def map_limit_line(mapped: float, per_trajectory: float, path: str = MAP_LIMIT_PATH) -> str:
    """Where a mapping count sits against the limit that makes `mmap` fail with memory to spare."""
    limit = map_limit(path)
    if not limit:
        return f"{mapped:.0f} mapped regions; this host publishes no vm.max_map_count"
    share = f"{mapped:.0f} mapped regions is {mapped / limit:.0%} of vm.max_map_count {limit}"
    if per_trajectory <= 0:
        return f"{share}, and not climbing"
    room = (limit - mapped) / per_trajectory
    return (f"{share}, and at {per_trajectory:+.0f} per trajectory it reaches the limit in "
            f"{room:.0f} more trajectories, where mmap fails with host memory still free")


def watched(samples: list[dict], windows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """The live part of the series, and the trajectories it actually watched.

    Two ways a verdict gets cut over something the sampler never saw, both measured on `long24`
    after the box-lost drill killed it at trajectory 12 on 2026-09-30:

    * The sampler outlives the campaign by one tick and stamps what a dead process reports:
      `{"alive": false, "rss": 0, "threads": 0, "fds": 0, "maps": 0}`. Those zeros are not a
      reading. The window holding the kill takes their median with the live ones and halves it,
      so the verdict said `open file handles 24 in trajectory 1 -> 12 in trajectory 10` and
      `12039 mapped regions ... and not climbing` about a campaign that had been dead for a
      minute. A handle leak ending in a crash reads as handles settling -- clean, from a corpse.
    * The completion times come from the project FOLDER, which a resumed leg keeps filling. So a
      killed leg's verdict was cut over 16 boundaries when its series covered 9, and every
      boundary past the end snapped to the last live sample: a real 3.7 GB rise divided by 15
      trajectories instead of 9.

    So: drop the samples taken after the campaign died, and judge only the windows that end
    inside the live series (one sampling interval of slack, because a boundary lands between
    ticks).
    """
    live = [row for row in samples if row.get("alive", True)]
    if not live:
        # The pace is read off the round stamps, so every window still gets judged on it.
        return live, windows, ["every sample was taken after the campaign died"]
    note = []
    if len(live) < len(samples):
        note.append(f"{len(samples) - len(live)} samples taken after the campaign died, dropped")
    inside = [w for w in windows if w["end"] <= live[-1]["t"] + SERIES_SLACK_S]
    if len(inside) < len(windows):
        note.append(f"{len(windows) - len(inside)} of {len(windows)} finished trajectories "
                    f"finished after this series ended, not judged")
    return live, inside, note


def verdict(samples: list[dict], windows: list[dict]) -> tuple[list[str], list[str]]:
    drift, read = [], []
    if not samples:
        return ["no drift samples"], read
    ended = samples[-1].get("alive") is False
    samples, windows, note = watched(samples, windows)
    read.extend(note)

    # Memory, handles and disk are read off the samples; the pace is read off the round
    # stamps, so a campaign whose series is entirely dead still gets a pace verdict.
    if samples:
        first, last = samples[0], samples[-1]
        hours = (last["t"] - first["t"]) / 3600
        read.append(f"{len(samples)} samples over {hours:.2f} h, {len(windows)} trajectories")

        floors = [(w["n"], at(samples, w["end"], "rss")) for w in windows]
        floors = [(n, value) for n, value in floors if value]
        if len(floors) >= 3:
            rise_gb = (floors[-1][1] - floors[0][1]) / GB
            per = rise_gb / (len(floors) - 1)
            read.append(f"rss at trajectory boundaries {floors[0][1] / GB:.2f} -> "
                        f"{floors[-1][1] / GB:.2f} GB, {per:+.2f} GB per trajectory")
            if per > MEMORY_PER_TRAJECTORY_GB:
                drift.append(f"host memory climbs {per:.2f} GB per trajectory "
                             f"({floors[0][1] / GB:.2f} -> {floors[-1][1] / GB:.2f} GB over "
                             f"{len(floors)} trajectory boundaries); a campaign long enough will be "
                             f"OOM-killed")
        else:
            read.append("fewer than 3 trajectory boundaries: no memory-per-trajectory verdict")

        # Judged per trajectory, not first sample against last. The first sample is taken before the
        # campaign exists -- the launcher hands over a `timeout` wrapper holding one thread and three
        # handles -- and the minutes after that are compile workers, 30 processes and 358 handles on
        # Blackhole, which then go away. Against the wrapper every healthy campaign "leaks" 2408x;
        # against the compile peak a real leak hides. So: the median of the first trajectory's window
        # against the median of the last one's, the same shape as the memory floors.
        for key, what in (("fds", "open file handles"), ("threads", "threads"),
                          ("maps", "mapped regions")):
            if len(windows) < 2:
                read.append(f"fewer than 2 trajectory boundaries: no {what} verdict")
                continue
            series = [(w["n"], window_median(samples, w, key)) for w in windows]
            series = [(n, value) for n, value in series if value is not None]
            if len(series) < 2 or not series[0][1]:
                continue
            early, late = series[0][1], series[-1][1]
            read.append(f"{what} {early:.0f} in trajectory 1 -> {late:.0f} in "
                        f"trajectory {series[-1][0]}")
            # The last third against its own start: a staircase that stopped climbing is warmup.
            tail = series[-max(2, len(series) // 3):]
            still = tail[-1][1] > tail[0][1] * HANDLE_SETTLED
            if late > early * HANDLE_GROWTH and not still:
                read.append(f"  {what} settled: {tail[0][1]:.0f} in trajectory {tail[0][0]} -> "
                            f"{tail[-1][1]:.0f} in {tail[-1][0]}, so the {late / early:.2f}x is "
                            f"warmup rather than growth")
            elif still:
                per = (tail[-1][1] - tail[0][1]) / max(tail[-1][0] - tail[0][0], 1)
                drift.append(f"{what} is still growing at trajectory {tail[-1][0]}: {tail[0][1]:.0f} "
                             f"-> {tail[-1][1]:.0f} over the last {len(tail)} windows, {per:+.0f} per "
                             f"trajectory, after {early:.0f} in the first")
                if key == "maps":
                    drift.append(map_limit_line(late, per))
            if key == "maps" and not still:
                # Printed whether or not the count grew: a campaign already near the limit is a soak
                # risk even if this run's series is flat.
                read.append(f"  {map_limit_line(late, 0.0)}")

        caches = {name for row in samples for name in (row.get("cache_bytes") or {})}
        for name in sorted(caches):
            values = [(row["t"], (row.get("cache_bytes") or {}).get(name)) for row in samples]
            values = [(t, v) for t, v in values if v is not None]
            if len(values) < 2:
                continue
            grew = values[-1][1] - values[0][1]
            read.append(f"cache {name} {values[0][1] / GB:.3f} -> {values[-1][1] / GB:.3f} GB")
            free = last.get("disk_free")
            if grew > 0 and free and windows:
                per_trajectory = grew / max(len(windows), 1)
                if per_trajectory > 0:
                    room = free / per_trajectory
                    read.append(f"  at {per_trajectory / GB:.3f} GB per trajectory that fills the "
                                f"filesystem in {room:.0f} more trajectories")
                    if room < 100:
                        drift.append(f"cache {name} grows {per_trajectory / GB:.3f} GB per trajectory "
                                     f"and the filesystem holds only {room:.0f} more")

    paced = [w for w in windows if w["s_per_round"]]
    full = max((w.get("arms", 0) for w in paced), default=0)
    drain = []                  # only the TAIL: an arm in MPNN or validation stamps no rounds
    for w in reversed(paced):   # either, so a short-handed window mid-campaign is not a drain
        if w.get("arms", full) >= full:
            break
        drain.insert(0, w["n"])
    if drain:
        read.append(f"trajectory {drain} ran on fewer than {full} arms (the campaign draining), "
                    f"so its amortised round is not judged for speed")
        paced = [w for w in paced if w["n"] not in drain]
    if len(paced) >= 4:
        rates = [w["s_per_round"] for w in paced]
        late = rates[-max(1, len(rates) // 3):]
        early = rates[:len(rates) - len(late)]
        median = statistics.median(early)
        worst = max(late)
        read.append(f"amortised round {median:.2f} s median over the first {len(early)} "
                    f"trajectories, worst of the last {len(late)}: {worst:.2f} s")
        if worst > median * SLOWDOWN:
            slow = [w["n"] for w in paced[-len(late):] if w["s_per_round"] > median * SLOWDOWN]
            drift.append(f"the campaign slows down: trajectory {slow} run {worst:.2f} s a round "
                         f"against a median of {median:.2f} s ({worst / median:.2f}x)")
    else:
        read.append("fewer than 4 paced trajectories: no slowdown verdict")

    if ended:
        read.append("the campaign had ended by the last sample")
    return drift, read


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="the launcher's output dir: drift.jsonl + project/rounds.json")
    ap.add_argument("--drift")
    ap.add_argument("--rounds")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    drift_path = pathlib.Path(args.drift) if args.drift else out / "drift.jsonl"
    rounds_path = pathlib.Path(args.rounds) if args.rounds else out / "project" / "rounds.json"
    if not drift_path.is_file():
        print(f"no drift series at {drift_path}", file=sys.stderr)
        return 2
    samples = read_samples(drift_path)
    rounds = []
    if rounds_path.is_file():
        try:
            rounds = json.loads(rounds_path.read_text())
        except ValueError:
            rounds = []
    boundaries = completion_times(rounds_path.parent)
    if boundaries:
        windows = boundary_windows(boundaries, rounds, samples)
        source = f"{len(boundaries)} finished trajectories, from the campaign's trajectory table"
    else:
        windows = trajectory_windows(rounds)
        source = "round stamps (no trajectory table yet; on an interleaved run these are arms)"
    drifted, read = verdict(samples, windows)
    print(f"  boundaries: {source}")
    for line in read:
        print(f"  {line}")
    # The per-trajectory lines are the windows the verdict judged, not every completion in the
    # folder: a resumed leg keeps filling that folder, and printing its trajectories under a
    # killed leg's verdict reads as a campaign that ran longer than this series ever watched.
    for window in watched(samples, windows)[1]:
        pace = (f"{window['s_per_round']:.2f} s/round" if window["s_per_round"]
                else "one round" if window["rounds"] else "pace not recorded")
        print(f"  trajectory {window['n']} ({window['slot']}): {window['rounds']} rounds, {pace}")
    for line in drifted:
        print(f"DRIFT: {line}")
    if drifted:
        print(f"DRIFT FOUND {len(drifted)} THINGS")
        return 1
    # Clean is a claim about memory across trajectories; with too few boundaries to judge it,
    # saying CLEAN would be a pass-shaped reading of nothing (bh24 at 16:40Z said exactly that).
    if any("no memory-per-trajectory verdict" in line for line in read):
        print("DRIFT NOT YET JUDGED: too few finished trajectories to judge memory")
        return 3
    print("DRIFT CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
