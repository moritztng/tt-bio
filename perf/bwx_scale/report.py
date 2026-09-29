#!/usr/bin/env python3
"""Read a `perf/bwx_scale/chain.sh` sitting: what a trajectory costs, and whether a second chip
of the same Galaxy costs the first one anything.

    report.py OUT/one OUT/two        (any number of phase dirs)

Per chip it prints the unit the charter grades on -- seconds of ONE CHIP per completed
trajectory -- taken from the campaign's own gradient-round timestamps rather than its wall, so
model load and compilation are outside it. The AICLK is the median of the samples taken DURING
that span on the chip the campaign ran, from the sitting's own 1 Hz sampler; a round time
without the clock it was taken at is not a measurement.

With more than one chip in a phase, the per-chip figures are directly comparable with the
one-chip phase: chain.sh runs the same seed, the same binder and the same campaign on every
chip, so anything but a flat per-chip price is interference.
"""
import argparse
import json
import os
import pathlib
import statistics
import sys

SYSFS = pathlib.Path("/sys/class/tenstorrent")


def node_of(chip: int, mapping: dict | None) -> int | None:
    """The `/dev/tenstorrent/<node>` number of UMD chip `chip`. Read from the box's own sysfs
    when the report runs there; otherwise from a mapping the sitting recorded."""
    if mapping and str(chip) in mapping:
        return int(mapping[str(chip)])
    if not SYSFS.is_dir():
        return None
    nodes = sorted(SYSFS.iterdir(), key=lambda d: os.path.basename(os.path.realpath(d / "device")))
    return int(nodes[chip].name.split("!", 1)[1])


def clock(samples: list, node: int | None, t0: float, t1: float) -> dict:
    """AICLK of one chip over one campaign's gradient rounds, from the sitting's sampler."""
    mhz = [s["mhz"][str(node)] for s in samples
           if t0 <= s["t"] <= t1 and str(node) in s.get("mhz", {})]
    load = [s["load1"] for s in samples if t0 <= s["t"] <= t1]
    if not mhz:
        return {}
    return {"median": statistics.median(mhz), "min": min(mhz), "max": max(mhz),
            "samples": len(mhz), "under_1000": sum(1 for m in mhz if m < 1000),
            "load1_median": round(statistics.median(load), 2) if load else None}


def phase(d: pathlib.Path, chips_hint: int) -> list:
    samples = []
    jsonl = d / "aiclk.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except ValueError:
                pass
    sitting = json.loads((d / "sitting.json").read_text()) if (d / "sitting.json").exists() else {}
    mapping = sitting.get("nodes")
    rows = []
    for chipdir in sorted(d.glob("chip*")):
        chip = int(chipdir.name[4:])
        rounds_path = chipdir / "proj" / "rounds.json"
        if not rounds_path.exists():
            print(f"{d.name}/chip{chip}: no rounds.json yet", file=sys.stderr)
            continue
        rows_json = json.loads(rounds_path.read_text())
        slots: dict[str, list] = {}
        for r in rows_json:
            slots.setdefault(r["slot"], []).append(r["t"])
        t0, t1 = min(r["t"] for r in rows_json), max(r["t"] for r in rows_json)
        span = t1 - t0
        gaps = [b - a for ts in slots.values() for a, b in zip(ts, ts[1:])]
        run = {}
        if (chipdir / "proj" / "run.json").exists():
            run = json.loads((chipdir / "proj" / "run.json").read_text())
        rows.append({
            "phase": d.name, "chip": chip, "chips_in_phase": chips_hint,
            "trajectories": len(slots), "rounds": len(rows_json),
            "rounds_per_slot": {s: len(t) for s, t in sorted(slots.items())},
            "span_seconds": round(span, 1),
            "wall_seconds": run.get("wall_seconds"),
            "round_amortised": round(statistics.median(gaps) / len(slots), 2) if gaps else None,
            "seconds_per_trajectory": round(span / len(slots), 1),
            "trajectories_an_hour_one_chip": round(3600.0 / (span / len(slots)), 3),
            "accepted": run.get("trajectories_returned"),
            "aiclk": clock(samples, node_of(chip, mapping), t0, t1),
            "auto_chose": (run.get("auto_would_choose") or [None])[0],
            "commit": (run.get("commit") or "")[:9],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="+")
    ap.add_argument("--chips", type=int, default=32, help="chips on the Galaxy")
    ap.add_argument("--accept-rate", type=float, default=7 / 31,
                    help="accepted per completed trajectory; a property of BindCraft 2's "
                         "settings and the target, not of the board (Blackhole's measured rate)")
    args = ap.parse_args()

    out = []
    for p in args.phases:
        d = pathlib.Path(p)
        out += phase(d, len(sorted(d.glob("chip*"))))

    print(f"{'phase':6} {'chip':>4} {'n':>2} {'traj':>4} {'rounds':>6} {'span s':>8} "
          f"{'round s':>8} {'s/traj':>8} {'traj/h':>7} {'AICLK med/min':>14} {'<1000':>6}")
    for r in out:
        c = r["aiclk"]
        print(f"{r['phase']:6} {r['chip']:>4} {r['chips_in_phase']:>2} {r['trajectories']:>4} "
              f"{r['rounds']:>6} {r['span_seconds']:>8.0f} {str(r['round_amortised']):>8} "
              f"{r['seconds_per_trajectory']:>8.0f} {r['trajectories_an_hour_one_chip']:>7.3f} "
              f"{str(c.get('median', '?')) + '/' + str(c.get('min', '?')):>14} "
              f"{c.get('under_1000', '?'):>6}")

    one = [r for r in out if r["chips_in_phase"] == 1]
    many = [r for r in out if r["chips_in_phase"] > 1]
    if one and many:
        a = statistics.mean(r["trajectories_an_hour_one_chip"] for r in one)
        b = statistics.mean(r["trajectories_an_hour_one_chip"] for r in many)
        n = many[0]["chips_in_phase"]
        print(f"\nper-chip throughput on {n} chips is {b / a:.4f}x one chip's "
              f"({b:.3f} against {a:.3f} trajectories an hour a chip)")
        print(f"{n} chips together: {b * n:.3f} trajectories an hour, "
              f"against {a * n:.3f} if they did not interfere")
        print(f"a {args.chips}-chip Galaxy at the measured per-chip rate: "
              f"{b * args.chips:.0f} trajectories an hour")
        print(f"chip-hours per accepted design at {args.accept_rate:.3f} accepted per "
              f"completed trajectory: {1 / (b * args.accept_rate):.2f}")
    print()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
