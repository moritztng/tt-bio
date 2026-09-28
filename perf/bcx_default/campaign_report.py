#!/usr/bin/env python3
"""The bcx-default campaign arms: what a PD-L1 campaign costs at each default.

    campaign_report.py out/campaign_old out/campaign_auto [--json out/campaign_report.json]

The round is a screen, not the result. A round speedup does not transfer to the trajectory or
to the campaign on its own, so this reads the quantities the user actually waits for:

  * wall seconds per COMPLETED trajectory, which is `wall / trajectories returned`;
  * accepted designs, straight out of the campaign's own summary.csv;
  * chip-seconds per accepted design, the number that sits beside BoltzGen's 264.3. With zero
    accepted on an arm it is undefined and is reported as such rather than as a large number.

The amortised round over the window every slot was in a gradient round is printed beside them,
so the screen and the result can be compared on the same run.
"""
import csv
import json
import pathlib
import sys

H200 = 0.6958


def accepted(project: pathlib.Path):
    f = project / "summary.csv"
    if not f.exists():
        return None
    for row in csv.DictReader(f.open()):
        if row.get("metric") == "accepted_designs":
            return float(row["mean"])
    return None


def window(rows, slots):
    """First round start of the LAST slot to join, to the last round start of the first to
    leave. Rounds inside it are counted pro rata by the same rule as the round harness."""
    per = {s: sorted(r["t"] for r in rows if r["slot"] == s) for s in slots}
    per = {s: v for s, v in per.items() if len(v) >= 3}
    if not per:
        return None
    start = max(v[1] for v in per.values())          # each slot's round 1 dropped
    end = min(v[-1] for v in per.values())
    if end <= start:
        return None
    n = sum(1 for v in per.values() for t in v if start <= t < end)
    return {"start": start, "end": end, "wall": end - start, "rounds": n,
            "round_s": (end - start) / n if n else None, "slots": len(per)}


def one(path):
    p = pathlib.Path(path)
    run = json.loads((p / "run.json").read_text())
    rows = json.loads((p / "rounds.json").read_text())
    slots = sorted({r["slot"] for r in rows})
    clk = [c for _, c, _ in run.get("aiclk_samples", [])]
    load = [l for _, _, l in run.get("aiclk_samples", [])]
    hwm = max((r.get("vmhwm", 0) for r in rows), default=0)
    avail = min((r.get("mem_available", 0) for r in rows), default=0)
    got = run.get("trajectories_returned")
    acc = accepted(p)
    w = window(rows, slots)
    out = {"arm": p.name, "argument": run.get("trajectories_arg"),
           "resolved": run.get("trajectories") or len(slots),
           "auto_would_choose": run.get("auto_would_choose"),
           "budget": run.get("max_trajectories"), "commit": (run.get("commit") or "")[:9],
           "card": run.get("card"), "wall_s": run.get("wall_seconds"),
           "trajectories": got, "rounds": len(rows),
           "rounds_per_slot": run.get("rounds_per_slot"), "accepted": acc,
           "error": run.get("error"),
           "s_per_trajectory": (run["wall_seconds"] / got) if got else None,
           "s_per_accepted": (run["wall_seconds"] / acc) if acc else None,
           "host_hwm_gb": round(hwm / 2**30, 2), "mem_available_min_gb": round(avail / 2**30, 2),
           "aiclk_med": sorted(clk)[len(clk) // 2] if clk else None,
           "aiclk_min": min(clk) if clk else None,
           "aiclk_under_1200": sum(1 for c in clk if c < 1200),
           "aiclk_samples": len(clk),
           "load1_med": round(sorted(load)[len(load) // 2], 2) if load else None,
           "window": w}
    return out


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a != js]
    rows = [one(p) for p in paths if (pathlib.Path(p) / "run.json").exists()]
    for r in rows:
        w = r["window"] or {}
        print(f"{r['arm']:16s} arg={r['argument']:>4} -> {r['resolved']} on card  "
              f"budget {r['budget']}  wall {r['wall_s']:.0f} s  "
              f"trajectories {r['trajectories']}  rounds {r['rounds']}  "
              f"accepted {r['accepted']}")
        print(f"{'':16s} s/trajectory {r['s_per_trajectory'] and round(r['s_per_trajectory'], 1)}"
              f"  s/accepted {r['s_per_accepted'] and round(r['s_per_accepted'], 1)}"
              f"  amortised round {w.get('round_s') and round(w['round_s'], 3)} s over "
              f"{w.get('rounds')} rounds in {w.get('slots')} slots")
        print(f"{'':16s} AICLK med {r['aiclk_med']} min {r['aiclk_min']} "
              f"(<1200: {r['aiclk_under_1200']} of {r['aiclk_samples']})  load1 med "
              f"{r['load1_med']}  host HWM {r['host_hwm_gb']} GB  MemAvailable min "
              f"{r['mem_available_min_gb']} GB  error {r['error']}")
    if len(rows) == 2:
        a, b = rows
        if a["s_per_trajectory"] and b["s_per_trajectory"]:
            print(f"\nper completed trajectory: {a['s_per_trajectory']:.1f} s -> "
                  f"{b['s_per_trajectory']:.1f} s, "
                  f"{a['s_per_trajectory'] / b['s_per_trajectory']:.4f}x")
        if a["window"] and b["window"] and a["window"]["round_s"] and b["window"]["round_s"]:
            print(f"amortised round: {a['window']['round_s']:.3f} s "
                  f"({a['window']['round_s'] / H200:.2f}x H200) -> "
                  f"{b['window']['round_s']:.3f} s ({b['window']['round_s'] / H200:.2f}x), "
                  f"{a['window']['round_s'] / b['window']['round_s']:.4f}x")
        if a["accepted"] and b["accepted"]:
            print(f"chip-seconds per accepted design: {a['s_per_accepted']:.0f} -> "
                  f"{b['s_per_accepted']:.0f}")
        else:
            print("chip-seconds per accepted design: undefined on at least one arm "
                  f"(accepted {a['accepted']} and {b['accepted']} on a {a['budget']}-trajectory "
                  "budget)")
    if js:
        pathlib.Path(js).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
