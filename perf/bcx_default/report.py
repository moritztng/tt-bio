#!/usr/bin/env python3
"""The bcx-default sitting: the round a BindCraft 2 user gets, old default against new.

    report.py out/a1 out/b1 ... [--json out/report.json]

`perf/bcx_p10_tritraj/report.py` times interleaved arms only, because a serial arm was its
leg-4 digest reference and never a result. Here the serial arm IS one of the two results: the
old default is one trajectory with no gate, so it has one slot and no gate events, and it still
has to be timed by the same rule as the arm it is compared with.

Both arms: round 1 dropped, rounds counted PRO RATA over the window in which every slot was
running, AICLK and loadavg sampled DURING that window off the card's class node, host
high-water from the round boundaries.
"""
import json
import pathlib
import statistics
import sys

H200 = 0.6958


def rounds(ev, slot):
    marks = sorted(((e["round"], e["t0"], e["kind"]) for e in ev
                    if e["kind"] in ("round_start", "round_stop") and e.get("slot") == slot),
                   key=lambda m: m[1])
    return [(n, t0, t1) for (n, t0, k), (_, t1, _k) in zip(marks, marks[1:])
            if k == "round_start"]


def clock(samples, t0, t1):
    got = [(c, l) for t, c, l in samples if t0 <= t <= t1]
    if not got:
        return {"n": 0}
    clk = sorted(c for c, _ in got)
    return {"n": len(clk), "aiclk_med": clk[len(clk) // 2], "aiclk_min": clk[0],
            "under_1200": sum(1 for c in clk if c < 1200),
            "load1": round(sum(l for _, l in got) / len(got), 2)}


def clipped(ev, kind, t0, t1):
    return sum(max(0.0, min(e["t1"], t1) - max(e["t0"], t0)) for e in ev if e["kind"] == kind)


def one(path):
    d = json.loads((pathlib.Path(path) / "round_events.json").read_text())
    st, ev = d["stamp"], d["events"]
    slots = sorted({e.get("slot") for e in ev if e["kind"] == "round_start"} - {None})
    warm = {s: [r for r in rounds(ev, s) if r[0] >= 2] for s in slots}
    rss = [e["reach"].get("vmhwm", 0) for e in ev if e["kind"] in ("round_start", "round_stop")
           and isinstance(e.get("reach"), dict)]
    g = st.get("gate") or {}
    out = {"arm": pathlib.Path(path).name, "interleave": bool(st.get("interleave")),
           "trajectories": st.get("trajectories", len(slots)), "commit": st.get("commit", "")[:9],
           "card": st.get("card"), "warm": {s: len(warm[s]) for s in slots},
           "host_hwm_gb": round(max(rss) / 1e9, 2) if rss else None, "gate": g}
    auto = pathlib.Path(path) / ".." / f"{pathlib.Path(path).name}"
    resolved = pathlib.Path(path).parent / pathlib.Path(path).name / "auto.json"
    if resolved.exists():
        out["auto"] = json.loads(resolved.read_text())
    if not slots or any(not warm[s] for s in slots):
        out["error"] = "a slot has no warm round"
        return out, d
    start = max(warm[s][0][1] for s in slots)
    end = min(warm[s][-1][2] for s in slots)
    if end <= start:
        out["error"] = "no common window"
        return out, d

    def share(r):
        return max(0.0, min(r[2], end) - max(r[1], start)) / (r[2] - r[1])
    n = sum(share(r) for s in slots for r in warm[s])
    wall = end - start
    held = clipped(ev, "gate", start, end)
    out.update(round_s=wall / n, n=round(n, 2), window_s=round(wall, 2),
               # Seconds of wall per COMPLETED trajectory-round, which is what a trajectory
               # actually advances at: the amortised round times how many share the card.
               per_trajectory_round_s=wall / n * len(slots),
               held_per_round=held / n if held else None,
               idle_per_round=(wall - held) / n if held else None,
               gate_busy=held / wall if held else None,
               clock=clock(d.get("aiclk", []), start, end))
    return out, d


def digests(d):
    return {(e["slot"], e["round"]): e["sha256"] for e in d["events"] if e["kind"] == "digest"}


def equality(ref_arms, arms, raw):
    ref = {}
    for r in ref_arms:
        for k, h in digests(raw[r["arm"]]).items():
            ref.setdefault(k, set()).add(h)
    same = diff = 0
    bad = []
    for r in arms:
        for k, h in digests(raw[r["arm"]]).items():
            if k not in ref:
                continue
            if ref[k] == {h}:
                same += 1
            else:
                diff += 1
                bad.append((r["arm"], k))
    return {"equal_to_serial": same, "differ": diff, "first_differ": bad[:6],
            "serial_keys": len(ref)}


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a != js]
    rows, raw = [], {}
    for p in paths:
        if not (pathlib.Path(p) / "round_events.json").exists():
            print(f"{p}: no round_events.json")
            continue
        r, d = one(p)
        rows.append(r)
        raw[r["arm"]] = d
    for r in rows:
        head = f"{r['arm']:4s} N={r['trajectories']} {'inter' if r['interleave'] else 'serial'}"
        if "error" in r:
            print(f"{head}  {r['error']}  warm {r['warm']}  hwm {r['host_hwm_gb']} GB")
            continue
        c, held = r["clock"], r["held_per_round"]
        print(f"{head}  round {r['round_s']:.3f} s  traj-round {r['per_trajectory_round_s']:.3f}  "
              f"held {held if held is None else round(held, 3)}  n {r['n']}  "
              f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} "
              f"(<1200: {c.get('under_1200')}) load1 {c.get('load1')}  "
              f"hwm {r['host_hwm_gb']} GB")
    summary = {"by_n": {}}
    timed = [r for r in rows if "round_s" in r]
    for n in sorted({r["trajectories"] for r in timed}):
        rs = [r for r in timed if r["trajectories"] == n]
        m = statistics.median(r["round_s"] for r in rs)
        summary["by_n"][n] = {
            "arms": [r["arm"] for r in rs], "round_s": m, "ratio_vs_h200": m / H200,
            "spread": [min(r["round_s"] for r in rs), max(r["round_s"] for r in rs)],
            "per_trajectory_round_s": statistics.median(
                r["per_trajectory_round_s"] for r in rs),
            "host_hwm_gb": max(r["host_hwm_gb"] or 0 for r in rs),
            "aiclk_med": statistics.median(r["clock"].get("aiclk_med", 0) for r in rs),
            "aiclk_min": min(r["clock"].get("aiclk_min", 0) for r in rs),
            "load1": statistics.median(r["clock"].get("load1", 0) for r in rs)}
        s = summary["by_n"][n]
        print(f"\nN={n}: {m:.3f} s/round ({m / H200:.2f}x H200), spread "
              f"{s['spread'][0]:.3f}-{s['spread'][1]:.3f}, per-trajectory round "
              f"{s['per_trajectory_round_s']:.3f} s, hwm {s['host_hwm_gb']} GB, "
              f"AICLK {s['aiclk_med']}/{s['aiclk_min']}, load1 {s['load1']}")
    ns = sorted(summary["by_n"])
    if 1 in ns:
        for n in ns[1:]:
            sp = summary["by_n"][1]["round_s"] / summary["by_n"][n]["round_s"]
            summary[f"speedup_{n}_over_1"] = sp
            print(f"\nthe default, N=1 -> N={n}: {sp:.4f}x a round "
                  f"({summary['by_n'][1]['ratio_vs_h200']:.2f}x H200 -> "
                  f"{summary['by_n'][n]['ratio_vs_h200']:.2f}x)")
    ser = [r for r in rows if not r["interleave"] and r["trajectories"] > 1]
    inter = [r for r in rows if r["interleave"]]
    if ser and inter:
        summary["equality"] = equality(ser, inter, raw)
        e = summary["equality"]
        print(f"\nequality: {e['equal_to_serial']} interleaved (trajectory, round) outputs equal "
              f"the same trajectory run serially, {e['differ']} differ, over "
              f"{e['serial_keys']} serial keys")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


if __name__ == "__main__":
    main()
