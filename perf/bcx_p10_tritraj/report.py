#!/usr/bin/env python3
"""The tritraj sitting: N=2 vs N=3 interleaved trajectories on one card, full stack5 stack.

    report.py out/a1 out/b1 ... [--json out/report.json]

Per interleaved arm: the amortised round over the window in which every trajectory ran (round
1 of each dropped, rounds counted PRO RATA, `perf/bcx_p10_duotraj/report.py`'s rule), the gate
held and gate idle per round inside that window (read off the `gate` hold events, which never
overlap), each trajectory's lock wait per round, AICLK median/min and loadavg1 sampled DURING
the window off the card's class node, and host high-water. A serial arm is only the leg-4
reference: every interleaved (trajectory, round) digest must equal it.
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
           "host_hwm_gb": round(max(rss) / 1e9, 2) if rss else None,
           "stopped": st.get("stopped"), "gate": g}
    total = {s: sum(1 for e in ev if e["kind"] == "round_start" and e.get("slot") == s)
             for s in slots}
    out["waited_per_round"] = {s: round(v / total[s], 3) for s, v in
                               (g.get("waited_s") or {}).items() if total.get(s)}
    if not out["interleave"]:
        return out, d
    if len(slots) < 2 or any(not warm[s] for s in slots):
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
               held_per_round=held / n, idle_per_round=(wall - held) / n,
               gate_busy=held / wall, clock=clock(d.get("aiclk", []), start, end))
    return out, d


def digests(d):
    return {(e["slot"], e["round"]): e["sha256"] for e in d["events"] if e["kind"] == "digest"}


def leg4(ref_arms, arms, raw):
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
        if not r["interleave"] or "error" in r:
            print(f"{head}  {r.get('error', '')}  warm {r['warm']}  hwm {r['host_hwm_gb']} GB")
            continue
        c = r["clock"]
        print(f"{head}  round {r['round_s']:.3f} s  held {r['held_per_round']:.3f}  "
              f"idle {r['idle_per_round']:.3f}  busy {r['gate_busy']:.1%}  n {r['n']}  "
              f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} load1 {c.get('load1')}  "
              f"hwm {r['host_hwm_gb']} GB  waited/round {r['waited_per_round']}")
    summary = {"by_n": {}}
    timed = [r for r in rows if r["interleave"] and "round_s" in r]
    for n in sorted({r["trajectories"] for r in timed}):
        rs = [r for r in timed if r["trajectories"] == n]
        m = statistics.mean(r["round_s"] for r in rs)
        summary["by_n"][n] = {
            "arms": [r["arm"] for r in rs], "round_s": m, "ratio_vs_h200": m / H200,
            "spread": [min(r["round_s"] for r in rs), max(r["round_s"] for r in rs)],
            "held_per_round": statistics.mean(r["held_per_round"] for r in rs),
            "idle_per_round": statistics.mean(r["idle_per_round"] for r in rs)}
        s = summary["by_n"][n]
        print(f"\nN={n}: {m:.3f} s/round ({m / H200:.2f}x H200), spread "
              f"{s['spread'][0]:.3f}-{s['spread'][1]:.3f}, held {s['held_per_round']:.3f}, "
              f"idle {s['idle_per_round']:.3f}")
    if 2 in summary["by_n"] and 3 in summary["by_n"]:
        sp = summary["by_n"][2]["round_s"] / summary["by_n"][3]["round_s"]
        summary["speedup_3_over_2"] = sp
        print(f"N=3 vs N=2: {sp:.4f}x  (GO needs >= 1.03x; bar {10 * H200:.3f} s)")
    ser = [r for r in rows if not r["interleave"]]
    inter = [r for r in rows if r["interleave"]]
    if ser and inter:
        summary["leg4"] = leg4(ser, inter, raw)
        l4 = summary["leg4"]
        print(f"leg 4: {l4['equal_to_serial']} interleaved (trajectory, round) outputs equal "
              f"serial, {l4['differ']} differ, over {l4['serial_keys']} serial keys")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


main()
