#!/usr/bin/env python3
"""The milestone sitting: serial vs duo on one card with every GO lever armed.

    report.py out/s1 out/d1 ... [--json out/report.json]

Per arm: the warm rounds (round 1 of each trajectory dropped), the device column (the card's
seam seconds per round), AICLK median/min and loadavg1 sampled DURING the warm rounds off the
card's class node, and host RSS at the round boundaries. A serial arm's round is its median warm
round; a duo arm's is the common window divided by the rounds inside it counted PRO RATA
(`perf/bcx_p10_duotraj/report.py`'s rule). Then leg 4: every (trajectory, round) digest of
`sequence_gradients`' output in a duo arm against the same trajectory and round serial.
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bcx_p10_duotraj"))
_argv, sys.argv = sys.argv, sys.argv[:1]
from report import clock, rounds  # noqa: E402
sys.argv = _argv

H200 = 0.6958


def device_in(ev, t0, t1):
    """Seam seconds inside [t0, t1], clipped at the edges."""
    s = 0.0
    for e in ev:
        if e["kind"] == "device":
            s += max(0.0, min(e["t1"], t1) - max(e["t0"], t0))
    return s


def one(path):
    d = json.loads((pathlib.Path(path) / "round_events.json").read_text())
    st, ev = d["stamp"], d["events"]
    slots = sorted({e.get("slot") for e in ev if e["kind"] == "round_start"} - {None})
    warm = {s: [r for r in rounds(ev, s) if r[0] >= 2] for s in slots}
    rss = [e["reach"].get("vmhwm", 0) for e in ev if e["kind"] in ("round_start", "round_stop")
           and isinstance(e.get("reach"), dict)]
    out = {"arm": pathlib.Path(path).name, "interleave": bool(st.get("interleave")),
           "commit": st.get("commit", "")[:9], "card": st.get("card"),
           "warm": {s: len(warm[s]) for s in slots},
           "host_hwm_gb": round(max(rss) / 1e9, 2) if rss else None,
           "gate": st.get("gate"), "stopped": st.get("stopped")}
    allw = [r for s in slots for r in warm[s]]
    if not allw:
        out["error"] = "no warm rounds"
        return out, d
    if not out["interleave"]:
        walls = [t1 - t0 for _n, t0, t1 in allw]
        devs = [device_in(ev, t0, t1) for _n, t0, t1 in allw]
        out.update(round_s=statistics.median(walls), device_s=statistics.median(devs),
                   n=len(walls), wall_min=min(walls), wall_max=max(walls))
        t0 = min(r[1] for r in allw); t1 = max(r[2] for r in allw)
        # Only the seconds inside warm rounds: the gap between A's last and B's first is the
        # second trajectory's compile and is not a round.
        samples = [x for x in d.get("aiclk", []) if any(a <= x[0] <= b for _n, a, b in allw)]
        out["clock"] = clock(samples, t0, t1)
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
    dev = device_in(ev, start, end)
    out.update(round_s=wall / n, device_s=dev / n, n=round(n, 2), window_s=round(wall, 2),
               card_busy=round(dev / wall, 4), clock=clock(d.get("aiclk", []), start, end))
    return out, d


def digests(d):
    return {(e["slot"], e["round"]): e["sha256"] for e in d["events"] if e["kind"] == "digest"}


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [p for p in paths if p != js]
    rows, raw = [], {}
    for p in paths:
        if not (pathlib.Path(p) / "round_events.json").exists():
            print(f"{p}: no round_events.json"); continue
        r, d = one(p); rows.append(r); raw[r["arm"]] = d
    for r in rows:
        c = r.get("clock", {})
        print(f"{r['arm']:4s} {'duo' if r['interleave'] else 'ser'} "
              + (r.get("error") or
                 f"round {r['round_s']:.3f} s  device {r['device_s']:.3f} s  n {r['n']}  "
                 f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} "
                 f"load1 {c.get('load1')}  hwm {r['host_hwm_gb']} GB"
                 + (f"  busy {r['card_busy']:.3f}" if r['interleave'] else "")))
    ser = [r for r in rows if not r["interleave"] and "round_s" in r]
    duo = [r for r in rows if r["interleave"] and "round_s" in r]
    summary = {}
    if ser and duo:
        s = statistics.mean(r["round_s"] for r in ser)
        dd = statistics.mean(r["round_s"] for r in duo)
        summary = {"serial_round_s": s, "duo_round_s": dd, "speedup": s / dd,
                   "ratio_vs_h200": dd / H200, "bar_s": 10 * H200,
                   "serial_device_s": statistics.mean(r["device_s"] for r in ser),
                   "duo_device_s": statistics.mean(r["device_s"] for r in duo)}
        print(f"\nserial {s:.3f}  duo {dd:.3f}  speedup {s/dd:.4f}x  "
              f"RATIO {dd/H200:.2f}x vs H200 (bar {10*H200:.3f} s)")
    # Leg 4: every duo digest against every serial arm's digest for the same (slot, round).
    ref = {}
    for r in ser:
        for k, h in digests(raw[r["arm"]]).items():
            ref.setdefault(k, set()).add(h)
    same = diff = 0; bad = []
    for r in duo:
        for k, h in digests(raw[r["arm"]]).items():
            if k not in ref:
                continue
            if h in ref[k] and len(ref[k]) == 1:
                same += 1
            else:
                diff += 1; bad.append((r["arm"], k))
    serial_self = sum(1 for v in ref.values() if len(v) > 1)
    summary["leg4"] = {"duo_equal_to_serial": same, "differ": diff, "first_differ": bad[:6],
                       "serial_keys_disagreeing_across_serial_arms": serial_self,
                       "serial_keys": len(ref)}
    print(f"leg 4: {same} duo (trajectory, round) outputs equal to serial, {diff} differ; "
          f"serial arms disagree among themselves on {serial_self} of {len(ref)} keys")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


main()
