#!/usr/bin/env python3
"""The trace sitting: eager against traced seams at the same N, same card, same sitting.

    report.py out/e1 out/t1 ... [--json out/report.json]

An arm is TRACED when its stamp's `seam_trace` counters show a replay. Per arm the tritraj
numbers (`perf/bcx_p10_tritraj/report.py`'s `one`: amortised round over the common window,
rounds 1 AND 2 dropped (see `_from_round_3`), pro rata, gate held/idle, AICLK median/min and load1 DURING, host high-water).
A serial arm has one trajectory, so its round is the mean warm round. Accuracy: every traced
(trajectory, round) digest against the eager arms' digest for the same key.
"""
import json
import pathlib
import statistics
import sys

_src = (pathlib.Path(__file__).resolve().parents[1] / "bcx_p10_tritraj" / "report.py").read_text()
_ns: dict = {"__name__": "tritraj_report"}
exec(compile(_src.rsplit("\nmain()", 1)[0], "tritraj/report.py", "exec"), _ns)
_rounds = _ns["rounds"]


def _from_round_3(ev, slot):
    """Rounds renumbered one down, so tritraj's `round >= 2` warm rule starts at round 3 here.
    Round 2 is where a traced arm primes every checkpoint and captures every trace, a one-time
    cost; both arms drop it alike."""
    return [(n - 1, t0, t1) for n, t0, t1 in _rounds(ev, slot) if n >= 2]


_ns["rounds"] = _from_round_3
one, rounds, clock, digests = _ns["one"], _ns["rounds"], _ns["clock"], _ns["digests"]
H200 = _ns["H200"]


def arm(path):
    r, d = one(path)
    st = d["stamp"].get("seam_trace") or {}
    r["traced"] = bool(st.get("replays_fwd"))
    r["seam_trace"] = st
    if not r["interleave"]:
        ev = d["events"]
        slots = sorted({e.get("slot") for e in ev if e["kind"] == "round_start"} - {None})
        warm = [x for s in slots for x in rounds(ev, s) if x[0] >= 2]
        if warm:
            r["round_s"] = statistics.mean(t1 - t0 for _, t0, t1 in warm)
            r["n"] = len(warm)
            r["clock"] = clock(d.get("aiclk", []), min(x[1] for x in warm),
                               max(x[2] for x in warm))
            held = sum((d["stamp"].get("gate") or {}).get("held_s", {}).values())
            total = sum(1 for e in ev if e["kind"] == "round_start")
            r["held_per_round_all"] = held / total if total else None
    return r, d


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a != js]
    rows, raw = [], {}
    for p in paths:
        if not (pathlib.Path(p) / "round_events.json").exists():
            print(f"{p}: no round_events.json")
            continue
        r, d = arm(p)
        rows.append(r)
        raw[r["arm"]] = d
    for r in rows:
        c = r.get("clock") or {}
        kind = "traced" if r["traced"] else "eager"
        if "round_s" not in r:
            print(f"{r['arm']:4s} N={r['trajectories']} {kind}  {r.get('error', 'no warm round')}")
            continue
        extra = (f"held {r['held_per_round']:.3f} idle {r['idle_per_round']:.3f} "
                 f"busy {r['gate_busy']:.1%} " if r["interleave"] else "")
        print(f"{r['arm']:4s} N={r['trajectories']} {kind:6s} round {r['round_s']:.3f} s  "
              f"{extra}n {r['n']:.2f}  AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} "
              f"load1 {c.get('load1')}  hwm {r['host_hwm_gb']} GB  trace {r['seam_trace']}")
    summary = {}
    for n in sorted({r["trajectories"] for r in rows}):
        grp = {}
        for kind in ("eager", "traced"):
            rs = [r for r in rows if r["trajectories"] == n and r["traced"] == (kind == "traced")
                  and "round_s" in r]
            if rs:
                grp[kind] = {"arms": [r["arm"] for r in rs],
                             "round_s": statistics.mean(r["round_s"] for r in rs),
                             "spread": [min(r["round_s"] for r in rs),
                                        max(r["round_s"] for r in rs)]}
                if all(r["interleave"] for r in rs):
                    grp[kind]["held_per_round"] = statistics.mean(r["held_per_round"] for r in rs)
                    grp[kind]["idle_per_round"] = statistics.mean(r["idle_per_round"] for r in rs)
        if len(grp) == 2:
            grp["speedup"] = grp["eager"]["round_s"] / grp["traced"]["round_s"]
        summary[f"N={n}"] = grp
        for kind, g in grp.items():
            if kind != "speedup":
                print(f"N={n} {kind}: {g['round_s']:.3f} s/round ({g['round_s'] / H200:.2f}x H200)"
                      f" spread {g['spread'][0]:.3f}-{g['spread'][1]:.3f} arms {g['arms']}")
        if "speedup" in grp:
            print(f"N={n} traced vs eager: {grp['speedup']:.4f}x (GO needs >= 1.05x)")
    ref = {}
    for r in rows:
        if not r["traced"]:
            for k, h in digests(raw[r["arm"]]).items():
                ref.setdefault(k, set()).add(h)
    same = diff = 0
    bad = []
    for r in rows:
        if r["traced"]:
            for k, h in digests(raw[r["arm"]]).items():
                if k in ref:
                    if ref[k] == {h}:
                        same += 1
                    else:
                        diff += 1
                        bad.append((r["arm"], k))
    summary["accuracy"] = {"traced_equal_eager": same, "differ": diff, "first_differ": bad[:6],
                           "eager_keys": len(ref),
                           "eager_self_consistent": all(len(v) == 1 for v in ref.values())}
    print(f"accuracy: {same} traced (trajectory, round) outputs equal eager, {diff} differ, "
          f"over {len(ref)} eager keys (eager self-consistent: "
          f"{summary['accuracy']['eager_self_consistent']})")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


main()
