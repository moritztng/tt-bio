#!/usr/bin/env python3
"""The bcx-p10-devtop sitting: N=3 stack (a*) against N=3 stack + TT_BIO_QKV_GRAD_JOIN (b*).

    report.py out/a1 out/b1 ... [--json out/report.json]

Per arm, `perf/bcx_p10_tritraj/report.py::one` (pro-rata amortised round, gate held/idle,
AICLK and loadavg1 DURING the window). Then the lever's speedup (mean a / mean b; GO >= 1.03x),
position-paired a_k against b_k in sitting order, and leg 4: every (trajectory, round) digest of
a b arm against the same key in the a arms.
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.bcx_p10_tritraj.report import H200, digests, one  # noqa: E402


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
        if "round_s" not in r:
            print(f"{r['arm']:4s}  {r.get('error', 'not interleaved')}  warm {r['warm']}")
            continue
        c = r["clock"]
        print(f"{r['arm']:4s}  round {r['round_s']:.3f} s  held {r['held_per_round']:.3f}  "
              f"idle {r['idle_per_round']:.3f}  busy {r['gate_busy']:.1%}  n {r['n']}  "
              f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')} load1 {c.get('load1')}  "
              f"hwm {r['host_hwm_gb']} GB")
    timed = [r for r in rows if "round_s" in r]
    arm = {k: [r for r in timed if r["arm"].startswith(k)] for k in "ab"}
    s = {}
    for k, rs in arm.items():
        if rs:
            s[k] = {"arms": [r["arm"] for r in rs],
                    "round_s": statistics.mean(r["round_s"] for r in rs),
                    "spread": [min(r["round_s"] for r in rs), max(r["round_s"] for r in rs)],
                    "held": statistics.mean(r["held_per_round"] for r in rs),
                    "idle": statistics.mean(r["idle_per_round"] for r in rs)}
            print(f"\n{k}: {s[k]['round_s']:.3f} s/round ({s[k]['round_s'] / H200:.2f}x H200)  "
                  f"spread {s[k]['spread'][0]:.3f}-{s[k]['spread'][1]:.3f}  held "
                  f"{s[k]['held']:.3f}  idle {s[k]['idle']:.3f}")
    if "a" in s and "b" in s:
        s["speedup"] = s["a"]["round_s"] / s["b"]["round_s"]
        pairs = list(zip(arm["a"], arm["b"]))
        s["paired"] = [round(a["round_s"] - b["round_s"], 3) for a, b in pairs]
        print(f"b vs a: {s['speedup']:.4f}x, {s['a']['round_s'] - s['b']['round_s']:+.3f} s "
              f"(GO needs >= 1.03x)  position-paired a_k - b_k {s['paired']}")
    ref = {}
    for r in arm["a"]:
        for key, h in digests(raw[r["arm"]]).items():
            ref.setdefault(key, set()).add(h)
    same = diff = 0
    for r in arm["b"]:
        for key, h in digests(raw[r["arm"]]).items():
            if key in ref:
                same, diff = (same + 1, diff) if ref[key] == {h} else (same, diff + 1)
    s["leg4"] = {"equal": same, "differ": diff, "a_keys": len(ref),
                 "a_self_consistent": all(len(v) == 1 for v in ref.values())}
    print(f"leg 4: {same} lever (trajectory, round) outputs equal the stack's, {diff} differ, "
          f"over {len(ref)} keys (stack arms agree among themselves: "
          f"{s['leg4']['a_self_consistent']})")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": s}, indent=1,
                                               default=str))


if __name__ == "__main__":
    main()
