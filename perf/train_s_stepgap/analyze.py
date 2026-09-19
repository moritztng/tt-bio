#!/usr/bin/env python3
"""Read a run's history and split the step-to-step cadence into its three accounted parts.

`wall` times `step.step()`. `data` is the micro-batch build and upload in front of it, `outer`
is everything after. t[n+1] - t[n] == data[n+1] + wall[n+1] + outer[n+1] by construction, so the
residual column below is a check on the instrument and not a result: anything but ~0 means a
line of the loop is not inside any of the three.
"""
import json, statistics as st, sys
from pathlib import Path

def load(p, lo):
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if r["step"] >= lo]

def report(tag, path, lo):
    rows = load(path, lo)
    if len(rows) < 3:
        print(f"{tag}: only {len(rows)} settled rows"); return None
    w = [r["wall"] for r in rows]
    d = [r["data"] for r in rows]
    o = [r["outer"] for r in rows]
    t2t, resid = [], []
    for a, b in zip(rows, rows[1:]):
        if b["step"] != a["step"] + 1: continue
        gap = b["t"] - a["t"]
        t2t.append(gap)
        resid.append(gap - (b["data"] + b["wall"] + b["outer"]))
    m = st.median
    print(f"{tag}: n={len(rows)} (steps {rows[0]['step']}-{rows[-1]['step']})")
    print(f"   wall   mean {st.mean(w):7.3f}  med {m(w):7.3f}")
    print(f"   data   mean {st.mean(d):7.3f}  med {m(d):7.3f}   min {min(d):.3f} max {max(d):.3f}")
    print(f"   outer  mean {st.mean(o):7.3f}  med {m(o):7.3f}")
    print(f"   t2t    mean {st.mean(t2t):7.3f}  med {m(t2t):7.3f}")
    print(f"   GAP (data+outer) mean {st.mean(d)+st.mean(o):7.3f}  "
          f"= {100*(st.mean(d)+st.mean(o))/st.mean(t2t):.1f} % of cadence")
    print(f"   instrument residual mean {st.mean(resid):+.4f} s  max |{max(abs(x) for x in resid):.4f}|")
    return {"wall": st.mean(w), "data": st.mean(d), "outer": st.mean(o), "t2t": st.mean(t2t),
            "t2t_med": m(t2t), "n": len(rows)}

if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base = Path("/home/ttuser/.coworker/wt/train-s-stepgap/perf/train_s_stepgap")
    out = {}
    for arm in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for rk in (0, 1):
            f = base / arm / f"history-rank{rk}.jsonl"
            if f.exists():
                r = report(f"{arm} rank{rk}", f, lo)
                if r: out[f"{arm}-r{rk}"] = r
        print()
    if "A-r0" in out and "B-r0" in out:
        a, b = out["A-r0"], out["B-r0"]
        print(f"LEVER A->B: cadence {a['t2t']:.3f} -> {b['t2t']:.3f} s  "
              f"= {100*(a['t2t']/b['t2t']-1):+.1f} % more steps in the same wall clock")
