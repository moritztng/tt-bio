#!/usr/bin/env python3
"""The cadence A/B, read off the arms' own history rows.

`wall` times `step.step()`, `data` is the micro-batch build and upload in front of it and `outer`
is everything after, so `t[n+1] - t[n] == data + wall + outer` by construction and the residual
column is a check on the instrument rather than a result. A is the host-built one-hot uploaded
over PCIe, B is the same one-hot expanded on the card.
"""
import json, statistics as st, sys
from pathlib import Path

BASE = Path("/home/ttuser/.coworker/wt/train-u-relpos-ondevice/perf/train_u_relpos")


def report(tag, path, lo):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["step"] >= lo]
    if len(rows) < 3:
        print(f"{tag}: only {len(rows)} settled rows")
        return None
    t2t, resid = [], []
    for a, b in zip(rows, rows[1:]):
        if b["step"] != a["step"] + 1:
            continue
        gap = b["t"] - a["t"]
        t2t.append(gap)
        resid.append(gap - (b["data"] + b["wall"] + b["outer"]))
    col = lambda k: [r[k] for r in rows]
    out = {k: st.mean(col(k)) for k in ("wall", "data", "outer")}
    out.update(t2t=st.mean(t2t), t2t_med=st.median(t2t), n=len(rows),
               data_med=st.median(col("data")),
               lo=rows[0]["step"], hi=rows[-1]["step"],
               digest=rows[-1]["digest"][:12], loss=rows[-1]["loss"])
    print(f"{tag}: n={out['n']} steps {out['lo']}-{out['hi']}  "
          f"data {out['data']:.3f} (med {out['data_med']:.3f})  wall {out['wall']:.3f}  "
          f"outer {out['outer']:.3f}  t2t {out['t2t']:.3f} (med {out['t2t_med']:.3f})  "
          f"resid {st.mean(resid):+.4f}/{max(abs(x) for x in resid):.4f}")
    return out


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = {}
    for arm in ("A1", "B1", "A2", "B2"):
        for rk in (0, 1):
            f = BASE / arm / f"history-rank{rk}.jsonl"
            if f.exists():
                r = report(f"{arm} rank{rk}", f, lo)
                if r:
                    res[f"{arm}-r{rk}"] = r
    print()
    for rk in (0, 1):
        a = [res[k] for k in (f"A1-r{rk}", f"A2-r{rk}") if k in res]
        b = [res[k] for k in (f"B1-r{rk}", f"B2-r{rk}") if k in res]
        if not a or not b:
            continue
        am, bm = st.mean([x["t2t"] for x in a]), st.mean([x["t2t"] for x in b])
        ad, bd = st.mean([x["data"] for x in a]), st.mean([x["data"] for x in b])
        aw, bw = st.mean([x["wall"] for x in a]), st.mean([x["wall"] for x in b])
        print(f"rank{rk}  A (host one-hot, {len(a)} arms) -> B (card one-hot, {len(b)} arms)")
        print(f"   data    {ad:7.3f} -> {bd:7.3f} s   {bd - ad:+.3f}")
        print(f"   wall    {aw:7.3f} -> {bw:7.3f} s   {bw - aw:+.3f}")
        print(f"   cadence {am:7.3f} -> {bm:7.3f} s   {100 * (am / bm - 1):+.1f} % more steps "
              f"in the same wall clock")
