#!/usr/bin/env python3
"""Read stub_aba.sh's three census arms: per-arm median wall, callbacks and host JAX over the warm
uninstrumented calls (calls 1-3 dropped: compile), load and clock, and stub minus mean(eager)."""
import json, statistics as st, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for arm in ("eager-a", "stub", "eager-b"):
    p = out / arm / "census.json"
    if not p.exists():
        print(f"{arm}: no census.json"); continue
    d = json.load(open(p))
    u = [r for r in d.get("uninstrumented", []) if r["n"] > 3 and not r.get("profiled")]
    if not u:
        print(f"{arm}: no warm calls"); continue
    med = lambda k: st.median(r[k] for r in u)
    cb = st.median(sum(r["callbacks"].values()) for r in u)
    hj = [r["host_jax_s"] for r in u]
    rows[arm] = dict(n=len(u), wall=med("s"), callbacks=cb, host_jax=st.median(hj), host_jax_range=[min(hj), max(hj)],
                     load1=st.median(r["load1"][0] if isinstance(r["load1"], (list, tuple)) else r["load1"] for r in u),
                     clock=d.get("cl"), sha=d.get("head", d).get("sha") if isinstance(d.get("head"), dict) else d.get("sha"))
    print(arm, json.dumps(rows[arm]))
if {"eager-a", "stub", "eager-b"} <= rows.keys():
    e = (rows["eager-a"]["host_jax"] + rows["eager-b"]["host_jax"]) / 2
    print(f"host JAX removed by the stub: {e - rows['stub']['host_jax']:.4f} s/call "
          f"(eager mean {e:.4f}, A drift {rows['eager-b']['host_jax'] - rows['eager-a']['host_jax']:+.4f})")
