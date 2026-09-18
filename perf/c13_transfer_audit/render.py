#!/usr/bin/env python3
import json
rows=json.load(open("transfer_table.json"))
KN={"replay":"replay gap","roof":"roof headroom","count":"count/rate model",
    "op_ab":"op/block A/B","fold_tx":"fold ratio transferred","novel":"novel mechanism"}
order={"fold":0,"fold(op x 560 calls)":0,"fold-stage":1,"in-situ fold s":2,"measured":2,
       "arith re-derive":3,"step":4,"block":5,"layer":5,"op":6}
rows.sort(key=lambda r:(order.get(r["level"],9), r["kind"], r["slug"]))
def fmt(v,u):
    if v is None: return "--"
    if u=="ratio-1": return f"{1+v:.5f}x"
    if u=="ratio":   return f"{v:.3f}"
    return f"{v:+.4f}" if u=="s" else f"{v:.4f}"
print("| # | slug | lever | screen kind | predicted | measured | unit | level | R | outcome | in band | source |")
print("|--:|---|---|---|--:|--:|---|---|--:|---|---|---|")
for i,r in enumerate(rows,1):
    band = "--" if r["inband"] is None else ("yes" if r["inband"] else "**no**")
    flag=""
    if not r.get("scored"): flag=" *(unscored)*"
    print(f"| {i} | `{r['slug']}` | {r['lever']}{flag} | {KN[r['kind']]} | {fmt(r['pred'],r['unit'])} | "
          f"{fmt(r['meas'],r['unit'])} | {r['unit']} | {r['level']} | "
          f"{'--' if r['Rv'] is None else f'{r[chr(82)+chr(118)]:.3f}'} | {r['outcome']} | {band} | `{r['cite']}` |")
