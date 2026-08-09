#!/usr/bin/env python3
"""Offline re-analysis of a block_working_set.py --mode ledger JSON.

Same traffic model as the collector, but runs on the saved rows so the tables can be
regenerated without touching the card. Adds the per-phase working-set view W7 needs:
for every distinct device buffer, its size, its lifetime in op indices, and whether that
lifetime crosses a phase boundary (which is what decides pin vs chunk vs stream).
"""
import json, sys
from collections import defaultdict

R, W, MIX = 403.2e9, 268.3e9, 443.7e9
VIEW_OPS = {"ttnn.unsqueeze", "ttnn.squeeze", "ttnn.reshape", "ttnn.chunk"}

d = json.load(open(sys.argv[1]))
rows = d["rows"]
N, c_z = d["n"], d["c_z"]

def is_view(r):
    if r["op"] not in VIEW_OPS: return False
    ia = {i["addr"] for i in r["ins"] if i["addr"] >= 0}
    return bool(ia) and all(o["addr"] in ia for o in r["outs"] if o["addr"] >= 0)

def floor_ms(rd, wr):
    return max(rd/R, wr/W, (rd+wr)/MIX) * 1e3

def summarize(key):
    agg = defaultdict(lambda: {"n":0,"read":0.0,"write":0.0,"l1_out":0.0,"host_ms":0.0})
    for r in rows:
        a = agg[key(r)]; a["n"] += 1; a["host_ms"] += r["host_ms"]
        if is_view(r): continue
        for i in r["ins"]:
            if i["buffer"] == "DRAM": a["read"] += i["bytes"]
        for o in r["outs"]:
            if o["buffer"] == "DRAM": a["write"] += o["bytes"]
            else: a["l1_out"] += o["bytes"]
    return agg

PHASE_ORDER = ["trimul_start","trimul_end","tri_att_start","tri_att_end",
               "transition_z","attention_pair_bias","transition_s","residual"]
def pname(p):
    return {"triangle_multiplication_start":"trimul_start",
            "triangle_multiplication_end":"trimul_end",
            "triangle_attention_start":"tri_att_start",
            "triangle_attention_end":"tri_att_end"}.get(p, p)

print(f"# ledger n={N} c_z={c_z} ops={len(rows)}")
for title, key in [("BY PHASE", lambda r: pname(r["phase"])), ("BY OP", lambda r: r["op"])]:
    agg = summarize(key)
    print(f"\n=== {title} ===")
    print(f"{'name':<46} {'n':>4} {'read_MB':>9} {'write_MB':>9} {'l1out_MB':>9} {'floor_ms':>9}")
    tr = tw = 0.0
    for name, a in sorted(agg.items(), key=lambda kv: -(kv[1]['read']+kv[1]['write'])):
        tr += a["read"]; tw += a["write"]
        print(f"{name:<46} {a['n']:>4} {a['read']/1e6:>9.1f} {a['write']/1e6:>9.1f} "
              f"{a['l1_out']/1e6:>9.1f} {floor_ms(a['read'],a['write']):>9.3f}")
    print(f"{'TOTAL':<46} {len(rows):>4} {tr/1e6:>9.1f} {tw/1e6:>9.1f} {'':>9} "
          f"{floor_ms(tr,tw):>9.3f}")

# --- working set with lifetimes ---
ts = {}
for r in rows:
    for t in r["ins"] + r["outs"]:
        if t["addr"] < 0: continue
        k = (t["buffer"], t["addr"], round(t["bytes"]))
        e = ts.setdefault(k, {"buffer":t["buffer"],"bytes":t["bytes"],"shape":t["shape"],
                              "dtype":t["dtype"],"first":r["idx"],"last":r["idx"],
                              "touches":0,"phases":set()})
        e["last"] = r["idx"]; e["touches"] += 1; e["phases"].add(pname(r["phase"]))
ws = sorted(ts.values(), key=lambda e: -e["bytes"])
print("\n=== WORKING SET: distinct device buffers, largest first ===")
print(f"{'MB':>8} {'buf':>5} {'shape':<24} {'first':>6} {'last':>5} {'span':>5} {'touch':>6}  phases")
for e in ws[:36]:
    print(f"{e['bytes']/1e6:>8.2f} {e['buffer']:>5} "
          f"{'x'.join(str(x) for x in e['shape']):<24} {e['first']:>6} {e['last']:>5} "
          f"{e['last']-e['first']:>5} {e['touches']:>6}  {','.join(sorted(e['phases']))}")
print(f"distinct buffers={len(ws)} sum={sum(e['bytes'] for e in ws)/1e6:.1f} MB "
      f"(DRAM {sum(e['bytes'] for e in ws if e['buffer']=='DRAM')/1e6:.1f}, "
      f"L1 {sum(e['bytes'] for e in ws if e['buffer']=='L1')/1e6:.1f})")

# --- live-set high-water: how much is simultaneously live, by op index ---
events = []
for e in ws:
    events.append((e["first"], e["bytes"], e["buffer"]))
    events.append((e["last"]+1, -e["bytes"], e["buffer"]))
live = defaultdict(float); peak = defaultdict(float); peak_at = {}
cur = defaultdict(float)
for idx in range(len(rows)+1):
    for (i, b, bt) in events:
        if i == idx: cur[bt] += b
    for bt in ("DRAM","L1"):
        if cur[bt] > peak[bt]:
            peak[bt] = cur[bt]; peak_at[bt] = idx
print("\n=== LIVE-SET HIGH WATER (lifetime model, addr-reuse counted as one buffer) ===")
for bt in ("DRAM","L1"):
    print(f"  {bt}: peak {peak[bt]/1e6:.1f} MB at op {peak_at.get(bt)} "
          f"({rows[min(peak_at.get(bt,0), len(rows)-1)]['op']} in "
          f"{pname(rows[min(peak_at.get(bt,0), len(rows)-1)]['phase'])})")

# --- re-read census: DRAM buffers read more than once ---
reads = defaultdict(lambda: {"n":0,"bytes":0.0,"shape":None,"phases":set()})
for r in rows:
    if is_view(r): continue
    for i in r["ins"]:
        if i["buffer"] == "DRAM" and i["addr"] >= 0:
            k = (i["addr"], round(i["bytes"]))
            reads[k]["n"] += 1; reads[k]["bytes"] = i["bytes"]
            reads[k]["shape"] = i["shape"]; reads[k]["phases"].add(pname(r["phase"]))
print("\n=== DRAM RE-READS (same buffer read by >1 op) — the chunk/pin candidates ===")
print(f"{'MB':>8} {'reads':>6} {'MB_total':>9} {'shape':<24}  phases")
tot_extra = 0.0
for k, v in sorted(reads.items(), key=lambda kv: -(kv[1]['bytes']*(kv[1]['n']-1))):
    if v["n"] < 2: continue
    extra = v["bytes"]*(v["n"]-1); tot_extra += extra
    print(f"{v['bytes']/1e6:>8.2f} {v['n']:>6} {v['bytes']*v['n']/1e6:>9.1f} "
          f"{'x'.join(str(x) for x in v['shape']):<24}  {','.join(sorted(v['phases']))}")
print(f"redundant read bytes (all but the first read) = {tot_extra/1e6:.1f} MB "
      f"= {tot_extra/R*1e3:.2f} ms at the {R/1e9:.0f} GB/s read roof")
