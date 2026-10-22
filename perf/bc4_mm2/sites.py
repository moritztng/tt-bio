#!/usr/bin/env python3
"""bc4-bet-mm2 grade: device ms per iteration saved at the two matmul sites, old route against new, one process.

A signature counts as rerouted by the number of calls per gradient iteration it lost at the hooked sites between
two shape censuses (perf/bc4_mm2/shapes.py): --base on the branch's base, --head on the branch. For each, on
random operands of the census shapes and dtypes:

* old  the call as the base issued it: `autograd._matmul` with split-K off, or `ttnn.linear` with the
       `_pair_proj_config` program config `tenstorrent._pair_proj_linear` falls back to
* new  what the branch runs instead: `autograd._matmul` with split-K on (a^T @ b), or `pair_mm.matmul`

Device time is a traced burst (--burst calls, --reps replays, min and median per call), AICLK read at every
replay. Accuracy: rel L2 against float64 of the same operands, both routes.
"""
import argparse, collections, json, os, statistics, subprocess, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--base", required=True); ap.add_argument("--head", required=True)
ap.add_argument("--burst", type=int, default=20); ap.add_argument("--reps", type=int, default=9)
A = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, autograd as AG, pair_mm as PM
dev = T.get_device(); cfg = AG.precise_config(); DRAM = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
PM.PAIR_MM_FUSED = True
clk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read())
DT = {"BFLOAT16": (ttnn.bfloat16, torch.bfloat16), "FLOAT32": (ttnn.float32, torch.float32)}


def key(r):
    return json.dumps([r["site"], r["a"]["shape"], r["a"]["dtype"], r["b"]["shape"], r["b"]["dtype"],
                       (r.get("bias") or {}).get("shape"), r.get("transpose_a"), r.get("transpose_b"),
                       r.get("extra"), r.get("out_dtype"), r.get("plan")])


def census(p):
    c, rows = collections.Counter(), {}
    for r in json.load(open(p))["rows"]:
        k = key(r); c[k] += r["calls_per_it"]; rows[k] = r
    return c, rows


def timed(f):
    ttnn.deallocate(f()); ttnn.synchronize_device(dev)
    outs = []; tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(A.burst):
        outs.append(f())
    ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
    ms, mhz = [], []
    for _ in range(A.reps):
        t0 = time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=False); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.burst); mhz.append(clk())
    ttnn.release_trace(dev, tid); [ttnn.deallocate(o) for o in outs]
    return dict(min=round(min(ms), 5), med=round(statistics.median(ms), 5), aiclk=mhz)


def rel(o, ref):
    got = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
    return round(float((got - ref).norm() / ref.norm()), 6)


bc, brows = census(A.base); hc, _ = census(A.head)
g = torch.Generator().manual_seed(7)
rows, tot = [], collections.Counter()
for k in sorted(bc, key=lambda k: -bc[k]):
    lost = bc[k] - hc.get(k, 0)
    r = brows[k]
    if lost <= 0:
        continue
    ta, tb = bool(r.get("transpose_a")), bool(r.get("transpose_b"))
    (ad, at), (bd, bt) = DT[r["a"]["dtype"]], DT[r["b"]["dtype"]]
    ha = torch.randn(*r["a"]["shape"], generator=g).to(at)
    kb = r["b"]["shape"][0 if not tb else 1] if not ta else r["b"]["shape"][0]
    hb = (torch.randn(*r["b"]["shape"], generator=g) / kb ** 0.5).to(bt)
    hbias = torch.randn(*r["bias"]["shape"], generator=g).bfloat16() if r.get("bias") else None
    up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    a, b = up(ha, ad), up(hb, bd); bias = up(hbias, ttnn.bfloat16) if hbias is not None else None
    A64 = ha.double().transpose(-2, -1) if ta else ha.double()
    B64 = hb.double().transpose(-2, -1) if tb else hb.double()
    ref = A64 @ B64 + (hbias.double() if hbias is not None else 0)
    odt = {"DataType.FLOAT32": ttnn.float32, "DataType.BFLOAT16": ttnn.bfloat16}.get(r.get("out_dtype"))
    if r["site"] == "_matmul":
        kw = dict(transpose_a=ta, transpose_b=tb, compute_kernel_config=cfg, **({"dtype": odt} if odt else {}))
        def old():
            AG.MM_SPLIT_K = False
            try:
                return AG._matmul(a, b, **kw)
            finally:
                AG.MM_SPLIT_K = True
        new = (lambda: AG._matmul(a, b, **kw)) if ta else (lambda: PM.matmul(a, b, compute_kernel_config=cfg, transpose_b=tb))
    else:
        pc = T._pair_proj_config(a, b) if "program_config" in (r.get("extra") or []) else None
        old = lambda: ttnn.linear(a, b, bias=bias, dtype=odt, memory_config=DRAM, compute_kernel_config=cfg,
                                  **({"program_config": pc} if pc is not None else {}))
        new = lambda: PM.matmul(a, b, bias, cfg, odt)
    s0 = dict(PM=PM.STATS[0], SK=AG.SPLIT_K["served"])
    o_new = new()
    served = (PM.STATS[0] - s0["PM"]) + (AG.SPLIT_K["served"] - s0["SK"])
    row = dict(site=r["site"], a=r["a"]["shape"], b=r["b"]["shape"], bias=hbias is not None, ta=ta, tb=tb,
               calls_lost=lost, calls_base=bc[k], served=served)
    if o_new is None or not served:
        row["note"] = "new route declined"; print(json.dumps(row), flush=True); rows.append(row); continue
    row["rel_new"] = rel(o_new, ref); row["rel_old"] = rel(old(), ref)
    row["old"] = timed(old); row["new"] = timed(new)
    row["save_ms_it"] = round(lost * (row["old"]["min"] - row["new"]["min"]), 3)
    tot[r["site"]] += row["save_ms_it"]
    print(json.dumps(row), flush=True); rows.append(row)
    for t in (a, b, bias):
        if t is not None:
            ttnn.deallocate(t)
    PM.forget()
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True).stdout.strip())
allclk = [c for r in rows for arm in ("old", "new") if arm in r for c in r[arm]["aiclk"]]
json.dump(dict(sha=sha, dirty=dirty, chip=CHIP, save_ms_it=dict(tot), save_ms_it_total=round(sum(tot.values()), 3),
               aiclk=dict(median=statistics.median(allclk) if allclk else None, min=min(allclk) if allclk else None),
               loadavg=open("/proc/loadavg").read().split()[:3], base=A.base, head=A.head, rows=rows),
          open(A.out, "w"), indent=1)
print(json.dumps(dict(save_ms_it=dict(tot))))
