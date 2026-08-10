#!/usr/bin/env python3
"""Which `in0_block_w` is bit-exact at each newly-admitted class, and what it costs.

`admit.py` found four of twelve classes not `torch.equal`, all four at `in0_block_w=2`.
`_batched_matmul_block_w` predicts the width ttnn itself picks, and ttnn's all-DRAM 2D factory
derives it from the shape, so a rule calibrated at 298 aa can mispredict at 117 aa. Bit-exactness
needs the two calls to walk the same K blocks, so if any width is exact it is the one ttnn used.

For every class the census marks newly admitted: sweep every width dividing Kt, at the per_core_M
the chooser picks and at per_core_M=1, and report `torch.equal` plus us/call. Exactness should not
depend on per_core_M (it splits M, not K) -- both are measured so that is a checked fact.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

TRIALS, WARM = 3, 3
DT = {"DataType.FLOAT32": (ttnn.float32, "fp32", 4),
      "DataType.BFLOAT16": (ttnn.bfloat16, "bf16", 2)}

dev = T.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = tuple(int(v) for v in T.COMPUTE_GRID_MAIN)
CORES = GRID[0] * GRID[1]
L1 = int(ttnn.get_max_worker_l1_unreserved_size())
print(f"grid {GRID} = {CORES} cores, L1 {L1}", flush=True)


def tiles(n):
    return -(-n // 32)


def cfg_for(per_core_M, n_tiles, block_w):
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    sub_h = max(h for h in range(1, min(4 // sub_w, per_core_M) + 1) if per_core_M % h == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=block_w,
        out_subblock_h=sub_h, out_subblock_w=sub_w, per_core_M=per_core_M, per_core_N=n_tiles)


def timed(fn, reps):
    for _ in range(WARM):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps


classes: dict = {}
for path in sorted(glob.glob("perf/atomwindow_reconcile/census_*.json")):
    c = json.load(open(path))
    for r in c["rows"]:
        if r["gate_declined"]:
            key = (tuple(r["in0"]), tuple(r["in1"]), r["dtype"])
            classes.setdefault(key, {}).setdefault(
                f"{c['model']}/s{c['samples']}", r["calls"])

rows = []
for (sa, sb, dtname), calls in sorted(classes.items(), key=lambda kv: -max(kv[1].values())):
    dt, dn, eb = DT[dtname]
    batch = 1
    for d in sa[:-2]:
        batch *= d
    mt, kt, nt = tiles(sa[-2]), tiles(sa[-1]), tiles(sb[-1])
    rule_bw = T._batched_matmul_block_w(mt, kt, nt)
    chosen = T._batched_matmul_search(batch, mt, kt, nt, eb, GRID, L1)
    chosen_p = int(chosen.per_core_M) if chosen is not None else 1

    g = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(*sa, generator=g), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    y = ttnn.from_torch(torch.randn(*sb, generator=g), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    ref = ttnn.to_torch(ttnn.matmul(x, y, compute_kernel_config=CKC))
    one = timed(lambda: ttnn.matmul(x, y, compute_kernel_config=CKC), 3)
    reps = max(6, min(300, int(5e-3 / max(one, 1e-6))))
    naive_us = st.median([timed(lambda: ttnn.matmul(x, y, compute_kernel_config=CKC), reps)
                          for _ in range(TRIALS)]) * 1e6

    widths = {}
    for bw in [w for w in (1, 2, 4, 8, 16) if kt % w == 0]:
        entry = {}
        for pname, p in (("chosen", chosen_p), ("p1", 1)):
            # CB footprint, same model the shipped search uses.
            if 2 * (p + nt) * bw * 1024 * eb + p * nt * (1024 * eb + 4096) > L1:
                entry[pname] = {"skipped": "L1"}
                continue
            c = cfg_for(p, nt, bw)
            f = lambda c=c: ttnn.matmul(x, y, program_config=c, compute_kernel_config=CKC)
            ex = bool(torch.equal(ttnn.to_torch(f()), ref))
            us = st.median([timed(f, reps) for _ in range(TRIALS)]) * 1e6
            entry[pname] = {"exact": ex, "us": round(us, 2)}
        widths[bw] = entry

    exact_ws = [bw for bw, e in widths.items()
                if all(v.get("exact") for v in e.values() if "us" in v)
                and any("us" in v for v in e.values())]
    rows.append(dict(in0=list(sa), in1=list(sb), dtype=dn, batch=batch, m_tiles=mt, k_tiles=kt,
                     n_tiles=nt, blocks=batch * mt, rule_block_w=rule_bw, chosen_per_core_M=chosen_p,
                     naive_us=round(naive_us, 2), reps=reps, widths=widths,
                     exact_widths=exact_ws, rule_is_exact=rule_bw in exact_ws, calls=calls))
    print(f"{dn} {list(sa)}@{list(sb)} b={batch} Mt={mt} Kt={kt} Nt={nt} blocks={batch*mt} "
          f"rule_bw={rule_bw} chosen_pM={chosen_p} naive={naive_us:.2f}", flush=True)
    for bw, e in widths.items():
        print("    bw=%d %s" % (bw, "  ".join(
            f"{k}: " + (v["skipped"] if "skipped" in v else
                        f"{'EXACT' if v['exact'] else 'DIFF ':5s} {v['us']:8.2f}us")
            for k, v in e.items())), flush=True)
    print(f"    exact widths {exact_ws}  rule({rule_bw}) exact={rule_bw in exact_ws}", flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(y)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/width_qb1c0.json"
json.dump({"grid": list(GRID), "cores": CORES, "l1": L1, "rows": rows}, open(out, "w"), indent=2)
print("\nclasses where the rule's width is not bit-exact:",
      [(r["dtype"], r["in0"], r["in1"], r["rule_block_w"], r["exact_widths"])
       for r in rows if not r["rule_is_exact"]], flush=True)
print("wrote", out, flush=True)
