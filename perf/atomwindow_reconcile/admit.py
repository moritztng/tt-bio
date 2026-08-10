#!/usr/bin/env python3
"""Every class the relaxed gate newly admits: bit-exactness, per-call win, and per-fold price.

Driven by `census_*.json` rather than a hand-written class list, so the set measured here is
exactly the set a real fold issues. For each class:

  naive   the plain `ttnn.matmul` call, which is what main does today at these sizes
  bmm     `batched_matmul` with the relaxed gate, i.e. the arm under test
  p<N>    the same factory with per_core_M pinned to N, for every legal N

The p sweep answers the one thing relaxing the gate does not settle on its own: below `cores` the
chooser's `max(saturating)` rule picks the FEWEST blocks that still reach 32, which leaves cores
idle when the whole batch already fits the grid once. If a smaller per_core_M wins, the rule and
not just the gate needs to change.

Arms rotate per trial and the device is synced on both sides of every timed region. `torch.equal`
against `naive` on every arm, both dtypes.
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

TRIALS, WARM = 5, 3
DT = {"DataType.FLOAT32": (ttnn.float32, "fp32", 4),
      "DataType.BFLOAT16": (ttnn.bfloat16, "bf16", 2)}

dev = T.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = tuple(int(v) for v in T.COMPUTE_GRID_MAIN)
CORES = GRID[0] * GRID[1]
L1 = int(ttnn.get_max_worker_l1_unreserved_size())
print(f"grid {GRID} = {CORES} cores, L1/core {L1} B, "
      f"gate floor {T._BATCHED_MATMUL_SATURATION_BLOCKS} blocks", flush=True)


def tiles(n: int) -> int:
    return -(-n // 32)


def legal_p(batch, m_tiles, k_tiles, n_tiles, elem_bytes):
    """Every per_core_M the factory may safely use, same two escapes as the shipped search."""
    bw = T._batched_matmul_block_w(m_tiles, k_tiles, n_tiles)
    tile, acc = 1024 * elem_bytes, 4096
    out = []
    for p in range(1, m_tiles + 1):
        if m_tiles % p or (p != m_tiles and batch * m_tiles // p > CORES):
            continue
        if 2 * (p + n_tiles) * bw * tile + p * n_tiles * (tile + acc) > L1:
            continue
        out.append(p)
    return bw, out


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


# --- the class set, straight out of the census -------------------------------------------------
classes: dict = {}
for path in sorted(glob.glob("perf/atomwindow_reconcile/census_*.json")):
    c = json.load(open(path))
    for r in c["rows"]:
        if not r["gate_declined"]:
            continue
        key = (tuple(r["in0"]), tuple(r["in1"]), r["dtype"])
        e = classes.setdefault(key, {"calls": {}, "blocks": r["blocks"]})
        run = f"{c['model']}/{c['target'].split('/')[-1]}/s{c['samples']}"
        e["calls"][run] = max(e["calls"].get(run, 0), r["calls"])
print(f"{len(classes)} newly-admitted classes from the census", flush=True)

rows = []
for (sa, sb, dtname), info in sorted(classes.items(), key=lambda kv: -max(kv[1]["calls"].values())):
    dt, dn, eb = DT[dtname]
    batch = 1
    for d in sa[:-2]:
        batch *= d
    mt, kt, nt = tiles(sa[-2]), tiles(sa[-1]), tiles(sb[-1])
    bw, ps = legal_p(batch, mt, kt, nt, eb)
    g = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(*sa, generator=g), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    y = ttnn.from_torch(torch.randn(*sb, generator=g), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)

    chosen = T._batched_matmul_search(batch, mt, kt, nt, eb, GRID, L1)
    arms = {"naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
            "bmm": lambda: T.batched_matmul(x, y, compute_kernel_config=CKC)}
    for p in ps:
        c = cfg_for(p, nt, bw)
        arms[f"p{p}"] = (lambda c=c: ttnn.matmul(x, y, program_config=c, compute_kernel_config=CKC))

    ref = ttnn.to_torch(arms["naive"]())
    exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}

    one = timed(arms["naive"], 3)
    reps = max(6, min(300, int(5e-3 / max(one, 1e-6))))
    names = list(arms)
    samples = {k: [] for k in names}
    for t in range(TRIALS):
        for k in names[t % len(names):] + names[:t % len(names)]:
            samples[k].append(timed(arms[k], reps))
    us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
    saved_us = us["naive"] - us["bmm"]
    best_p = min((k for k in names if k.startswith("p")), key=lambda k: us[k])
    rows.append(dict(in0=list(sa), in1=list(sb), dtype=dn, batch=batch, m_tiles=mt, k_tiles=kt,
                     n_tiles=nt, blocks=batch * mt, in0_block_w=bw, legal_p=ps, reps=reps,
                     chosen=str(chosen), us=us, bit_exact=exact,
                     vs_naive={k: round(us["naive"] / us[k], 3) for k in names if k != "naive"},
                     saved_us=round(saved_us, 2), best_p=best_p,
                     best_p_saved_us=round(us["naive"] - us[best_p], 2),
                     calls=info["calls"],
                     saved_ms_per_fold={r: round(n * saved_us / 1e3, 1)
                                        for r, n in info["calls"].items()}))
    print(f"{dn} {list(sa)}@{list(sb)} b={batch} Mt={mt} Kt={kt} Nt={nt} blocks={batch*mt} "
          f"bw={bw} legal_p={ps} reps={reps}\n    naive={us['naive']:8.2f} bmm={us['bmm']:8.2f} "
          f"({us['naive']/us['bmm']:5.2f}x) " +
          " ".join(f"{k}={us[k]:.2f}" for k in names if k.startswith("p")) +
          f"\n    exact={exact} saved={saved_us:.2f} us/call -> " +
          ", ".join(f"{r} {n * saved_us / 1e3:.1f} ms/fold" for r, n in info["calls"].items()),
          flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(y)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/admit_qb1c0.json"
per_run: dict = {}
for r in rows:
    for run, ms in r["saved_ms_per_fold"].items():
        per_run[run] = round(per_run.get(run, 0.0) + ms, 1)
json.dump({"grid": list(GRID), "cores": CORES, "l1": L1,
           "gate_floor": T._BATCHED_MATMUL_SATURATION_BLOCKS,
           "priced_saving_ms_per_fold": per_run, "rows": rows}, open(out, "w"), indent=2)
print("\npriced saving per fold:", json.dumps(per_run), flush=True)
print("all bit-exact:", all(all(r["bit_exact"].values()) for r in rows), flush=True)
print("wrote", out, flush=True)
