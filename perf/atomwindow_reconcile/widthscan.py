#!/usr/bin/env python3
"""Is main's `_batched_matmul_block_w` bit-exact at token counts nobody pinned?

The DiT attention classes are `[1, H, T, dh]` and `[1, H, T, T]` for T = the padded token count, so
the class set a fold issues is a function of the INPUT LENGTH. `_batched_matmul_block_w` is a
closed-form rule fitted to the classes at 117 aa and 298 aa. If the exact width varies with Kt in a
way the rule does not track, then some other protein length silently gets a non-bit-exact config
from the merged helper, at every size where the gate already applies.

This sweeps T and, for each class, reports: does main's gate apply (blocks >= cores), what width
does the rule predict, and is that width `torch.equal` against the plain call.
"""
from __future__ import annotations

import json
import sys

import torch
import ttnn

import tt_bio.tenstorrent as T_

dev = T_.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = tuple(int(v) for v in T_.COMPUTE_GRID_MAIN)
CORES = GRID[0] * GRID[1]
L1 = int(ttnn.get_max_worker_l1_unreserved_size())
print(f"grid {GRID} = {CORES} cores", flush=True)

H, DH = 16, 64
TOKENS = [117, 160, 192, 224, 256, 298, 352, 416, 480]


def tiles(n):
    return -(-n // 32)


rows = []
for dt, dn, eb in ((ttnn.float32, "fp32", 4), (ttnn.bfloat16, "bf16", 2)):
    for tok in TOKENS:
        g = torch.Generator().manual_seed(0)
        q = ttnn.from_torch(torch.randn(1, H, tok, DH, generator=g), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=dt)
        kT = ttnn.from_torch(torch.randn(1, H, DH, tok, generator=g), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=dt)
        a = ttnn.from_torch(torch.randn(1, H, tok, tok, generator=g), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=dt)
        v = ttnn.from_torch(torch.randn(1, H, tok, DH, generator=g), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=dt)
        for lbl, x, y in (("q@kT", q, kT), ("attn@v", a, v)):
            sa, sb = tuple(int(d) for d in x.shape), tuple(int(d) for d in y.shape)
            mt, kt, nt = tiles(sa[-2]), tiles(sa[-1]), tiles(sb[-1])
            blocks = H * mt
            bw = T_._batched_matmul_block_w(mt, kt, nt)
            ref = ttnn.to_torch(ttnn.matmul(x, y, compute_kernel_config=CKC))
            got = ttnn.to_torch(T_.batched_matmul(x, y, compute_kernel_config=CKC))
            exact = bool(torch.equal(got, ref))
            # which widths ARE exact, so a misprediction can be named and not just flagged
            ex_ws, oom = [], []
            chosen = T_._batched_matmul_search(H, mt, kt, nt, eb, GRID, L1)
            p = int(chosen.per_core_M) if chosen is not None else mt
            for w in [w for w in (1, 2, 4, 8) if kt % w == 0]:
                sub_w = max(s for s in range(1, min(4, nt) + 1) if nt % s == 0)
                sub_h = max(s for s in range(1, min(4 // sub_w, p) + 1) if p % s == 0)
                c = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=GRID, in0_block_w=w, out_subblock_h=sub_h,
                    out_subblock_w=sub_w, per_core_M=p, per_core_N=nt)
                try:
                    got_w = ttnn.to_torch(
                        ttnn.matmul(x, y, program_config=c, compute_kernel_config=CKC))
                except RuntimeError as e:
                    # The shipped CB model underestimates at some shapes; record, do not die.
                    oom.append(w)
                    continue
                if torch.equal(got_w, ref):
                    ex_ws.append(w)
            applies_main = blocks >= CORES
            rows.append(dict(dtype=dn, tokens=tok, op=lbl, in0=list(sa), in1=list(sb),
                             m_tiles=mt, k_tiles=kt, n_tiles=nt, blocks=blocks,
                             gate_applies_on_main=applies_main, rule_block_w=bw,
                             exact_widths=ex_ws, cb_overflow_widths=oom, main_bit_exact=exact))
            flag = ""
            if applies_main and not exact:
                flag = "  <-- MERGED MAIN NOT BIT-EXACT HERE"
            print(f"{dn} T={tok:4d} {lbl:6s} Mt={mt:2d} Kt={kt:2d} Nt={nt:2d} blocks={blocks:4d} "
                  f"applies_main={str(applies_main):5s} rule_bw={bw} exact_widths={ex_ws} "
                  f"main_exact={exact}{flag}", flush=True)
        for t in (q, kT, a, v):
            ttnn.deallocate(t)

bad = [r for r in rows if r["gate_applies_on_main"] and not r["main_bit_exact"]]
out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/widthscan_qb1c0.json"
json.dump({"grid": list(GRID), "cores": CORES, "rows": rows,
           "main_non_bit_exact": bad}, open(out, "w"), indent=2)
print(f"\n{len(bad)} of {sum(1 for r in rows if r['gate_applies_on_main'])} classes where main's "
      f"gate applies are NOT bit-exact:", flush=True)
for r in bad:
    print(f"  {r['dtype']} T={r['tokens']} {r['op']} Mt={r['m_tiles']} Kt={r['k_tiles']} "
          f"Nt={r['n_tiles']} rule={r['rule_block_w']} exact={r['exact_widths']}", flush=True)
print("wrote", out, flush=True)
