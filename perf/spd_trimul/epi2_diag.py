"""Where tests/test_trimul_tail_epi.py's EPI 2 element bound fails: the resident tail's z + p * sigmoid(g)
against float64, next to the test's own reference (z + EPI 1's bf16 product).

Per n: elements over the test's tolerance, the max error of each against float64, and where the worst
elements sit (row tile, column tile, row in tile), so a kernel indexing fault and a rounding-bound
question look different.

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/epi2_diag.py [--n 64,160]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", default="64,160")
A = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TTL, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

CZ = 256
dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
print(json.dumps({"ckc": [str(c) for c in ckc], "grid": grid}), flush=True)
for n in [int(v) for v in A.n.split(",")]:
    g = torch.Generator().manual_seed(n)
    bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).float()
    xa, xb, z = bf(1, n, n, CZ), bf(1, n, n, CZ), bf(1, n, n, CZ)
    wa, wb = bf(CZ, CZ, sc=CZ ** -0.5), bf(CZ, CZ, sc=CZ ** -0.5)
    d = torch.float64
    ref = (xa.to(d) @ wa.to(d)) * torch.sigmoid(xb.to(d) @ wb.to(d))
    xa_d, xb_d, wa_d, wb_d = up(xa), up(xb), up(wa), up(wb)
    rec = {"n": n}
    for epi in (1, 2):
        TTL.set_epi(epi)
        for res in ((False, True) if epi == 2 else (False,)):
            TTL.set_res(res)
            z_d = up(z)
            st0 = (list(TTL.RES_STATS), list(TTL.RESID_STATS), dict(TTL.REJECTS))
            y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid, resid=z_d if epi == 2 else None)
            if y is None:
                rec[f"epi{epi}_res{int(res)}"] = {"declined": {f"{k[0]}": v for k, v in TTL.REJECTS.items()}}
                continue
            route = {"res": [a - b for a, b in zip(TTL.RES_STATS, st0[0])],
                     "resid": [a - b for a, b in zip(TTL.RESID_STATS, st0[1])],
                     "same_buffer_as_z": y is z_d, "rne": TTL.RNE, "RES": TTL.RES}
            out = ttnn.to_torch(y).to(d)
            ttnn.deallocate(z_d)
            if epi == 1:
                p1 = out
                rec["epi1_err_max"] = round(float((out - ref).abs().max()), 5)
                continue
            exact = z.to(d) + ref
            want = z.to(d) + p1
            tol = 2.0 ** -7 * (z.abs().to(d) + p1.abs()) + 1e-6
            over = (out - want).abs() > tol
            e = (out - exact).abs()
            idx = torch.nonzero(over)[:8].tolist()
            rec[f"epi2_res{int(res)}"] = {"route": route,
                "n_over_tol": int(over.sum()), "of": out.numel(),
                "max_err_vs_f64": round(float(e.max()), 5),
                "test_ref_max_err_vs_f64": round(float((want - exact).abs().max()), 5),
                "rel_rms_vs_f64": round(float(e.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()), 6),
                "test_ref_rel_rms_vs_f64": round(float((want - exact).pow(2).mean().sqrt()
                                                       / exact.pow(2).mean().sqrt()), 6),
                "over_at": [[i[1], i[2], i[3], round(float(out[tuple(i)]), 4), round(float(want[tuple(i)]), 4),
                             round(float(exact[tuple(i)]), 4)] for i in idx],
            }
    TTL.set_res(False)
    print(json.dumps(rec), flush=True)
