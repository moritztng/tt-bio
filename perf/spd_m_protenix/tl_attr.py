"""Stage attribution of triatt_ln (triangle attention's fused LayerNorm + q/k/v/gate/bias projection).

    TT_VISIBLE_DEVICES=<card> python perf/spd_m_protenix/tl_attr.py [--n 512] [--c 256] [--reps 10]

Protenix-v2's shape: x [N, N, c], 8 heads of 32, weight [c, 4c + 32]. For the starting (plain read) and the ending
(transposed read) variant and each fidelity, times the kernel whole and with triatt_ln.ABL stage skips: 1 the
layer_norm, 2 the projection, 4 the DRAM writes, 8 the DRAM reads of x, or several ORed. A skipped stage makes the
output wrong by design, so this prints time only: ms/call, median of --reps after 3 warm calls. One JSON line per arm.
--b8 0,1,2 repeats every arm with lever triatt_ln_b8 (1: bfp8 q, k, v) and + triatt_ln_b8g (2: the gate too); at ABL 0 each B8 arm also runs on a grid one row
shorter and prints whether its outputs are torch.equal to the full grid's (grid_invariant), and the bf16 arm's
deviation from a float64 reference beside the bfp8 arm's (rel_rms_f64).
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--abl", default="0,1,2,4,8,12,3,7,11,14,13", help="comma list of ABL masks")
ap.add_argument("--fids", default="HiFi3,HiFi2")
ap.add_argument("--b8", default="0", help="comma list: bf16 (0) / triatt_ln_b8 (1) / + triatt_ln_b8g (2)")
ap.add_argument("--tr", default="0,1", help="comma list: plain (0) / transposed (1) read; OpenDDE's c 384 takes 0 only")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_ln as TL  # noqa: E402

g = torch.Generator().manual_seed(0)
c, H = a.c, a.c // 32
dev = T.get_device()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
x = up(torch.randn(a.n, a.n, c, generator=g))
w = up(torch.randn(c, 4 * c + 32, generator=g) / c ** 0.5)
lw, lb = up(torch.ones(c)), up(torch.zeros(c))
grid = tuple(T.COMPUTE_GRID_MAIN)


B8 = 0


def run(fid, tr, grid=grid):
    with T.lever_scope(["triatt_ln"] + ["triatt_ln_b8", "triatt_ln_b8g"][:B8]):
        r = TL.ln_qkvgb_heads(x, lw, lb, w, (fid, False, True, False), H, 32, 8, grid, transposed=tr)
    assert r is not None, TL.REJECTS
    return r


def free(r):
    (q, k, v), gt, b = r
    for t in (q, k, v, gt, b):
        ttnn.deallocate(t)


def timed(fn):
    for _ in range(3):
        free(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        free(o)
    return round(1e3 * statistics.median(ts), 3)


def host(r):
    (q, k, v), gt, b = r
    return [ttnn.to_torch(t).float() for t in (q, k, v, gt, b)]


def reference(tr):
    """float64 q, k, v, gate (head-major) and bias of layer_norm(x) @ w, as the kernel lays them out."""
    xs = ttnn.to_torch(x).double()
    if tr:
        xs = xs.transpose(0, 1)
    xn = torch.nn.functional.layer_norm(xs, (c,), eps=1e-5)
    y = xn @ ttnn.to_torch(w).double()
    heads = [y[..., i * c:(i + 1) * c].reshape(a.n, a.n, H, 32).permute(0, 2, 1, 3) for i in range(4)]
    return heads + [y[..., 4 * c:4 * c + 8]]


def rel_rms(out, ref):
    o = [t[..., :r.shape[-1]].double().reshape(r.shape) for t, r in zip(out, ref)]
    return round(max(float((p - q).pow(2).mean().sqrt() / q.pow(2).mean().sqrt()) for p, q in zip(o, ref)), 5)


small = (grid[0], grid[1] - 1)
for fid in [getattr(ttnn.MathFidelity, f) for f in a.fids.split(",")]:
    for tr in [bool(int(d)) for d in a.tr.split(",")]:
        ref = None
        for b8 in [int(d) for d in a.b8.split(",")]:
            B8 = b8
            for abl in [int(m) for m in a.abl.split(",")]:
                TL.ABL = abl
                ms = timed(lambda: run(fid, tr))
                row = dict(n=a.n, c=c, card=os.environ.get("TT_VISIBLE_DEVICES"), fid=str(fid), tr=tr, b8=b8,
                           abl=abl, ms=ms)
                if abl == 0 and (a.b8 != "0"):
                    if ref is None:
                        ref = reference(tr)
                    r = run(fid, tr)
                    row["q_dtype"] = str(r[0][0].dtype)
                    out = host(r)
                    free(r)
                    row["rel_rms_f64"] = rel_rms(out, ref)
                    if b8:
                        r = run(fid, tr, small)
                        row["grid_invariant"] = all(torch.equal(p, q) for p, q in zip(out, host(r)))
                        free(r)
                print(json.dumps(row), flush=True)
TL.ABL = 0
