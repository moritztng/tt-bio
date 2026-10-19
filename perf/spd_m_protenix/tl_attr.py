"""Stage attribution of triatt_ln (triangle attention's fused LayerNorm + q/k/v/gate/bias projection).

    TT_VISIBLE_DEVICES=<card> python perf/spd_m_protenix/tl_attr.py [--n 512] [--c 256] [--reps 10]

Protenix-v2's shape: x [N, N, c], 8 heads of 32, weight [c, 4c + 32]. For the starting (plain read) and the ending
(transposed read) variant and each fidelity, times the kernel whole and with triatt_ln.ABL stage skips: 1 the
layer_norm, 2 the projection, 4 the DRAM writes, 8 the DRAM reads of x, or several ORed; 16 writes every tile to
q's pages in row-major order instead of head-major (same bytes, consecutive pages: is the layout the cost?). A skipped stage makes the
output wrong by design, so this prints time only: ms/call, median of --reps after 3 warm calls. One JSON line per arm.
--dnoc 0,1 repeats every arm with triatt_ln.DNOC off and on (the writer drains half its tiles on the other NOC) and, at
ABL 0, prints whether the five outputs are torch.equal between the two.
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--abl", default="0,1,2,4,8,12,3,7,11,14,13", help="comma list of ABL masks")
ap.add_argument("--fids", default="HiFi3,HiFi2")
ap.add_argument("--dnoc", default="0", help="comma list of triatt_ln.DNOC values")
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


def run(fid, tr):
    with T.lever_scope(["triatt_ln"]):
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
    return [ttnn.to_torch(t) for t in (q, k, v, gt, b)]


for fid in [getattr(ttnn.MathFidelity, f) for f in a.fids.split(",")]:
    for tr in (False, True):
        ref = None
        for dn in [bool(int(d)) for d in a.dnoc.split(",")]:
            TL.DNOC = dn
            for abl in [int(m) for m in a.abl.split(",")]:
                TL.ABL = abl
                ms = timed(lambda: run(fid, tr))
                row = dict(n=a.n, c=c, card=os.environ.get("TT_VISIBLE_DEVICES"), fid=str(fid), tr=tr, dnoc=dn,
                           abl=abl, ms=ms)
                if abl == 0:
                    r = run(fid, tr)
                    out = host(r)
                    free(r)
                    if ref is None:
                        ref = out
                    else:
                        row["equal_to_first"] = all(torch.equal(p, q) for p, q in zip(ref, out))
                print(json.dumps(row), flush=True)
TL.ABL = 0
TL.DNOC = False
