"""The gated in-projection with its channel moves fused in (`trimul_tail.gin_moved`) against today's
route (resident split call `fused_tail(split=2, mask)` then two plain `_channel_move`s).

Both give a, b as [1, C, S, S]; a carries the pair mask. Reports the float64 relative RMS of each,
the max difference between the two routes, and ms per call (synced, best of reps).

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/gin_move.py [--n 736] [--c 128] [--calls 10]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--c", type=int, default=128, help="channels per output chunk (a and b each)")
ap.add_argument("--calls", type=int, default=10)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--nomask", action="store_true")
ap.add_argument("--abl", default="", help="time only, moved route: stage ablation bits, e.g. 1,2,4,8,16")
ap.add_argument("--nb", default="", help="time only, moved route: writer tiles per read barrier, e.g. 1,4,8")
ap.add_argument("--sigpoly", action="store_true", help="add a 'moved_poly' route: the polynomial sigmoid")
ap.add_argument("--only-timing", action="store_true", help="skip the two-route check, run only --abl/--nb")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
n, C, K = A.n, A.c, 256
g = torch.Generator().manual_seed(0)
x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
wp = (torch.randn(K, 2 * C, generator=g) / 16).to(torch.bfloat16)
wg = (torch.randn(K, 2 * C, generator=g) / 16).to(torch.bfloat16)
m = None if A.nomask else (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16)
up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
dx, dwp, dwg, dwpT, dwgT = up(x), up(wp), up(wg), up(wp.t()), up(wg.t())
dm = None if m is None else up(m)
TT.GIN_MOVE = True
TT.set_epi(1)   # production's trimul_tail lever: the resident split route
print(json.dumps({"n": n, "c": C, "grid": grid, "ckc": [str(c) for c in ckc],
                  "ok": TT.gin_moved_ok(dx, dwpT, dm), "mask": dm is not None}), flush=True)

x64 = x.double().reshape(n * n, K)
ab = (x64 @ wp.double()) * torch.sigmoid(x64 @ wg.double())
ab = ab.reshape(1, n, n, 2 * C)
ref = [ab[..., :C], ab[..., C:]]
if m is not None:
    ref[0] = ref[0] * m.double()[..., None]
ref = [r.permute(0, 3, 1, 2) for r in ref]


def today():
    outs = TT.fused_tail(dx, dx, dwp, dwg, ckc, grid, split=2, mask=dm)
    res = [T._channel_move(o, ttnn.DRAM_MEMORY_CONFIG) for o in outs]
    for o in outs:
        ttnn.deallocate(o)
    return res


def moved():
    return TT.gin_moved(dx, dwpT, dwgT, ckc, grid, mask=dm)


def rel(a, r):
    return float(((a.double() - r) ** 2).mean().sqrt() / (r ** 2).mean().sqrt())


def timed(f):
    for t in f():
        ttnn.deallocate(t)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(A.calls):
            for t in f():
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
    return round(min(ms), 4)


# NB first (an output-preserving change: checked against float64 too), then the ablations at the default NB.
for nb in [int(v) for v in A.nb.split(",") if v]:
    TT.GIN_MOVE_NB = nb
    res = moved()
    host = [ttnn.to_torch(t) for t in res]
    for t in res:
        ttnn.deallocate(t)
    print(json.dumps({"nb": nb, "ms": timed(moved), "rel_rms_f64": [round(rel(h, r), 6) for h, r in zip(host, ref)]}),
          flush=True)
NB0 = TT.GIN_MOVE_NB = int(__import__("os").environ.get("TT_BIO_TRIMUL_GIN_MOVE_NB", "2"))
for abl in [int(v) for v in A.abl.split(",") if v]:
    TT.GIN_MOVE_ABL = abl
    print(json.dumps({"abl": abl, "nb": NB0, "ms": timed(moved)}), flush=True)
TT.GIN_MOVE_ABL = 0
if A.only_timing:
    sys.exit(0)

out = {}
def moved_poly():
    TT.GIN_SIGPOLY = True
    try:
        return moved()
    finally:
        TT.GIN_SIGPOLY = False


routes = (("today", today), ("moved", moved)) + ((("moved_poly", moved_poly),) if A.sigpoly else ())
for name, f in routes:
    rec = {"route": name}
    try:
        res = f()
        ttnn.synchronize_device(dev)
        host = [ttnn.to_torch(t) for t in res]
        out[name] = host
        rec["shape"] = list(host[0].shape)
        rec["rel_rms_f64"] = [round(rel(h, r), 6) for h, r in zip(host, ref)]
        rec["finite"] = all(bool(torch.isfinite(h).all()) for h in host)
        for t in res:
            ttnn.deallocate(t)
        ms = []
        for _ in range(A.reps):
            t0 = time.perf_counter()
            for _ in range(A.calls):
                for t in f():
                    ttnn.deallocate(t)
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
        rec["ms"] = round(min(ms), 4)
        rec["spread_ms"] = round(max(ms) - min(ms), 4)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    print(json.dumps(rec), flush=True)
if "today" in out and "moved" in out:
    d = [float((a.float() - b.float()).abs().max()) for a, b in zip(out["today"], out["moved"])]
    ne = [int((a != b).sum()) for a, b in zip(out["today"], out["moved"])]
    print(json.dumps({"max_abs_today_vs_moved": d, "differing": ne, "of": out["today"][0].numel()}))
# Worst elements of each route against float64: an index bug shows up as a large error at few places.
for name, host in out.items():
    for which, (hh, r) in enumerate(zip(host, ref)):
        e = (hh.double() - r).abs()
        top = torch.topk(e.flatten(), 5)
        idx = [list(map(int, torch.unravel_index(i, e.shape))) for i in top.indices]
        print(json.dumps({"route": name, "out": "ab"[which], "max_err": round(float(top.values[0]), 4),
                          "n_err_gt_0.1": int((e > 0.1).sum()),
                          "worst": [[i, round(float(hh.flatten()[j]), 4), round(float(r.flatten()[j]), 4)]
                                    for i, j in zip(idx, top.indices)]}))
print(json.dumps({"stats": {"move": TT.GIN_MOVE_STATS, "res": TT.RES_STATS, "rejects": {f"{k[0]}:{k[1]}": v for k, v in TT.REJECTS.items()}}}))
