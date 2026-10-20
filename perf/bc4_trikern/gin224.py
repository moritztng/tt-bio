"""BindCraft 2's trimul input stage at a whole-tile grid (224) with AF2's biases: `trimul_tail.gin_ragged`
under GIN_WHOLE (projection + both gates + channel move in one program) against the route BC2 runs today
(biased projection into a [1, S, S, 512] DRAM tensor, then `reblock_permute_gated` for a and for b).

Every arm is graded against float64 (rel RMS of a and b), never against the other arm. Timing: device ms is
synced, best of reps x calls; host ms is the enqueue loop alone (a cached-descriptor incumbent can win end to
end on host cost).

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_trikern/gin224.py [--n 224 288 320] --out x.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TT_BIO_TRIMUL_GIN_WHOLE"] = "1"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[224, 288, 320])
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG, reblock_permute as R
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
ckc = compute_kernel_config()
DR = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
out = open(A.out, "a")


def emit(rec):
    rec["aiclk"] = open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().strip()
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def timed(fn):
    dev_best = host_best = 1e9
    for _ in range(A.reps):
        for o in fn():
            o.deallocate()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(A.calls)]
        t1 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t2 = time.perf_counter()
        for outs in keep:
            for o in outs:
                o.deallocate()
        host_best = min(host_best, (t1 - t0) / A.calls)
        dev_best = min(dev_best, (t2 - t0) / A.calls)
    return dev_best * 1e3, host_best * 1e3


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=DR)


def rr(d, r):
    return float(((d.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())


for n in A.n:
    C, K = A.c, A.k
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)          # LayerNorm-scale input
    wp = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)     # [p_a | p_b]
    wg = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)     # [g_a | g_b]
    bp = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    bg = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    x64 = x.double().reshape(n * n, K)
    ab = (x64 @ wp.double() + bp.double()) * torch.sigmoid(x64 @ wg.double() + bg.double())
    ab = ab.reshape(1, n, n, 2 * C).permute(0, 3, 1, 2)
    ref = [ab[:, :C], ab[:, C:]]
    dx = up(x)

    dwpT, dwgT = up(wp.t()), up(wg.t())
    dbias = TT.gin_ragged_bias(bp, bg, n, dev)
    ok = TT.gin_ragged_ok(dx, dwpT, n)
    rec = {"n": n, "route": "gin", "ok": ok, "grid": grid}
    if ok:
        def gin():
            return TT.gin_ragged(dx, dwpT, dwgT, dbias, MG.ckc_args(ckc), grid, n, memory_config=DR)
        a, b = gin()
        ta, tb = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
        a.deallocate(); b.deallocate()
        rec.update(finite=bool(torch.isfinite(ta).all() and torch.isfinite(tb).all()),
                   rel_a=rr(ta, ref[0]), rel_b=rr(tb, ref[1]))
        rec["dev_ms"], rec["host_ms"] = timed(gin)
    emit(rec)

    # today's BC2 route: biased projection [p_a | p_b | g_a | g_b] into DRAM, then two gated moves
    W = up(torch.cat([wp, wg], 1))
    Bv = up(torch.cat([bp, bg]).reshape(1, -1))

    def inc():
        y = ttnn.linear(dx, W, bias=Bv, compute_kernel_config=ckc, memory_config=DR)
        a = R.reblock_permute_gated(y, 0, 2 * C, C)
        b = R.reblock_permute_gated(y, C, 3 * C, C)
        y.deallocate()
        return [a, b]

    a, b = inc()
    ta, tb = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
    a.deallocate(); b.deallocate()
    rec = {"n": n, "route": "incumbent", "rel_a": rr(ta, ref[0]), "rel_b": rr(tb, ref[1])}
    rec["dev_ms"], rec["host_ms"] = timed(inc)
    emit(rec)

    def proj():
        return [ttnn.linear(dx, W, bias=Bv, compute_kernel_config=ckc, memory_config=DR)]
    rec = {"n": n, "route": "incumbent_proj_only"}
    rec["dev_ms"], rec["host_ms"] = timed(proj)
    emit(rec)
    for t in (dx, dwpT, dwgT, dbias, W, Bv):
        t.deallocate()
emit({"stats_gin": TT.GIN_RAGGED_STATS, "done": True})
