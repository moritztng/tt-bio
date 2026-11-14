"""gin_ragged per call at BC2's two production shapes (224 tokens, K = 128, masked): the whole call (C = 128, the
traced forward's 104/it) and the 64-channel chunk (C = 64, the eager forwards' 208/it), at qs 1, 2 and 4, against
float64. Then the compute stage ablations at each shape's best qs (wrong on purpose, timing only: bit 16 no
matmul, 1 no sigmoid, 2 no gate multiply; 19 no compute at all, the reads + move + write floor).

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_genroof/gin_probe.py --out x.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--c", type=int, nargs="+", default=[128, 64])
ap.add_argument("--qs", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--abl", type=int, nargs="+", default=[16, 1, 3, 19])
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
DR = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
out = open(A.out, "a")
CK = MG.ckc_args(compute_kernel_config())


def emit(rec):
    rec["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def timed(fn):
    runs = []
    for _ in range(A.reps):
        for o in fn():
            o.deallocate()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(A.calls)]
        ttnn.synchronize_device(dev)
        runs.append((time.perf_counter() - t0) / A.calls * 1e3)
        for outs in keep:
            for o in outs:
                o.deallocate()
    return min(runs), sorted(runs)[len(runs) // 2], max(runs)


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)


def rr(d, r):
    return float(((d.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())


emit({"grid": grid, "fidelity": str(CK[0]), "sha": os.popen("git rev-parse HEAD").read().strip()})
n, K = A.n, 128
for C in A.c:
    g = torch.Generator().manual_seed(C)
    x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
    wp = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
    wg = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
    bp = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    bg = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16)
    x64 = x.double().reshape(n * n, K)
    ab = (x64 @ wp.double() + bp.double()) * torch.sigmoid(x64 @ wg.double() + bg.double())
    ab = ab.reshape(1, n, n, 2 * C).permute(0, 3, 1, 2)
    ref = [ab[:, :C] * m.double()[:, None], ab[:, C:]]
    dx, dwpT, dwgT = up(x), up(wp.t()), up(wg.t())
    dm = up(m)
    dbias = TT.gin_ragged_bias(bp, bg, n, dev)
    run = lambda: TT.gin_ragged(dx, dwpT, dwgT, dbias, CK, grid, n, mask=dm, memory_config=DR)  # noqa: E731
    best = None
    for qs in A.qs:
        TT.GIN_RAGGED_QS = qs
        r = run()
        if r is None:
            emit({"c": C, "qs": qs, "declined": True})
            continue
        ta, tb = ttnn.to_torch(r[0]).double(), ttnn.to_torch(r[1]).double()
        for t in r:
            t.deallocate()
        lo, med, hi = timed(run)
        emit({"c": C, "qs": qs, "units": (-(-n // 32)) ** 2 * TT._gin_ragged_qs(2 * C // 32),
              "rel_a": rr(ta, ref[0]), "rel_b": rr(tb, ref[1]), "ms": lo, "ms_med": med, "ms_max": hi})
        if best is None or lo < best[1]:
            best = (qs, lo)
    TT.GIN_RAGGED_QS = best[0]
    for bits in A.abl:
        TT.GIN_MOVE_ABL = bits
        lo, med, hi = timed(run)
        emit({"c": C, "qs": best[0], "abl": bits, "ms": lo, "ms_med": med})
    TT.GIN_MOVE_ABL = 0
    TT.GIN_RAGGED_QS = 0
    for t in (dx, dwpT, dwgT, dbias, dm):
        t.deallocate()
emit({"stats_gin": TT.GIN_RAGGED_STATS, "done": True})
