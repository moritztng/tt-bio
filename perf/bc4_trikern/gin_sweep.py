"""Where gin_ragged's time goes at whole-tile grids: channel splits per cube (GIN_RAGGED_QS, fills the grid at
the cost of re-reading x), math fidelity, and the compute stage ablation (bit 16 no matmul, 1 no sigmoid, 2 no
gate multiply; the ablated rows are wrong on purpose and timing only). Graded rows report rel RMS vs float64.

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_trikern/gin_sweep.py --out x.jsonl [--n 224 288 320]
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TT_BIO_TRIMUL_GIN_WHOLE"] = "1"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[224, 288, 320])
ap.add_argument("--qs", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--abl", type=int, nargs="+", default=[16, 1, 2, 19])
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=5)
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
base = compute_kernel_config()
CKC = {"trunk": MG.ckc_args(base)}
CKC["hifi2"] = (ttnn.MathFidelity.HiFi2,) + CKC["trunk"][1:]


def emit(rec):
    rec["aiclk"] = open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().strip()
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def timed(fn):
    best = 1e9
    for _ in range(A.reps):
        for o in fn():
            o.deallocate()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(A.calls)]
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) / A.calls)
        for outs in keep:
            for o in outs:
                o.deallocate()
    return best * 1e3


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=DR)


def rr(d, r):
    return float(((d.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())


emit({"grid": grid, "trunk_fidelity": str(CKC["trunk"][0])})
for n in A.n:
    C = K = 128
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
    wp = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
    wg = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
    bp = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    bg = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
    x64 = x.double().reshape(n * n, K)
    ab = (x64 @ wp.double() + bp.double()) * torch.sigmoid(x64 @ wg.double() + bg.double())
    ab = ab.reshape(1, n, n, 2 * C).permute(0, 3, 1, 2)
    ref = [ab[:, :C], ab[:, C:]]
    dx, dwpT, dwgT = up(x), up(wp.t()), up(wg.t())
    dbias = TT.gin_ragged_bias(bp, bg, n, dev)

    def run(ck):
        return TT.gin_ragged(dx, dwpT, dwgT, dbias, ck, grid, n, memory_config=DR)

    for fid, ck in CKC.items():
        for qs in A.qs:
            TT.GIN_RAGGED_QS = qs
            a, b = run(ck)
            ta, tb = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
            a.deallocate(); b.deallocate()
            emit({"n": n, "fid": fid, "qs": qs, "units": (-(-n // 32)) ** 2 * qs,
                  "rel_a": rr(ta, ref[0]), "rel_b": rr(tb, ref[1]), "ms": timed(lambda: run(ck))})
    TT.GIN_RAGGED_QS = 1
    for bits in A.abl:
        TT.GIN_MOVE_ABL = bits
        emit({"n": n, "fid": "trunk", "qs": 1, "abl": bits, "ms": timed(lambda: run(CKC["trunk"]))})
    TT.GIN_MOVE_ABL = 0
    for t in (dx, dwpT, dwgT, dbias):
        t.deallocate()
emit({"stats_gin": TT.GIN_RAGGED_STATS, "done": True})
