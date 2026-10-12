"""AF2's trimul in-projection at a ragged pair grid (195 tokens) with biases: `trimul_tail.gin_ragged`
(projection + gate + channel move in one program, padding zeroed) against float64, and its time
against a stock route (biased ttnn.linear into L1, sigmoid-multiply, permute).

Checks, in order: every value finite; float64 rel RMS of a and b on the logical region; both
contractions over the padded axis (a @ b^T and a^T @ b per channel, the two trimul endings) against
float64, which fail if the padding is not zero. Timing is synced, best of reps x calls.

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_bind/gin195.py [--n 195] [--c 64] [--k 128]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=195)
ap.add_argument("--c", type=int, default=64, help="channels per output chunk (a and b each)")
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--mem", default="l1", choices=["l1", "dram"], help="x and the outputs")
ap.add_argument("--qs", default="0", help="channel-range splits per cube, comma list (0: by fill)")
ap.add_argument("--abl", default="", help="time only, af2 ckc: compute ablation bits, e.g. 1,2,16 (16: no matmul)")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
n, C, K = A.n, A.c, A.k
mc = ttnn.L1_MEMORY_CONFIG if A.mem == "l1" else ttnn.DRAM_MEMORY_CONFIG
g = torch.Generator().manual_seed(0)
x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
wp = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
wg = (torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16)
bp = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
bg = (torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16)
up = lambda t, m=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(
    t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=m)
dx, dwpT, dwgT = up(x, mc), up(wp.t()), up(wg.t())
dbias = TT.gin_ragged_bias(bp, bg, n, dev)

x64 = x.double().reshape(n * n, K)
ab = (x64 @ wp.double() + bp.double()) * torch.sigmoid(x64 @ wg.double() + bg.double())
ab = ab.reshape(1, n, n, 2 * C).permute(0, 3, 1, 2)
ref = [ab[:, :C], ab[:, C:]]
rr = lambda d, r: float(((d.double() - r).pow(2).mean() / r.pow(2).mean()).sqrt())

ckcs = {"af2": compute_kernel_config(),
        "hifi3": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3,
                                                  math_approx_mode=False, fp32_dest_acc_en=True,
                                                  packer_l1_acc=False)}
print(json.dumps({"n": n, "c": C, "k": K, "grid": grid, "mem": A.mem,
                  "ok": TT.gin_ragged_ok(dx, dwpT, n)}), flush=True)


def timed(fn):
    best = 1e9
    for _ in range(A.reps):
        fn(); ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(A.calls):
            outs = fn()
            for o in outs:
                o.deallocate()
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) / A.calls)
    return best * 1e3


for name, ckc in ckcs.items():
    for qs in [int(q) for q in A.qs.split(",")]:
        TT.GIN_RAGGED_QS = qs
        a, b = TT.gin_ragged(dx, dwpT, dwgT, dbias, MG.ckc_args(ckc), grid, n, memory_config=mc)
        ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
        fin = bool(torch.isfinite(ta).all() and torch.isfinite(tb).all())
        rec = {"route": "ragged", "ckc": name, "qs": TT._gin_ragged_qs(2 * C // 32)
               if qs == 0 else qs, "finite": fin}
        if fin:
            rec.update(rel_a=rr(ta, ref[0]), rel_b=rr(tb, ref[1]),
                       maxabs=float(max((ta.double() - ref[0]).abs().max(), (tb.double() - ref[1]).abs().max())))
            # Both endings contract over a padded axis: wrong if the padding is not zero.
            out = ttnn.to_torch(ttnn.matmul(a, ttnn.transpose(b, -2, -1), compute_kernel_config=ckc))
            inc = ttnn.to_torch(ttnn.matmul(ttnn.transpose(a, -2, -1), b, compute_kernel_config=ckc))
            rec.update(rel_outgoing=rr(out, ref[0] @ ref[1].transpose(-2, -1)),
                       rel_incoming=rr(inc, ref[0].transpose(-2, -1) @ ref[1]),
                       rel_outgoing_ref_ab=rr(out, ta.double() @ tb.double().transpose(-2, -1)))
        rec["ms"] = timed(lambda: TT.gin_ragged(dx, dwpT, dwgT, dbias, MG.ckc_args(ckc), grid, n,
                                                memory_config=mc))
        print(json.dumps(rec), flush=True)
        a.deallocate(); b.deallocate()
    TT.GIN_RAGGED_QS = 0

for bits in [int(b) for b in A.abl.split(",") if b]:
    TT.GIN_MOVE_ABL = bits
    ck = MG.ckc_args(ckcs["af2"])
    print(json.dumps({"route": "ragged", "abl": bits, "ms": timed(
        lambda: TT.gin_ragged(dx, dwpT, dwgT, dbias, ck, grid, n, memory_config=mc))}), flush=True)
TT.GIN_MOVE_ABL = 0

# Stock route: biased projection of [value | gate] into L1, gate, split, channel move.
W = up(torch.cat([wp, wg], 1))
Bv = up(torch.cat([bp, bg]).reshape(1, -1))
for name, ckc in ckcs.items():
    def stock():
        y = ttnn.linear(dx, W, bias=Bv, compute_kernel_config=ckc, memory_config=mc)
        p, gt = ttnn.slice(y, [0, 0, 0, 0], [1, n, n, 2 * C]), ttnn.slice(y, [0, 0, 0, 2 * C], [1, n, n, 4 * C])
        y.deallocate()
        o = ttnn.multiply(p, ttnn.sigmoid(gt, memory_config=mc), memory_config=mc)
        p.deallocate(); gt.deallocate()
        a = ttnn.permute(ttnn.slice(o, [0, 0, 0, 0], [1, n, n, C]), (0, 3, 1, 2), memory_config=mc)
        b = ttnn.permute(ttnn.slice(o, [0, 0, 0, C], [1, n, n, 2 * C]), (0, 3, 1, 2), memory_config=mc)
        o.deallocate()
        return [a, b]
    a, b = stock()
    rec = {"route": "stock", "ckc": name, "rel_a": rr(ttnn.to_torch(a), ref[0]),
           "rel_b": rr(ttnn.to_torch(b), ref[1]), "ms": timed(stock)}
    print(json.dumps(rec), flush=True)
print(json.dumps({"stats": TT.GIN_RAGGED_STATS}))
