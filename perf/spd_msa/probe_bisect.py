"""spd-msa: pin a HiFi4 matmul outlier to the K element that causes it.

probe_outliers.py found ~27 of 138M outputs of [541696,1024]x[1024,256] (randn, weights / 32, seed 0) off by
1-4 under HiFi4 and none under HiFi2. For each of the first outliers this rebuilds one 32x32 output tile from
the exact operand rows/columns (row i of x in tile row 0, column j of w in tile column 0, the rest zero), checks
it still misfires, then bisects K by zeroing halves until one (x_k, w_k) pair (or a set) carries the error.

usage: TT_VISIBLE_DEVICES=<chip> python probe_bisect.py OUT [ROWS=541696]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 541696
LOG = open(OUT / "bisect.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
K, N = 1024, 256
torch.manual_seed(0)
x_h = torch.randn(ROWS, K).bfloat16(); w_h = (torch.randn(K, N) / 32).bfloat16()
F = ttnn.MathFidelity


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)


def dev_dot(a, b, fid=F.HiFi4):
    """a [K], b [K] -> device result of the 32x32-tile matmul's [0,0] (fp32 output)."""
    A = torch.zeros(32, a.numel(), dtype=torch.bfloat16); A[0] = a
    B = torch.zeros(b.numel(), 32, dtype=torch.bfloat16); B[:, 0] = b
    At = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    Bt = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = ttnn.matmul(At, Bt, compute_kernel_config=ckc(fid), dtype=ttnn.float32)
    v = float(ttnn.to_torch(o)[0, 0])
    for t in (At, Bt, o):
        ttnn.deallocate(t)
    return v


def exact(a, b):
    return float((a.double() * b.double()).sum())


def bits(v):
    return format(torch.tensor([v], dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF, "016b")


def one(x, w):
    """x and w alone, at K index 0 of a 32-long contraction."""
    a = torch.zeros(32, dtype=torch.bfloat16); b = torch.zeros(32, dtype=torch.bfloat16)
    a[0], b[0] = x, w
    return a, b


# the outliers probe_outliers.py logged for this seed (row, col)
cases = [(45719, 6), (68124, 71), (82499, 131), (88192, 98), (15121, 229), (57582, 94)]
ref_full = None
for i, j in cases:
    a, b = x_h[i].clone(), w_h[:, j].clone()
    r4, r2, ex = dev_dot(a, b), dev_dot(a, b, F.HiFi2), exact(a, b)
    log(ev="case", i=i, j=j, exact=ex, hifi4=r4, hifi2=r2, err4=r4 - ex, err2=r2 - ex)
    if abs(r4 - ex) < 0.25:
        continue
    # bisect K: keep the half whose own HiFi4 result still misses its own exact sum
    lo, hi = 0, K
    keep = torch.ones(K, dtype=torch.bool)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        found = None
        for s0, s1 in ((lo, mid), (mid, hi)):
            m = torch.zeros(K, dtype=torch.bool); m[s0:s1] = True
            am, bm = a * m, b * m
            e = dev_dot(am, bm) - exact(am, bm)
            if abs(e) > 0.25:
                found = (s0, s1, e); break
        if found is None:
            log(ev="split_loses_it", i=i, j=j, lo=lo, hi=hi)   # needs both halves: an accumulation effect
            break
        lo, hi = found[0], found[1]
        log(ev="narrow", i=i, j=j, lo=lo, hi=hi, err=found[2])
    if hi - lo == 1:
        k = lo
        log(ev="pair", i=i, j=j, k=k, x=float(a[k]), w=float(b[k]), x_bits=bits(float(a[k])),
            w_bits=bits(float(b[k])), product=float(a[k]) * float(b[k]),
            dev_single=dev_dot(*one(a[k], b[k])), dev_single_hifi2=dev_dot(*one(a[k], b[k]), F.HiFi2))
log(ev="end")
