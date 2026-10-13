#!/usr/bin/env python3
"""ESMFold2's MSA outer-product mean at 512 aa: the shipped tile path against two reshape-free ones.

Census c3 (p150a, 1350 MHz, ESMFold2 --fast): msa_enc/opm is 0.457 s of device a fold, 0.380 s of it two
ttnn.reshape calls per block that turn each 32x32 product tile into a row of 1024. Arms (OuterProductMean in
tt_bio/esmfold2.py): tile = `_rows`, rm = `_rows_rm`, kq = `_rows_kq`. Each is timed as the back-to-back slope of
--calls calls between two synchronizes, arms round-robin per rep; error vs a float64 reference of the same inputs.

    python perf/spd_gap_esmf/opm_bench.py OUT [--n 512] [--m 35]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--m", type=int, nargs="+", default=[35, 64])
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--calls", type=int, default=4)
ap.add_argument("--arms", nargs="+", default=["tile", "rm", "kq"])
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.esmfold2 import OuterProductMean

dev = T.get_device()
ck = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                             packer_l1_acc=True) if "BLACKHOLE" in str(dev.arch()).upper() else \
    ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           packer_l1_acc=True)
g = torch.Generator().manual_seed(0)
L = A.n
sd = {"norm.weight": torch.ones(128), "norm.bias": torch.zeros(128),
      "W.weight": torch.randn(64, 128, generator=g) / 128 ** 0.5,
      "Wout.weight": torch.randn(256, 1024, generator=g) / 1024 ** 0.5,
      "Wout.bias": torch.randn(256, generator=g) * 0.1}
opm = OuterProductMean(sd, ck)
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
log(ev="setup", arch=str(dev.arch()), n=L, ms=A.m, arms=A.arms)

for M in A.m:
    ah = torch.randn(1, L, M, 32, generator=g).to(torch.bfloat16)
    bh = torch.randn(1, L, M, 32, generator=g).to(torch.bfloat16)
    rh = (1.0 / torch.full((1, L, L, 1), float(M))).to(torch.bfloat16)
    W = sd["Wout.weight"].to(torch.bfloat16).double()
    outer = torch.einsum("bimp,bjmq->bijpq", ah.double(), bh.double()).reshape(1, L, L, 1024)
    ref = (outer @ W.t() + sd["Wout.bias"].to(torch.bfloat16).double()) * rh.double()
    del outer
    a, b, r = up(ah), up(bh), up(rh)
    b2 = ttnn.reshape(ttnn.permute(b, (0, 2, 1, 3)), (1, M, L * 32))
    fns = {"tile": lambda: opm._rows(a, b2, r, L, M), "rm": lambda: opm._rows_rm(a, b2, r, L, M),
           "kq": lambda: opm._rows_kq(a, b, r, L, M)}
    for arm in A.arms:  # warm (program cache) + error
        try:
            o = fns[arm]()
            h = ttnn.to_torch(o).double()[..., :256]
            ttnn.deallocate(o)
            err = (h - ref)
            log(ev="err", M=M, arm=arm, finite=bool(torch.isfinite(h).all()),
                rel_rms=float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(err.abs().max()))
        except Exception as exc:
            log(ev="arm_failed", M=M, arm=arm, err=f"{type(exc).__name__}: {str(exc)[:400]}")
            fns.pop(arm)
    ts = {k: [] for k in fns}
    for rep in range(A.reps):
        for arm, fn in fns.items():
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(A.calls):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts[arm].append((time.perf_counter() - t0) / A.calls * 1e3)
    for arm, v in ts.items():
        log(ev="time", M=M, arm=arm, ms_median=round(statistics.median(v), 3), ms=[round(x, 3) for x in v])
log(ev="end")
