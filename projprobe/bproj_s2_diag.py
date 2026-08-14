#!/usr/bin/env python3
"""Bisect the S2' fault, then verify a candidate fix, by substituting operands whose exact answer
is known and reading the error's STRUCTURE rather than guessing at it.

  T5  coef = [1,0,0], selt = [I,0,0], n contributions -> the exact answer is n * win, so the
      least-squares scale IS the landed contribution count and the fault reads straight off it.
  T1..T4  one substitution at a time: identity selection, identity coefficients, both, neither.

A least-squares scale of chunk-1 at every (ncontrib, chunk) means one matmul per acquire produces
nothing; a scale that depends on `chunk` and not on `ncontrib` means the cross-chunk carry is lost.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bproj_e2e import (A, ELEM, NCOPY, NROWS, WIN, build_s2a, coef_tiles, selt_matrices, shear)


def push(x, dev):
    return ttnn.from_torch(torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).reshape(1, 1, -1, 32),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


ACC = {}


def run(dev, sl, coef, selt, row_el, offs_el, rowidx1, nc, ch, kern):
    # The W store is bf16 unconditionally now: section 8.3's rule is that every CB the compute
    # kernel packs into carries ONE format and the accumulator lives in fp32 DST instead.
    w1 = ttnn.from_torch(torch.zeros(1, 1, 32 * 64, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    pd = build_s2a(sl, coef, selt, w1, 1, 1, 1, row_el, offs_el * ELEM, rowidx1, nc, ch,
                   compute=kern, mid=ACC["mid"], dstacc=ACC["dstacc"])
    ttnn.generic_op([sl, coef, selt, w1], pd)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(w1)[0, 0, :32, :].to(torch.float64).numpy()
    ttnn.deallocate(w1)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dstacc", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--mid", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--kernels", default="compute_bproj_ds.cpp,compute_bproj_ds_v1.cpp,compute_bproj_ds_v2.cpp")
    a = ap.parse_args()
    ACC["dstacc"] = a.dstacc != "off"
    ACC["mid"] = None if a.mid == "bf16" else ttnn.float32
    row_el, ncore = 512, 1
    dev = ttnn.open_device(device_id=0)
    try:
        rng = np.random.default_rng(97)
        sl_rows = ncore * 32 * NCOPY
        sl_np = rng.integers(-100, 100, size=(sl_rows, row_el)).astype(np.float32)
        sl_t = torch.from_numpy(sl_np).to(torch.bfloat16)
        sln = sl_t.to(torch.float64).numpy()
        sl = ttnn.from_torch(sl_t.reshape(1, 1, sl_rows, row_el), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED,
                                                             ttnn.BufferType.L1))
        k0, h, rho, offs_el = shear(0.77, 5.3)
        win = np.stack([sln[r * NCOPY + rho[r], offs_el[r]:offs_el[r] + WIN] for r in range(NROWS)])
        winb = torch.from_numpy(win).to(torch.bfloat16).to(torch.float64).numpy()
        rowidx1 = [(np.arange(NROWS) * NCOPY + rho).tolist()]

        cf_real = coef_tiles(h)
        cf_1 = np.zeros((3, 32, 32), dtype=np.float32); cf_1[0] = 1.0
        q_real = np.stack(selt_matrices(A))
        q_I = np.zeros((3, 32, 32), dtype=np.float32); q_I[0] = np.eye(32)
        bq = lambda x: torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).to(torch.float64).numpy()

        c1, qI = push(cf_1.reshape(96, 32), dev), push(q_I.reshape(96, 32), dev)
        cR, qR = push(cf_real.reshape(96, 32), dev), push(q_real.reshape(96, 32), dev)
        ref4 = sum((bq(cf_real)[d] * winb) @ bq(q_real)[d] for d in range(3))

        # Which of the two candidates is it: the FIRST matmul_tiles of an acquire producing
        # nothing, or the first CONTRIBUTION being staged wrong? Put the whole value on d=1 instead
        # of d=0. The kernel still issues nmid=3 matmuls per contribution, but now the first one of
        # the acquire (n=0, i=0) has a zero operand. If the fault is the first matmul, moving the
        # value off it costs nothing and every contribution lands. If the fault is upstream in the
        # staging, contribution 0 is still lost wherever its value sits.
        cf_d1 = np.zeros((3, 32, 32), dtype=np.float32); cf_d1[1] = 1.0
        q_d1 = np.zeros((3, 32, 32), dtype=np.float32); q_d1[1] = np.eye(32)
        cD, qD = push(cf_d1.reshape(96, 32), dev), push(q_d1.reshape(96, 32), dev)

        for kern in a.kernels.split(","):
            print(f"\n== {kern}  T6 value on d=1, first matmul of the acquire has a zero operand",
                  flush=True)
            for nc, ch in ((1, 1), (2, 2), (4, 4), (8, 8)):
                got = run(dev, sl, cD, qD, row_el, offs_el, rowidx1, nc, ch, kern)
                k = float((got * winb).sum() / (winb * winb).sum())
                print(f"      ncontrib {nc:3d}  chunk {ch:2d}: landed {k:8.3f}", flush=True)

        for kern in a.kernels.split(","):
            print(f"\n== {kern}", flush=True)
            print("  T5 coef=1 selt=I: landed contribution count (exact answer is ncontrib)", flush=True)
            for nc, ch in ((1, 1), (2, 2), (4, 4), (8, 8), (8, 4), (8, 2), (8, 1), (48, 8)):
                got = run(dev, sl, c1, qI, row_el, offs_el, rowidx1, nc, ch, kern)
                k = float((got * winb).sum() / (winb * winb).sum())
                resid = float(np.linalg.norm(got - k * winb) / max(np.linalg.norm(got), 1e-30))
                print(f"      ncontrib {nc:3d}  chunk {ch:2d}  ({nc//ch} chunks): landed {k:8.3f}"
                      f"   scale-removed residual {resid:.3e}", flush=True)
            for tag, cf, q, ref in (("T4 coef=c selt=Q", cR, qR, ref4),
                                    ("T1 coef=1 selt=I", c1, qI, winb)):
                for nc, ch in ((8, 8), (48, 8)):
                    got = run(dev, sl, cf, q, row_el, offs_el, rowidx1, nc, ch, kern)
                    r = nc * ref
                    rel = float(np.linalg.norm(got - r) / max(np.linalg.norm(r), 1e-30))
                    e = np.abs(got - r).ravel()
                    print(f"  {tag}  nc {nc:3d} chunk {ch}: rel L2 {rel:.4e}  max {e.max():.3e}"
                          f"  p50 {np.percentile(e,50):.3e}", flush=True)
    finally:
        ttnn.close_device(dev)


main()
