#!/usr/bin/env python3
"""The same question as `rate_ab.py`, one class over: is TriangleMultiplication's rate a stock-op
rate too?

After the TriangleAttention re-pricing (`TRIATT_RATE.md`) `TriangleMultiplication` is the only
class left whose floor is materially above the time the fold takes for it, 4.391 s against 3.544 s,
123.9 %. `QUIET_REFOLD.md` read that as the `sum(max(...))` construction rather than a rate error,
because the class's arithmetic sum (3.157 s) and traffic sum (2.120 s) each sit below measured. But
its in-fold capture holds four `ttnn.generic_op` calls beside one `ttnn.matmul` and one
`ttnn.linear`, and two of its three catalogue arms are stock ops -- the same shape of mistake.

Arms, interleaved per rep, minimum over reps, one session, one device:

  trimul      the shipped TriangleMultiplication unit at the fold's 512 aa shape with a pair mask.
              The fold spends 6.348 ms on this call (528 of them,
              `roof_quiet/out_quiet_trimmed/ROOF_BUDGET.md`).
  trimul2     the A/A floor
  cat_in      `trimul_in_flat`, the catalogue's winning in-projection arm, verbatim
  cat_einsum  `trimul_einsum`, the catalogue's triangle-product arm, verbatim
  cat_out     `pair_out128_flat`, the catalogue's out-projection arm, verbatim
  cube        the dense 4096^3, this session's own denominator

The sum of the three catalogue arms is what the floor charges the class per call. The unit is what
the shipped kernels plus everything that is not a matmul actually take. If the two differ the way
TriangleAttention's did, the class is mis-rated for the same reason.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio import tenstorrent as T                                           # noqa: E402

CZ, HIDDEN = 128, 128


def weights(cz=CZ, hidden=HIDDEN, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float32) * 0.05
    return {
        "norm_in.weight": torch.ones(cz), "norm_in.bias": torch.zeros(cz),
        "norm_out.weight": torch.ones(hidden), "norm_out.bias": torch.zeros(hidden),
        "g_in.weight": r(2 * hidden, cz), "p_in.weight": r(2 * hidden, cz),
        "g_out.weight": r(cz, cz), "p_out.weight": r(cz, hidden),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", default="perf/roof_triatt_rate/trimul_rate_ab_512_qb2c3.json")
    a = ap.parse_args()
    S, C, H = a.n, CZ, HIDDEN

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)

    meta = {"n": S, "c_z": C, "hidden": H, "arch": str(dev.arch()),
            "grid": list(T.COMPUTE_GRID_MAIN), "card": os.environ.get("TT_VISIBLE_DEVICES"),
            "host": os.uname().nodename, "loadavg": os.getloadavg(),
            "reps": a.reps, "warm": a.warm}
    print(json.dumps(meta), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mod = T.TriangleMultiplication(False, weights(), ckc)
    g = torch.Generator().manual_seed(1)
    z = dram(torch.randn(1, S, S, C, generator=g, dtype=torch.float32).to(torch.bfloat16))
    s_ = (torch.rand(1, S, generator=g) > 0.25).float()
    mask = dram((s_[:, :, None] * s_[:, None, :]).to(torch.bfloat16))
    zflat = dram(torch.randn(S * S, C).to(torch.bfloat16) * 0.1)
    w640 = dram(torch.randn(C, 5 * C).to(torch.bfloat16))
    w128 = dram(torch.randn(C, C).to(torch.bfloat16))
    ta = dram(torch.randn(1, C, S, S).to(torch.bfloat16))
    tb = dram(torch.randn(1, C, S, S).to(torch.bfloat16))
    cube_a = dram(torch.randn(4096, 4096).to(torch.bfloat16))
    cube_b = dram(torch.randn(4096, 4096).to(torch.bfloat16))

    def f_trimul():
        ttnn.deallocate(mod(z, mask))

    def f_cat_in():
        ttnn.deallocate(ttnn.linear(zflat, w640, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))

    def f_cat_einsum():
        ttnn.deallocate(ttnn.matmul(ta, tb, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))

    def f_cat_out():
        ttnn.deallocate(ttnn.linear(zflat, w128, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))

    def f_cube():
        ttnn.deallocate(ttnn.matmul(cube_a, cube_b, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))

    IN_F = 2 * S * S * C * 5 * C
    EIN_F = 2 * C * S * S * S
    OUT_F = 2 * S * S * C * C
    arms = {
        "trimul": (f_trimul, IN_F + EIN_F + OUT_F),
        "trimul2": (f_trimul, IN_F + EIN_F + OUT_F),
        "cat_in": (f_cat_in, IN_F),
        "cat_einsum": (f_cat_einsum, EIN_F),
        "cat_out": (f_cat_out, OUT_F),
        "cube": (f_cube, 2 * 4096 ** 3),
    }
    ts = {kk: [] for kk in arms}
    for i in range(a.warm + a.reps):
        for name, (fn, _f) in arms.items():
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            if i >= a.warm:
                ts[name].append(dt)
        print("rep %d  %s" % (i, "  ".join("%s %.3fms" % (nm, 1e3 * ts[nm][-1])
                                           for nm in arms if ts[nm])), flush=True)

    res = {"meta": meta, "arms": {}}
    for name, (_fn, f) in arms.items():
        v_ = ts[name]
        res["arms"][name] = {"min_ms": 1e3 * min(v_), "median_ms": 1e3 * st.median(v_),
                             "GFLOP": f / 1e9, "TFLOPs_at_min": f / min(v_) / 1e12,
                             "reps": len(v_)}
    res["meta"]["loadavg_after"] = os.getloadavg()
    Path(a.out).write_text(json.dumps(res, indent=1))

    A = res["arms"]
    cat = A["cat_in"]["min_ms"] + A["cat_einsum"]["min_ms"] + A["cat_out"]["min_ms"]
    print(json.dumps({k: round(v["min_ms"], 3) for k, v in A.items()}))
    print("A/A floor %.2f %%" % (100 * abs(A["trimul"]["min_ms"] - A["trimul2"]["min_ms"])
                                 / min(A["trimul"]["min_ms"], A["trimul2"]["min_ms"])))
    print("catalogue arm sum %.3f ms/call   shipped unit %.3f ms   the fold 6.348 ms"
          % (cat, A["trimul2"]["min_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
