#!/usr/bin/env python3
"""Price L3's fused landing with a kernel that already exists, before writing one.

E6 measured that `trimul_tail.eligible()` accepts Transition's fc1/fc2 chunk shape as it stands.
F1 computes `p * sigmoid(g)`; swiglu wants `silu(p) * g`. The ARITHMETIC differs, the WORK does not:
two GEMMs on the same operands, one eltwise epilogue, one write. So dropping F1 in unmodified gives
the fused arm's real wall at this shape and turns L3's 0.846 s/fold ceiling into a measured landing.

This is a PERF PROTOTYPE and its output is numerically wrong on purpose -- sigmoid where production
wants silu, and the activation on the other operand. Nothing here is a parity claim, and no arm of
it is bit-exact against anything. It answers one question: does the fused shape beat 0.4178 ms/chunk,
and how much of the answer is F1's DRAM destination.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                     # noqa: E402
import ttnn                                                                      # noqa: E402
import tt_bio.tenstorrent as T                                                   # noqa: E402
from tt_bio import trimul_tail as TT                                             # noqa: E402
from tt_bio import mm_generic as MG                                              # noqa: E402

CHUNK_CALLS_PER_FOLD = 16768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cka = MG.ckc_args(ckc)
    torch.manual_seed(0)
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=DR)

    x = dram(torch.randn(1, a.rows, a.w, a.c, dtype=torch.bfloat16))
    nw = dram(torch.randn(a.c, dtype=torch.bfloat16))
    nb = dram(torch.randn(a.c, dtype=torch.bfloat16))
    w1 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))
    w2 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))
    w3 = dram(torch.randn(a.dff, a.c, dtype=torch.bfloat16))

    def ln(t):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc,
                               memory_config=L1)

    def lin(t, w, act=None, mc=L1):
        return ttnn.linear(t, w, activation=act, compute_kernel_config=ckc, memory_config=mc,
                           core_grid=T.CORE_GRID_MAIN)

    def full():
        xn = ln(x)
        x1 = lin(xn, w1, "silu")
        x2 = lin(xn, w2)
        ttnn.deallocate(xn)
        p = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        o = lin(p, w3, mc=DR)
        ttnn.deallocate(p)
        return o

    def f1_dram():
        xn = ln(x)
        p = TT.fused_tail(xn, xn, w1, w2, cka, grid)
        ttnn.deallocate(xn)
        if p is None:
            raise SystemExit("F1 declined the Transition chunk; E6 said it would not")
        o = lin(p, w3, mc=DR)
        ttnn.deallocate(p)
        return o

    def f1_only():
        """The fused GEMM pair alone, so fc3's cost can be separated from it."""
        xn = ln(x)
        p = TT.fused_tail(xn, xn, w1, w2, cka, grid)
        ttnn.deallocate(xn)
        return p

    def pair_only():
        """The shipped fc1+fc2+multiply alone, the leg f1_only replaces."""
        xn = ln(x)
        x1 = lin(xn, w1, "silu")
        x2 = lin(xn, w2)
        ttnn.deallocate(xn)
        p = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        return p

    ARMS = [("full", full), ("f1_dram", f1_dram), ("f1_only", f1_only),
            ("pair_only", pair_only), ("full_2", full)]

    def time_arm(fn):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.trials):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(a.reps):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / a.reps)
        return statistics.median(ts), min(ts), max(ts)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": 1, "grid": list(grid),
           "shape": [1, a.rows, a.w, a.c], "d_ff": a.dff, "reps": a.reps, "trials": a.trials,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "note": "PERF PROTOTYPE. F1 computes p*sigmoid(g), swiglu wants silu(p)*g. "
                   "Numerically wrong on purpose; nothing here is a parity claim.",
           "arms": {}}
    for name, fn in ARMS:
        med, lo, hi = time_arm(fn)
        res["arms"][name] = {"ms": round(1e3 * med, 5), "min_ms": round(1e3 * lo, 5),
                             "max_ms": round(1e3 * hi, 5)}
        print(f"{name:10s} {1e3*med:8.4f} ms/chunk  [{1e3*lo:.4f}, {1e3*hi:.4f}]", flush=True)

    A = res["arms"]
    res["aa_floor_ms"] = round(abs(A["full"]["ms"] - A["full_2"]["ms"]), 5)
    d_chain = A["full"]["ms"] - A["f1_dram"]["ms"]
    d_leg = A["pair_only"]["ms"] - A["f1_only"]["ms"]
    res["delta_whole_chain_ms"] = round(d_chain, 5)
    res["delta_leg_only_ms"] = round(d_leg, 5)
    res["landing_s_per_fold_dram_out"] = round(d_chain * CHUNK_CALLS_PER_FOLD / 1e3, 4)
    res["leg_s_per_fold"] = round(d_leg * CHUNK_CALLS_PER_FOLD / 1e3, 4)
    res["dram_destination_penalty_ms"] = round(d_leg - d_chain, 5)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nA/A floor {res['aa_floor_ms']:.5f} ms/chunk")
    print(f"whole chain, F1 with its DRAM output: {d_chain:+.5f} ms/chunk = "
          f"{res['landing_s_per_fold_dram_out']:+.4f} s/fold")
    print(f"the replaced leg alone:               {d_leg:+.5f} ms/chunk = "
          f"{res['leg_s_per_fold']:+.4f} s/fold")
    print(f"cost of F1's DRAM destination:        {res['dram_destination_penalty_ms']:.5f} ms/chunk")
    print("wrote", a.out)


main()
