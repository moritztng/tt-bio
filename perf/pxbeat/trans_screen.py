#!/usr/bin/env python3
"""L3 of `protenix-v2-beat-dgx-h200` §4: the subtractive screen on `Transition`, before any kernel.

`Transition.__call__`'s swiglu, at the chunk shape the 512 aa fold actually issues
(`perf/pxbeat/lin_l1_512_c1.json`: [1,16,512,256], 16768 chunk-calls of each linear per fold):

    x_norm = layer_norm(x)            -> L1
    x_1    = linear(x_norm, fc1, activation="silu")   -> L1
    x_2    = linear(x_norm, fc2)                      -> L1
    x      = multiply_(x_1, x_2)
    out    = linear(x, fc3)                           -> DRAM

A fused kernel absorbs the norm and the multiply into the GEMM prologue/epilogue and stops
materialising `x_norm`, `x_1` and `x_2`. It still has to run their MATH, so the cost of those two
ops measured by subtraction is an UPPER BOUND on what the fusion can delete, which is exactly what a
kill gate wants: if the upper bound misses, the lever is dead without writing a line of kernel.

Method, from `opendde-to-4x-per-dollar` §9.1, because that design is what caught the oversync error:
ONE sync around R repeats of the whole chain per arm, never a sync per op
(`tt-bio-isolated-op-timing-oversync-inflates-cost` prices per-op timing ~2x high, found
independently twice in one day). `full_2` runs LAST as the A/A control, so the arm spread is
measured in the same session as the effect.

Registered kill gate, written before the run (state doc §4 L3): GO to a `generic_op` build only if
the screen shows >= 0.75 ms deletable per Transition CALL. At 959 Transition calls and 16768
chunk-calls per fold that is 17.49 chunks/call, so the gate in the units this screen measures is

    >= 0.75 / 17.49 = 0.0429 ms per chunk   ==   >= 0.719 s per fold

Below it, NO-GO with a number.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                     # noqa: E402
import ttnn                                                                      # noqa: E402
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN                        # noqa: E402

CHUNK_CALLS_PER_FOLD = 16768
TRANSITION_CALLS_PER_FOLD = 959
GATE_MS_PER_CALL = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)      # TRANSITION_H_CHUNK_SIZE
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=40, help="chain repeats inside ONE sync")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    torch.manual_seed(0)

    def dram(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = dram(torch.randn(1, a.rows, a.w, a.c, dtype=torch.bfloat16))
    nw = dram(torch.randn(a.c, dtype=torch.bfloat16))
    nb = dram(torch.randn(a.c, dtype=torch.bfloat16))
    w1 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))
    w2 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))
    w3 = dram(torch.randn(a.dff, a.c, dtype=torch.bfloat16))

    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

    def ln(t):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5,
                               compute_kernel_config=ckc, memory_config=L1)

    def lin(t, w, act=None, mc=L1):
        return ttnn.linear(t, w, activation=act, compute_kernel_config=ckc, memory_config=mc,
                           core_grid=CORE_GRID_MAIN)

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

    def no_ln():
        """The norm gone entirely. full - no_ln is the norm op's whole cost, math included."""
        x1 = lin(x, w1, "silu")
        x2 = lin(x, w2)
        p = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        o = lin(p, w3, mc=DR)
        ttnn.deallocate(p)
        return o

    def no_mul():
        """The multiply gone. fc3 consumes x_1 directly; x_2 is still produced and dropped."""
        xn = ln(x)
        x1 = lin(xn, w1, "silu")
        x2 = lin(xn, w2)
        ttnn.deallocate(xn)
        ttnn.deallocate(x2)
        o = lin(x1, w3, mc=DR)
        ttnn.deallocate(x1)
        return o

    def no_fc2():
        """fc2 gone: prices the SECOND read of x_norm plus one d_ff-wide L1 write."""
        xn = ln(x)
        x1 = lin(xn, w1, "silu")
        ttnn.deallocate(xn)
        p = ttnn.multiply_(x1, x1)
        o = lin(p, w3, mc=DR)
        ttnn.deallocate(p)
        return o

    def mm_floor():
        """The three GEMMs alone, no norm, no multiply: the irreducible arithmetic of the chain."""
        x1 = lin(x, w1, "silu")
        x2 = lin(x, w2)
        ttnn.deallocate(x2)
        o = lin(x1, w3, mc=DR)
        ttnn.deallocate(x1)
        return o

    ARMS = [("full", full), ("no_ln", no_ln), ("no_mul", no_mul), ("no_fc2", no_fc2),
            ("mm_floor", mm_floor), ("full_2", full)]

    def time_arm(fn):
        for _ in range(2):                                  # compile + warm, outside the wall
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
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": 1,
           "shape": [1, a.rows, a.w, a.c], "d_ff": a.dff, "reps": a.reps, "trials": a.trials,
           "grid": [CORE_GRID_MAIN.x, CORE_GRID_MAIN.y],
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "chunk_calls_per_fold": CHUNK_CALLS_PER_FOLD,
           "transition_calls_per_fold": TRANSITION_CALLS_PER_FOLD,
           "gate_ms_per_call": GATE_MS_PER_CALL, "arms": {}}
    for name, fn in ARMS:
        med, lo, hi = time_arm(fn)
        res["arms"][name] = {"ms": round(1e3 * med, 5), "min_ms": round(1e3 * lo, 5),
                             "max_ms": round(1e3 * hi, 5)}
        print(f"{name:10s} {1e3*med:8.4f} ms/chunk  [{1e3*lo:.4f}, {1e3*hi:.4f}]", flush=True)

    A = res["arms"]
    aa = abs(A["full"]["ms"] - A["full_2"]["ms"])
    res["aa_floor_ms"] = round(aa, 5)
    ln_ms = A["full"]["ms"] - A["no_ln"]["ms"]
    mul_ms = A["full"]["ms"] - A["no_mul"]["ms"]
    fc2_ms = A["full"]["ms"] - A["no_fc2"]["ms"]
    res["legs_ms_per_chunk"] = {"layer_norm": round(ln_ms, 5), "multiply_": round(mul_ms, 5),
                                "fc2": round(fc2_ms, 5),
                                "norm+mul (fusion upper bound)": round(ln_ms + mul_ms, 5)}
    per_chunk = ln_ms + mul_ms
    chunks_per_call = CHUNK_CALLS_PER_FOLD / TRANSITION_CALLS_PER_FOLD
    res["deletable_ms_per_call"] = round(per_chunk * chunks_per_call, 4)
    res["deletable_s_per_fold"] = round(per_chunk * CHUNK_CALLS_PER_FOLD / 1e3, 4)
    res["gate_s_per_fold"] = round(GATE_MS_PER_CALL * TRANSITION_CALLS_PER_FOLD / 1e3, 4)
    res["verdict"] = ("GO to a generic_op build"
                      if res["deletable_ms_per_call"] >= GATE_MS_PER_CALL else "NO-GO")
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nA/A floor (full vs full_2) {aa:.5f} ms/chunk")
    print(f"layer_norm {ln_ms:.5f}  multiply_ {mul_ms:.5f}  fc2 {fc2_ms:.5f} ms/chunk")
    print(f"fusion UPPER BOUND {per_chunk:.5f} ms/chunk = "
          f"{res['deletable_ms_per_call']:.4f} ms/call = {res['deletable_s_per_fold']:.4f} s/fold")
    print(f"gate {GATE_MS_PER_CALL} ms/call = {res['gate_s_per_fold']} s/fold -> {res['verdict']}")
    print("wrote", a.out)


main()
