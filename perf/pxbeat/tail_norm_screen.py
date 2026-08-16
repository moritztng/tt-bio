#!/usr/bin/env python3
"""L2 of `protenix-v2-beat-dgx-h200` §4: can the trimul tail's `out_norm` be folded into F1?

At `tenstorrent.py:2626` the tail runs

    x = ttnn.layer_norm(x, out_norm...)          # DRAM out, 134.22 MB at 512 aa
    fused = trimul_tail.fused_tail(x, x_norm_in, out_p_weight, g_out_weight, ...)

so F1 reads straight back what the norm just wrote. Folding the norm into the kernel's first pass
deletes one pair write and one pair read per firing call. The kernel would still run the norm's
MATH, so the norm op's whole measured cost is an UPPER BOUND on the deletion, which is what the
gate needs: if the upper bound misses, the lever is dead before a line of kernel.

Registered kill gate (state doc §4 L2): GO only if >= 0.25 ms/call is deleted on the trimul body
wall. F1 fires on 1048 of 1208 trimul calls per 512 aa fold, so the gate is >= 0.262 s/fold.

Arms, ONE sync around R repeats each, `full_2` last as the A/A control
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).
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

F1_CALLS_PER_FOLD = 1048
GATE_MS_PER_CALL = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    torch.manual_seed(0)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = dram(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16))
    xg = dram(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16))
    wa = dram(torch.randn(a.c, a.c, dtype=torch.bfloat16))
    wb = dram(torch.randn(a.c, a.c, dtype=torch.bfloat16))
    nw = dram(torch.randn(a.c, dtype=torch.bfloat16))
    nb = dram(torch.randn(a.c, dtype=torch.bfloat16))
    cka = MG.ckc_args(ckc)

    why = TT.eligible(x, xg, wa, wb)
    print(f"F1 eligible: {'YES' if why is None else why}", flush=True)
    if why is not None:
        raise SystemExit(f"the screen's operands do not reach F1 ({why}); "
                         "a screen on a path the fold does not take is not a screen")

    def ln(t):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc)

    def full():
        xn = ln(x)
        o = TT.fused_tail(xn, xg, wa, wb, cka, grid)
        ttnn.deallocate(xn)
        return o

    def no_ln():
        """The norm gone entirely: full - no_ln is its whole cost, math and movement together."""
        return TT.fused_tail(x, xg, wa, wb, cka, grid)

    def ln_only():
        return ln(x)

    ARMS = [("full", full), ("no_ln", no_ln), ("ln_only", ln_only), ("full_2", full)]

    def time_arm(fn):
        for _ in range(2):
            o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
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
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": 1, "n": a.n, "c": a.c,
           "grid": list(grid), "reps": a.reps, "trials": a.trials,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "f1_calls_per_fold": F1_CALLS_PER_FOLD, "gate_ms_per_call": GATE_MS_PER_CALL,
           "arms": {}}
    for name, fn in ARMS:
        med, lo, hi = time_arm(fn)
        res["arms"][name] = {"ms": round(1e3 * med, 5), "min_ms": round(1e3 * lo, 5),
                             "max_ms": round(1e3 * hi, 5)}
        print(f"{name:9s} {1e3*med:9.4f} ms/call  [{1e3*lo:.4f}, {1e3*hi:.4f}]", flush=True)

    A = res["arms"]
    res["aa_floor_ms"] = round(abs(A["full"]["ms"] - A["full_2"]["ms"]), 5)
    del_ms = A["full"]["ms"] - A["no_ln"]["ms"]
    res["deletable_ms_per_call"] = round(del_ms, 5)
    res["deletable_s_per_fold"] = round(del_ms * F1_CALLS_PER_FOLD / 1e3, 4)
    res["gate_s_per_fold"] = round(GATE_MS_PER_CALL * F1_CALLS_PER_FOLD / 1e3, 4)
    res["verdict"] = "GO to a kernel" if del_ms >= GATE_MS_PER_CALL else "NO-GO"
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nA/A floor {res['aa_floor_ms']:.5f} ms/call")
    print(f"norm UPPER BOUND {del_ms:.5f} ms/call = {res['deletable_s_per_fold']:.4f} s/fold "
          f"(standalone norm reads {A['ln_only']['ms']:.4f} ms)")
    print(f"gate {GATE_MS_PER_CALL} ms/call = {res['gate_s_per_fold']} s/fold -> {res['verdict']}")
    print("wrote", a.out)


main()
