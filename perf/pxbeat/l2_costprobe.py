#!/usr/bin/env python3
"""L2 pre-gate: what does one more per-tile SFPU pass cost INSIDE F1?

E12's lesson is that a subtractive screen prices a fusion's prize and says nothing about its cost.
L2's prize is measured (E5: 0.8557 ms/call deletable). Its cost is the layer_norm math moved from
its own DRAM-bound op, where it hides behind 0.831 ms of traffic, into F1's compute pipeline, where
it does not hide behind anything.

F1 already carries two compile-time switches that add or remove exactly one per-tile SFPU pass over
the same 65536-tile stream the norm prologue would run on:

    SKIP_SIGMOID  1 -> 0  adds one sigmoid + one round trip through `sig_cb`
    ROUND         0 -> 2  adds the integer bf16 rounding chain

So the two deltas price a "per-tile SFPU pass" in this kernel, at this shape, with no new kernel
code. The norm prologue is ~5 such passes (sub_bcast, mul, mul_bcast, mul_bcast_rows,
add_bcast_rows) plus two REDUCE_ROW reductions over 8 tiles and one rsqrt per tile-row.

Budget, from E5's own arms: the fused arm must land at or under 3.2055 - 0.25 = 2.9555 ms/call,
and F1 alone is 2.3499, so the prologue's marginal cost has 0.6056 ms/call to spend.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1].name == "pxbeat") else Path("/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200-p2")
sys.path.insert(0, str(REPO))

import torch                                                                     # noqa: E402
import ttnn                                                                      # noqa: E402
import tt_bio.tenstorrent as T                                                   # noqa: E402
from tt_bio import trimul_tail as TT                                             # noqa: E402
from tt_bio import mm_generic as MG                                              # noqa: E402

F1_CALLS_PER_FOLD = 1048
PROLOGUE_PASSES = 5          # the eltwise passes a layer_norm prologue adds per in0 tile
BUDGET_MS_PER_CALL = 0.6056  # 3.2055 (E5 full) - 0.25 (registered gate) - 2.3499 (E5 no_ln)


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
        raise SystemExit(f"the probe's operands do not reach F1 ({why})")

    def variant(round_, skip_sig):
        def fn():
            TT.ROUND, TT.SKIP_SIGMOID = round_, skip_sig
            return TT.fused_tail(x, xg, wa, wb, cka, grid)
        return fn

    def ln_only():
        return ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc)

    def full():
        xn = ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc)
        TT.ROUND, TT.SKIP_SIGMOID = 2, 0
        o = TT.fused_tail(xn, xg, wa, wb, cka, grid)
        ttnn.deallocate(xn)
        return o

    ARMS = [("full",       full),
            ("f1",         variant(2, 0)),
            ("f1_nosig",   variant(2, 1)),
            ("f1_noround", variant(0, 0)),
            ("f1_bare",    variant(0, 1)),
            ("ln_only",    ln_only),
            ("f1_2",       variant(2, 0)),
            ("full_2",     full)]

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
           "budget_ms_per_call": BUDGET_MS_PER_CALL, "prologue_passes": PROLOGUE_PASSES,
           "arms": {}}
    for name, fn in ARMS:
        med, lo, hi = time_arm(fn)
        res["arms"][name] = {"ms": round(1e3 * med, 5), "min_ms": round(1e3 * lo, 5),
                             "max_ms": round(1e3 * hi, 5)}
        print(f"{name:11s} {1e3*med:9.4f} ms/call  [{1e3*lo:.4f}, {1e3*hi:.4f}]", flush=True)

    A = res["arms"]
    res["aa_floor_f1_ms"] = round(abs(A["f1"]["ms"] - A["f1_2"]["ms"]), 5)
    res["aa_floor_full_ms"] = round(abs(A["full"]["ms"] - A["full_2"]["ms"]), 5)
    sig = A["f1"]["ms"] - A["f1_nosig"]["ms"]
    rnd = A["f1"]["ms"] - A["f1_noround"]["ms"]
    both = A["f1"]["ms"] - A["f1_bare"]["ms"]
    res["sigmoid_pass_ms"] = round(sig, 5)
    res["round_pass_ms"] = round(rnd, 5)
    res["both_passes_ms"] = round(both, 5)
    unit = max(sig, rnd)
    res["per_pass_unit_ms"] = round(unit, 5)
    res["predicted_prologue_ms"] = round(PROLOGUE_PASSES * unit, 5)
    res["deletable_ms"] = round(A["full"]["ms"] - A["f1"]["ms"], 5)
    res["verdict"] = ("GO to the prototype" if PROLOGUE_PASSES * unit < BUDGET_MS_PER_CALL
                      else "NO-GO on cost")
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nA/A floor  f1 {res['aa_floor_f1_ms']:.5f}   full {res['aa_floor_full_ms']:.5f} ms/call")
    print(f"one sigmoid pass {sig:.4f}   one round pass {rnd:.4f}   both {both:.4f} ms/call")
    print(f"prologue at {PROLOGUE_PASSES} passes x {unit:.4f} = {PROLOGUE_PASSES*unit:.4f} ms/call "
          f"against a {BUDGET_MS_PER_CALL} budget -> {res['verdict']}")
    print(f"this session's deletable (full - f1) {res['deletable_ms']:.4f} ms/call "
          f"(E5 read 0.8557)")
    print("wrote", a.out)


main()
