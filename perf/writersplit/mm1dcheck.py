#!/usr/bin/env python3
"""The transcription gate: `tt_bio/mm1d_generic.py` against the native 1D mcast_in1 matmul.

mm_generic cleared this same bar for `minimal_matmul` at 1.018x with `torch.equal` True.  The row's
kill criterion is the same one: a faithful transcription reads within ~2 % of the native op and is
bit-exact against it.  Both arms take the SAME explicit MatmulMultiCoreReuseMultiCast1DProgramConfig,
so the comparison isolates the transcription and not the block-config chooser.

Correctness is measured with no clock requirement; timing (--time) refuses to run unless the clock
holds, since a number without a DURING-sampled clock is not a measurement.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import clk  # noqa: E402
import ttnn  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
B, M, N = 16, 512, 512

#: (K, in0_block_w, per_core_M, per_core_N, out_block_h, out_block_w, subblock_h, subblock_w).
#: Both arms take these, so the chooser is out of the comparison.  per_core_M=3 is div_up(256, 110) on the 11x10 grid, what
#: create_matmul_1d_systolic_array_program_config picks here; 256 is not a multiple of 3, so the
#: ragged last M block is covered rather than dodged.
CASES = {
    4: (128, 1, 3, 16, 3, 16, 1, 4),
    32: (1024, 4, 3, 16, 3, 16, 1, 4),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", default=str(HERE / "mm1dcheck.json"))
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T
    from tt_bio.mm1d_generic import generic_mm1d

    device = T.get_device()
    R = {"cases": {}, "time": bool(a.time)}

    if a.time:
        held = clk.force(a.clock, clk.nodes_open_by_this_process())
        t0 = time.time()
        while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
            time.sleep(0.01)
        reached = {n: clk.aiclk(n) for n in held}
        if any(v < a.clock - 5 for v in reached.values()):
            print("REFUSING to measure: %r" % (reached,))
            return 2
        print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)
        sampler = clk.Sampler(held[0])

    GRID = T.CORE_GRID_MAIN
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    CKC = (ttnn.MathFidelity.HiFi4, False, True, True)
    torch.manual_seed(0)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)

    arms = {}
    for kt, (K, in0_bw, pcM, pcN, obh, obw, sbh, sbw) in CASES.items():
        x = dev(torch.randn(1, B, M, K) * 0.05)
        w = dev(torch.randn(K, N) * 0.05)
        pcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(GRID.x, GRID.y),
            in0_block_w=in0_bw, out_subblock_h=sbh, out_subblock_w=sbw,
            out_block_h=obh, out_block_w=obw, per_core_M=pcM, per_core_N=pcN,
            fuse_batch=True, mcast_in0=False)
        pc = ((GRID.x, GRID.y), in0_bw, sbh, sbw, obh, obw, pcM, pcN)
        out_g = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(ttnn.Shape([1, B, M, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                            ttnn.BufferType.DRAM), device)
        arms[kt] = (x, w, pcfg, pc, out_g)

    def native(kt):
        x, w, pcfg, _, _ = arms[kt]
        return ttnn.linear(x, w, program_config=pcfg, compute_kernel_config=KC,
                           memory_config=DRAM, dtype=ttnn.bfloat16)

    def generic(kt):
        x, w, _, pc, out_g = arms[kt]
        return generic_mm1d(device, x, w, out_g, pc, CKC)

    # ---- correctness -----------------------------------------------------------------------
    for kt in CASES:
        ref = native(kt)
        got = generic(kt)
        ttnn.synchronize_device(device)
        tr, tg = ttnn.to_torch(ref), ttnn.to_torch(got)
        eq = bool(torch.equal(tr, tg))
        mad = float((tr.float() - tg.float()).abs().max())
        x, w, _, _, _ = arms[kt]
        fp32 = (ttnn.to_torch(x).float().reshape(B * M, -1) @ ttnn.to_torch(w).float())
        def pcc(t):
            u, v = t.float().reshape(-1), fp32.reshape(-1)
            return float(torch.corrcoef(torch.stack([u, v]))[0, 1])
        R["cases"]["kt%d" % kt] = {"torch_equal": eq, "max_abs_diff": mad,
                                   "pcc_native": pcc(tr), "pcc_generic": pcc(tg)}
        print("kt=%-3d torch.equal %s  max|diff| %.3e  PCC native %.6f generic %.6f"
              % (kt, eq, mad, pcc(tr), pcc(tg)), flush=True)
        ttnn.deallocate(ref)

    if not a.time:
        Path(a.out).write_text(json.dumps(R, indent=1))
        return 0 if all(c["torch_equal"] for c in R["cases"].values()) else 1

    # ---- timing, arms interleaved in one process -------------------------------------------
    def fit(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((v - mx) ** 2 for v in xs)
        sxy = sum((v - mx) * (u - my) for v, u in zip(xs, ys))
        b = sxy / sxx
        return my - b * mx, b, max(abs(u - (my - b * mx + b * v)) for v, u in zip(xs, ys))

    def ladder(fn, kt, ns=(1, 2, 4, 8, 16)):
        pts = []
        for n in ns:
            ts = []
            for i in range(a.reps + a.warm):
                ttnn.synchronize_device(device)
                s = time.perf_counter()
                rs = [fn(kt) for _ in range(n)]
                ttnn.synchronize_device(device)
                e = time.perf_counter()
                for r in rs:
                    if r is not arms[kt][4]:
                        ttnn.deallocate(r)
                if i >= a.warm:
                    ts.append((e - s) * 1e3)
            pts.append((n, st.median(ts)))
        L, c, res = fit([p[0] for p in pts], [p[1] for p in pts])
        return {"pts": pts, "L": L, "c": c, "resid": res}

    for kt in CASES:
        for _ in range(3):
            native(kt); generic(kt)
        ttnn.synchronize_device(device)

    plan = []
    for _ in range(a.rounds):
        plan += [("native", native), ("generic", generic)]
    plan.append(("native_aa", native))
    for name, fn in plan:
        for kt in CASES:
            key = "%s_kt%d" % (name, kt)
            res = ladder(fn, kt)
            R.setdefault("arms", {}).setdefault(key, []).append(res)
            print("  %-16s c = %.5f ms  L = %.5f  resid %.5f"
                  % (key, res["c"], res["L"], res["resid"]), flush=True)

    R["clock"] = sampler.stop()
    print("AICLK during: %r" % (R["clock"],), flush=True)
    best = lambda k: st.median([r["c"] for r in R["arms"][k]])
    for kt in CASES:
        nv, gn = best("native_kt%d" % kt), best("generic_kt%d" % kt)
        aa = R["arms"]["native_aa_kt%d" % kt][0]["c"]
        print("kt=%-3d native %.5f ms  generic %.5f ms  generic/native %.4fx  "
              "A/A floor %.3f %%" % (kt, nv, gn, gn / nv, 100 * abs(nv - aa) / aa), flush=True)
        R["cases"]["kt%d" % kt].update(
            {"native_ms": nv, "generic_ms": gn, "ratio": gn / nv,
             "aa_floor_pct": 100 * abs(nv - aa) / aa})
    Path(a.out).write_text(json.dumps(R, indent=1))
    return 0 if all(c["torch_equal"] for c in R["cases"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
