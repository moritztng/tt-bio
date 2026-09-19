#!/usr/bin/env python3
"""Two controls the first session could not supply, on the bare n-ladder.

1. A WRITE-ONLY ROOF THAT IS NOT A COPY.  Session 1 measured the write roof with an L1 -> DRAM
   clone, and `trix-floor` found the copy instrument reads ~9 % low on the combined roof.  A
   binary add with BOTH operands in L1 and the result in DRAM moves 0 DRAM read bytes and 1 W,
   on the same kernel class as the 2R+1W starved add that gives the combined roof -- so the two
   roofs are read on one instrument and the copy's bias cancels out of the comparison.

2. IS ANY OF THE WRITE OVERLAPPED WITH COMPUTE?  ship vs (out_l1 then an explicit clone of the
   SAME 8.389 MB to DRAM).  If the shipped op pipelines its write against its own compute, it
   must beat the two-step by the overlapped part.  If it equals the two-step, it overlaps
   nothing.  8.389 MB exactly, so no size correction is needed.
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import clk                                                                     # noqa: E402
import ttnn                                                                    # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
B, M, K, N = 16, 512, 128, 512


def fit(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    a = my - b*mx
    return a, b, max(abs(y-(a+b*x)) for x, y in zip(xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", default=str(HERE / "roof_s2.json"))
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T

    device = T.get_device()
    held = clk.force(a.clock, clk.nodes_open_by_this_process())
    t0 = time.time()
    while time.time()-t0 < 10.0 and not all(clk.aiclk(n) >= a.clock-5 for n in held):
        time.sleep(0.01)
    if any(clk.aiclk(n) < a.clock-5 for n in held):
        print("REFUSING to measure: %r" % ({n: clk.aiclk(n) for n in held},)); return 2
    print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)
    sampler = clk.Sampler(held[0])
    GRID = T.CORE_GRID_MAIN
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)

    def dev(t, mc=DRAM):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=mc)

    R = {"arms": {}}

    def ladder(fn, ns):
        pts = []
        for n in ns:
            ts = []
            for i in range(a.reps + a.warm):
                ttnn.synchronize_device(device)
                s = time.perf_counter()
                rs = [fn() for _ in range(n)]
                ttnn.synchronize_device(device)
                e = time.perf_counter()
                for r in rs:
                    if isinstance(r, ttnn.Tensor):
                        ttnn.deallocate(r)
                if i >= a.warm:
                    ts.append((e-s)*1e3)
            pts.append((n, st.median(ts)))
        L, c, res = fit([p[0] for p in pts], [p[1] for p in pts])
        return {"pts": pts, "L": L, "c": c, "resid": res}

    def run(name, fn, ns, **meta):
        r = ladder(fn, ns); r.update(meta); R["arms"][name] = r
        ex = "  %7.1f GB/s" % (meta["bytes"]/(r["c"]*1e-3)/1e9) if "bytes" in meta else ""
        print("  %-26s c = %.5f ms  resid %.5f%s" % (name, r["c"], r["resid"], ex), flush=True)
        return r

    hold = []
    # ---- 1. the write roof on the add instrument, 96 MiB written --------------------------
    # 6144x4096 bf16 = 48 MiB per operand; both in L1 (96 MiB of 165 MiB aggregate), result DRAM.
    print("WRITE ROOF, add instrument (0 DRAM read, 1 DRAM write):", flush=True)
    EL = 6144 * 4096
    pl, ql = dev(torch.randn(6144, 4096)*0.05, L1), dev(torch.randn(6144, 4096)*0.05, L1)
    pd, qd = dev(torch.randn(6144, 4096)*0.05), dev(torch.randn(6144, 4096)*0.05)
    hold += [pl, ql, pd, qd]
    run("add_0r1w_out_dram", lambda: ttnn.add(pl, ql, memory_config=DRAM), (1, 2, 4),
        bytes=float(EL*2))
    run("add_2r1w_out_dram", lambda: ttnn.add(pd, qd, memory_config=DRAM), (1, 2, 4),
        bytes=3.0*EL*2)
    for t in (pl, ql):            # 96 MiB of L1 operands must go before the 8 MB L1 arms
        ttnn.deallocate(t)

    # ---- 2. does the shipped op overlap its write with its compute? ------------------------
    print("OVERLAP, key A vs the same work in two explicit steps:", flush=True)
    x_d, w_d = dev(torch.randn(1, B, M, K)*0.05), dev(torch.randn(K, N)*0.05)
    o_l1 = dev(torch.randn(1, B, M, N)*0.05, L1)
    hold += [x_d, w_d, o_l1]

    def lin(mc):
        return lambda: ttnn.linear(x_d, w_d, compute_kernel_config=KC, memory_config=mc,
                                   dtype=ttnn.bfloat16, core_grid=GRID)
    ship = run("key_ship", lin(DRAM), (1, 2, 4, 8, 16))
    outl1 = run("key_out_l1", lin(L1), (1, 2, 4, 8))
    mv = run("move_8mb_l1_to_dram", lambda: ttnn.clone(o_l1, memory_config=DRAM), (1, 2, 4, 8),
             bytes=float(B*M*N*2))
    two = outl1["c"] + mv["c"]
    print("  => ship %.5f  vs  out_l1 + explicit move %.5f  (%.4fx).  A fused op that "
          "overlapped its write would be FASTER than the two-step by the overlapped part."
          % (ship["c"], two, two / ship["c"]), flush=True)
    R["ship"], R["two_step"] = ship["c"], two

    R["clock"] = sampler.stop()
    print("AICLK during: %r" % (R["clock"],), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
