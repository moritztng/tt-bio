#!/usr/bin/env python3
"""Is the thin-K matmul's K-independent DRAM output write unhidden, and against WHICH roof?

`fixterm-decompose` measured the write at 0.03580 ms = 234.3 GB/s and called it "56.7 % of the
measured roof".  That denominator is the COMBINED 2R+1W roof.  A pure output write is a write-only
stream, and `trix-floor` measured a write-only roof of 245.2 GB/s against a combined 410.3 on its
own part.  If that holds here the write is not at 57 % of anything, it is nearly AT its roof -- a
different mechanism with a different fix.  So this row re-measures both the term and its correct
denominator in ONE session on ONE part.

Instrument: n-ladder only (`synchronize; n calls; synchronize`, fit t(n) = L + n*c).  The synced
single-call bracket charges 0.05142 ms of host launch/drain per bracket on this class of op, which
is comparable to the whole call -- see FIXTERM-HANDOFF section 1.  Every number printed is a
marginal `c`, never a bracket.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import clk                                                                     # noqa: E402
import ttnn                                                                    # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# key A, the fold's own shape: b=16, M=512, K=128, N=512.  Output 16x512x512 bf16 = 8.389 MB and
# it does not depend on K at all, which is the whole point.
B, M, K, N = 16, 512, 128, 512


def fit(xs, ys):
    """Least squares y = a + b*x.  Returns (a, b, max_abs_residual)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    r = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", default=str(HERE / "ladder.json"))
    ap.add_argument("--stages", default="roof,key,k")
    a = ap.parse_args()
    stages = set(a.stages.split(","))

    import torch
    from tt_bio import tenstorrent as T

    device = T.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(a.clock, nodes)
    t0 = time.time()
    while time.time() - t0 < 10.0:
        if all(clk.aiclk(n) >= a.clock - 5 for n in held):
            break
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()):
        print("REFUSING to measure: %r" % (reached,))
        return 2
    print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)
    sampler = clk.Sampler(held[0])

    g = device.compute_with_storage_grid_size()
    GRID = T.CORE_GRID_MAIN
    print("grid %dx%d = %d cores, CORE_GRID_MAIN=%r, arch=%r"
          % (g.x, g.y, g.x * g.y, GRID, device.arch()), flush=True)

    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)

    def dev(t, mc=DRAM):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=mc)

    def ladder(fn, ns):
        """t(n) = L + n*c over the given bracket sizes.  Returns the fit and the raw medians."""
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
                    ts.append((e - s) * 1e3)
            pts.append((n, st.median(ts)))
        L, c, res = fit([p[0] for p in pts], [p[1] for p in pts])
        return {"pts": pts, "L": L, "c": c, "resid": res}

    R = {"clock_target": a.clock, "arch": str(device.arch()),
         "grid": [g.x, g.y], "core_grid_main": [GRID.x, GRID.y], "arms": {}}

    def run(name, fn, ns, **meta):
        r = ladder(fn, ns)
        r.update(meta)
        R["arms"][name] = r
        extra = ""
        if "bytes" in meta:
            extra = "  %7.1f GB/s" % (meta["bytes"] / (r["c"] * 1e-3) / 1e9)
        print("  %-22s c = %.5f ms  L = %.5f  resid %.5f%s"
              % (name, r["c"], r["L"], r["resid"], extra), flush=True)
        return r

    hold = []

    # ---- roofs, measured in THIS session on THIS part -------------------------------------
    # The write roof is the denominator the inherited 56.7 % should have used.  Measured three
    # ways so a single instrument's bias is visible rather than assumed: a starved 2R+1W add
    # (the combined roof, and the one fixterm quoted), an L1 -> DRAM clone (write only) and a
    # DRAM -> L1 clone (read only).  `trix-floor` found the copy instrument reads ~9 % low on
    # the combined roof, so the clone-derived write roof is a LOWER bound on the write roof.
    if "roof" in stages:
        print("ROOF, this part, this session:", flush=True)
        ea = 8192 * 8192
        ad, bd = dev(torch.randn(8192, 8192) * 0.05), dev(torch.randn(8192, 8192) * 0.05)
        hold += [ad, bd]
        run("roof_add_2r1w", lambda: ttnn.add(ad, bd, memory_config=DRAM), (1, 2, 4, 8),
            bytes=3.0 * ea * 2, kind="combined")
        # 48 MiB = 3072 x 4096 bf16, matching trix-floor's own probe size
        w48 = torch.randn(3072, 4096) * 0.05
        c_l1, c_dr = dev(w48, L1), dev(w48, DRAM)
        hold += [c_l1, c_dr]
        run("roof_write_l1_to_dram", lambda: ttnn.clone(c_l1, memory_config=DRAM), (1, 2, 4, 8),
            bytes=float(3072 * 4096 * 2), kind="write")
        run("roof_read_dram_to_l1", lambda: ttnn.clone(c_dr, memory_config=L1), (1, 2, 4),
            bytes=float(3072 * 4096 * 2), kind="read")
        run("roof_clone_dram_dram", lambda: ttnn.clone(c_dr, memory_config=DRAM), (1, 2, 4, 8),
            bytes=2.0 * 3072 * 4096 * 2, kind="combined_copy")

    # ---- the key-A write term ---------------------------------------------------------------
    x_d = dev(torch.randn(1, B, M, K) * 0.05)
    w_d = dev(torch.randn(K, N) * 0.05)
    hold += [x_d, w_d]
    OUT_BYTES = float(B * M * N * 2)
    assert OUT_BYTES == 8388608.0

    def lin(xa, wa, mc):
        return lambda: ttnn.linear(xa, wa, compute_kernel_config=KC, memory_config=mc,
                                   dtype=ttnn.bfloat16, core_grid=GRID)

    if "key" in stages:
        print("KEY A, b=16 M=512 K=128 N=512, marginal per call:", flush=True)
        ship = run("key_ship", lin(x_d, w_d, DRAM), (1, 2, 4, 8, 16))
        run("key_ship_aa", lin(x_d, w_d, DRAM), (1, 2, 4, 8, 16))
        # 16 live 8.39 MB L1 outputs do not fit; the L1 ladder tops out at 8.
        outl1 = run("key_out_l1", lin(x_d, w_d, L1), (1, 2, 4, 8))
        run("key_out_l1_aa", lin(x_d, w_d, L1), (1, 2, 4, 8))
        d = ship["c"] - outl1["c"]
        R["write_term_ms"] = d
        R["write_gbs"] = OUT_BYTES / (d * 1e-3) / 1e9
        print("  => write term %.5f ms, %.1f GB/s on %.3f MB"
              % (d, R["write_gbs"], OUT_BYTES / 1e6), flush=True)

    # ---- does the term shrink as a fraction at larger K? ------------------------------------
    if "k" in stages:
        print("K LADDER, same output, more compute:", flush=True)
        for mult in (1, 4, 8):
            kk = K * mult
            xk = dev(torch.randn(1, B, M, kk) * 0.05)
            wk = dev(torch.randn(kk, N) * 0.05)
            hold += [xk, wk]
            s = run("k%02d_ship" % (kk // 32), lin(xk, wk, DRAM), (1, 2, 4, 8), kt=kk // 32)
            o = run("k%02d_out_l1" % (kk // 32), lin(xk, wk, L1), (1, 2, 4), kt=kk // 32)
            d = s["c"] - o["c"]
            print("  => kt=%d write term %.5f ms = %.1f %% of the call, %.1f GB/s"
                  % (kk // 32, d, 100 * d / s["c"], OUT_BYTES / (d * 1e-3) / 1e9), flush=True)

    R["clock"] = sampler.stop()
    print("AICLK during the whole measuring window: %r" % (R["clock"],), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
