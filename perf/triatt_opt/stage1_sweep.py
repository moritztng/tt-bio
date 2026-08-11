#!/usr/bin/env python3
"""triatt-absolute-optimal, EXECUTION pass: the three bit-exact Stage-1 levers, swept.

Card 2 on qb2, P300c, 11x10 = 110 cores, ttnn 0.68.0, branch wk/triatt-absolute-optimal.
Stage 1 of state/triatt-absolute-optimal.md 5.3. Every lever here must be bit-exact.

PREDICTIONS, WRITTEN BEFORE THE RUN:

S1  `out` (K=256, N=256) via minimal_matmul with the 4-block config beats the shipped
    _pair_proj_linear by 1.20-1.30x at 512 aa, and is torch.equal to it. The planning pass
    measured 0.989 -> 0.787 at 512 only; I expect the ratio to HOLD at every size >= 384 and
    to shrink below 320, where M is small enough that the 2-block config's finer work split
    matters.
S2  the nt=8 4-block config (4,8,1,4,1) beats the shipped 2-block (2,8,1,2,1) at every
    M >= 65536 and loses below M ~ 16384. Both bit-exact (K_block=8=full K in both, so the
    contraction order is identical).
S3  the fused projection N=1056 beats the three separate matmuls by 1.30-1.45x at 512 aa and
    stays above 1.15x at all six sizes. The N=1024 pathology (a 1.16x LOSS) does not recur at
    1056; if it does at any size, the fused path must be size-gated.
S4  every fused column slice is torch.equal to its separate matmul at all six sizes.
S5  the three levers together are 2.4-2.6 ms/block of the measured 37.327, i.e. 1.065-1.075x
    on the sub-block. Below the fold wall's resolution, so it is claimed on the synced
    PairformerLayer block wall, never on a pair of fold times.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device

RES = {"predictions": __doc__, "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "0.68.0"}


def timed(fn, warm=3, reps=5):
    dev = T.get_device()
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        del r
    return st.median(ts)


def mm_cfg(mt, kt, nt, M, K, N, sh, sw):
    if mt % M or nt % N or kt % K:
        return None
    return ttnn.MinimalMatmulConfig(
        M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN))


def mk(dev, shape, dt=ttnn.bfloat16, mc=None):
    t = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt,
                           memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="298,320,384,512,576,640")
    ap.add_argument("--out", default="perf/triatt_opt/stage1_sweep.json")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    dev = get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)
    RES["grid"] = list(T.COMPUTE_GRID_MAIN)
    RES["loadavg"] = open("/proc/loadavg").read().strip()

    c = 256
    # ---------- S2: the _MM_BLOCK nt=8 sweep, over M ---------------------------------------
    s2 = []
    w8 = mk(dev, (c, 256))
    for mt in (256, 512, 1024, 2048, 4096, 8192, 12800):
        x = mk(dev, (mt * 32, c))
        row = {"m_tiles": mt, "M": mt * 32}
        for name, blk in (("blk2", (2, 8, 1, 2, 1)), ("blk4", (4, 8, 1, 4, 1))):
            cfg = mm_cfg(mt, 8, 8, *blk)
            if cfg is None:
                row[name] = None
                continue
            row[name] = timed(lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w8, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=cfg)) * 1e3
        if row.get("blk2") and row.get("blk4"):
            row["blk4_vs_blk2"] = row["blk2"] / row["blk4"]
            o2 = ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w8, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=mm_cfg(mt, 8, 8, 2, 8, 1, 2, 1))
            o4 = ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w8, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=mm_cfg(mt, 8, 8, 4, 8, 1, 4, 1))
            row["bit_exact"] = bool(torch.equal(ttnn.to_torch(o2), ttnn.to_torch(o4)))
            ttnn.deallocate(o2); ttnn.deallocate(o4)
        ttnn.deallocate(x)
        s2.append(row)
        print("S2", row, flush=True)
    ttnn.deallocate(w8)
    RES["s2_mm_block_sweep"] = s2

    # ---------- S1 + S3 + S4: per size ------------------------------------------------------
    per_size = []
    for S in sizes:
        mt = S * (-(-S // 32))
        row = {"S": S, "m_tiles": mt, "M": mt * 32}
        x = mk(dev, (S, S, c))
        wq = mk(dev, (c, 768)); wg = mk(dev, (c, 256)); wb = mk(dev, (c, 32))
        wf = ttnn.from_torch(
            torch.cat([ttnn.to_torch(wq), ttnn.to_torch(wg), ttnn.to_torch(wb)], dim=1),
            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        def mm(w, blk):
            nt = -(-int(w.shape[-1]) // 32)
            cfg = mm_cfg(mt, 8, nt, *blk)
            return ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=cfg)

        # --- S3: separate vs fused
        row["qkv_ms"] = timed(lambda: mm(wq, (4, 8, 1, 4, 1))) * 1e3
        row["g_ms"] = timed(lambda: mm(wg, (2, 8, 1, 2, 1))) * 1e3
        row["g_ms_blk4"] = timed(lambda: mm(wg, (4, 8, 1, 4, 1))) * 1e3
        row["bias_ms"] = timed(lambda: T._pair_proj_linear(x, wb, ckc, ttnn.bfloat16)) * 1e3
        row["fused_ms"] = timed(lambda: mm(wf, (4, 8, 1, 4, 1))) * 1e3
        row["sep_ms"] = row["qkv_ms"] + min(row["g_ms"], row["g_ms_blk4"]) + row["bias_ms"]
        row["fused_speedup"] = row["sep_ms"] / row["fused_ms"]

        # --- S4: parity of the fused slices
        oq = mm(wq, (4, 8, 1, 4, 1)); og = mm(wg, (2, 8, 1, 2, 1))
        ob = T._pair_proj_linear(x, wb, ckc, ttnn.bfloat16)
        of = mm(wf, (4, 8, 1, 4, 1))
        tf = ttnn.to_torch(of)
        row["eq_qkv"] = bool(torch.equal(tf[..., 0:768], ttnn.to_torch(oq)))
        row["eq_g"] = bool(torch.equal(tf[..., 768:1024], ttnn.to_torch(og)))
        row["eq_bias"] = bool(torch.equal(tf[..., 1024:1056], ttnn.to_torch(ob)))
        for t in (oq, og, ob, of):
            ttnn.deallocate(t)

        # --- S1: the out projection, shipped vs minimal_matmul
        o_in = mk(dev, (S, S, c)); wo = mk(dev, (c, 256))
        row["out_shipped_ms"] = timed(
            lambda: T._pair_proj_linear(o_in, wo, ckc, ttnn.bfloat16, l1_out=False)) * 1e3
        row["out_shipped_l1_ms"] = timed(
            lambda: T._pair_proj_linear(o_in, wo, ckc, ttnn.bfloat16, l1_out=True)) * 1e3
        for name, blk in (("out_mm2_ms", (2, 8, 1, 2, 1)), ("out_mm4_ms", (4, 8, 1, 4, 1))):
            cfg = mm_cfg(mt, 8, 8, *blk)
            row[name] = None if cfg is None else timed(
                lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                    input_tensor=o_in, weight_tensor=wo, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=cfg)) * 1e3
        a_ = T._pair_proj_linear(o_in, wo, ckc, ttnn.bfloat16, l1_out=False)
        b_ = ttnn.experimental.minimal_matmul(
            input_tensor=o_in, weight_tensor=wo, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=mm_cfg(mt, 8, 8, 4, 8, 1, 4, 1))
        row["eq_out"] = bool(torch.equal(ttnn.to_torch(a_), ttnn.to_torch(b_)))
        best_out = min(v for k, v in row.items()
                       if k.startswith("out_mm") and isinstance(v, float))
        row["out_speedup"] = min(row["out_shipped_ms"], row["out_shipped_l1_ms"]) / best_out
        for t in (a_, b_, o_in, wo, x, wq, wg, wb, wf):
            ttnn.deallocate(t)

        row["saving_ms_per_call"] = (row["sep_ms"] - row["fused_ms"]) + (
            min(row["out_shipped_ms"], row["out_shipped_l1_ms"]) - best_out)
        per_size.append(row)
        print("SIZE", json.dumps(row), flush=True)
    RES["per_size"] = per_size

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(RES, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
