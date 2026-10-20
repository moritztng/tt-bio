#!/usr/bin/env python3
"""minimal_matmul block sweep at Protenix-v2's per-step DiT linears, Wormhole.

The c4 census (WH, cdk2x2 512, .107, AICLK 1000) puts the token DiT's linears at 0.28-0.34 of roof, 4800 calls each
a fold: normal runs them through `k1_linear` (bf16 operands, fp32 dest, HiFi4, in0_block_w 1) for 2.14 s, fast
through `ttnn.linear(core_grid=...)` (bfp8 operands, HiFi2) for 1.17 s. Shapes [512, K] x [K, N] below.

Per mode: the site exactly as the model calls it, then minimal_matmul over a block grid at the site's formats and
fidelity. Normal keeps K block 1 (fp32 dest over more than one K tile is the Wormhole cancellation fault,
dest_guard.py). `exact` is torch.equal against the site; rel L2 is against float64 of the same rounded operands.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch
import ttnn
from tt_bio import tenstorrent as tt

WARM, REPS = 3, 7
SHAPES = [(768, 3072), (768, 1536), (1536, 768), (768, 768)]
MODES = {  # in dtype, weight dtype, out dtype, fidelity, fp32 dest, K blocks
    "normal": ("bfloat16", "bfloat16", "bfloat16", "HiFi4", True, (1,)),
    "fast": ("bfloat8_b", "bfloat8_b", "bfloat16", "HiFi2", False, (1, 2, 4, 8)),
}


def timed(dev, fn, inner):
    for _ in range(WARM):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            o = fn(); ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    o = fn(); ttnn.synchronize_device(dev)
    return statistics.median(ts), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=50)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--modes", default="normal,fast")
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    res = {"arch": str(dev.arch()), "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3],
           "rows_m": a.rows, "inner": a.inner, "rows": []}
    for mode in a.modes.split(","):
        xd, wd, od, fid, f32, kblocks = MODES[mode]
        ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                                     fp32_dest_acc_en=f32, packer_l1_acc=True)
        for K, N in SHAPES:
            torch.manual_seed(0)
            up = lambda t, d: ttnn.from_torch(t, dtype=getattr(ttnn, d), layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
            x, w = up(torch.randn(1, a.rows, K), xd), up(torch.randn(K, N) / K ** 0.5, wd)
            ref64 = ttnn.to_torch(x).double() @ ttnn.to_torch(w).double()
            rel = lambda g: float((g.double().reshape(ref64.shape) - ref64).norm() / ref64.norm())  # noqa: E731
            odt = getattr(ttnn, od)
            if f32:
                site = lambda: tt.k1_linear(x, w, compute_kernel_config=ckc, dtype=odt)  # noqa: E731
            else:
                site = lambda: ttnn.linear(x, w, dtype=odt, compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN)  # noqa: E731
            base, o = timed(dev, site, a.inner)
            ref = ttnn.to_torch(o); ttnn.deallocate(o)
            res["rows"].append({"mode": mode, "K": K, "N": N, "arm": "site", "us": 1e6 * base, "rel_l2_f64": rel(ref)})
            print(f"== {mode} [{a.rows},{K}]x[{K},{N}] site {1e6 * base:.1f} us relL2 {rel(ref):.3e}", flush=True)
            best = None
            for Mb in (1, 2, 4, 8, 16):
                for Kb in kblocks:
                    for Nb in (1, 2, 4, 8, 16):
                        if (K // 32) % Kb:
                            continue
                        sh, sw = max(((h, v) for h in (1, 2, 4) for v in (1, 2, 4)
                                      if Mb % h == 0 and Nb % v == 0 and h * v <= (4 if f32 else 8)),
                                     key=lambda p: (p[0] * p[1], p[1]))
                        cfg = ttnn.MinimalMatmulConfig(M_block_size=Mb, K_block_size=Kb, N_block_size=Nb, subblock_h=sh,
                                                       subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
                        row = {"mode": mode, "K": K, "N": N, "arm": "minimal", "blk": [Mb, Kb, Nb, sh, sw]}
                        try:
                            s, o = timed(dev, lambda: ttnn.experimental.minimal_matmul(
                                x, w, dtype=odt, compute_kernel_config=ckc, config=cfg), a.inner)
                            got = ttnn.to_torch(o).reshape(ref.shape); ttnn.deallocate(o)
                            row.update(us=1e6 * s, speedup=base / s, exact=bool(torch.equal(got, ref)), rel_l2_f64=rel(got))
                            if best is None or s < best[0]:
                                best = (s, row)
                        except Exception as e:  # noqa: BLE001
                            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                        res["rows"].append(row)
            if best:
                b = best[1]
                print(f"   best {b['blk']} {b['us']:.1f} us {b['speedup']:.2f}x exact={b['exact']} relL2 {b['rel_l2_f64']:.3e}",
                      flush=True)
            ttnn.deallocate(x); ttnn.deallocate(w)
            Path(a.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
