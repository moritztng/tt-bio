#!/usr/bin/env python3
"""Program-config sweep at Protenix-v2's dit_shoist conditioning linears, Wormhole.

The c4 census (WH normal, cdk2x2 512, .107, AICLK 1000) puts protenix.py `_dit_shoist`'s three
`ttnn.linear` calls at 2.31 ms each, 600 calls, 1.39 s a fold: [512,384] x [384,36864], HiFi4,
fp32_dest_acc, core_grid=CORE_GRID_MAIN. Normal's DiT is fp32, so weights and outputs are fp32:
57 + 75 MB a call, 57 GB/s. Fast (bf16) reads 0.97 ms. Every arm here keeps the site's dtype and
fidelity, so a win is a pure schedule change; `exact` is torch.equal against the site's own call.

Arms per dtype: `site` (ttnn.linear exactly as the site calls it, no activation), then
ttnn.experimental.minimal_matmul over a block grid. Warm 2, median of 5 reps of `--inner` calls.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch
import ttnn
from tt_bio import tenstorrent as tt

WARM, REPS = 2, 5
M_ROWS, K_DIM, N_DIM = 512, 384, 36864


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
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--cands", default="", help="M:K:N,... instead of the full grid")
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"arch": str(dev.arch()), "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3],
           "shape": [M_ROWS, K_DIM, N_DIM], "inner": a.inner, "rows": []}
    torch.manual_seed(0)
    xt = torch.randn(1, M_ROWS, K_DIM)
    wt = torch.randn(K_DIM, N_DIM) / K_DIM ** 0.5
    ref64 = (xt.double() @ wt.double())

    def rel(got):
        d = got.double().reshape(ref64.shape) - ref64
        return float(d.norm() / ref64.norm()), float(d.abs().max())
    for dname in a.dtypes.split(","):
        dt = getattr(ttnn, dname)
        up = lambda t: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
        x, w = up(xt), up(wt)
        site = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN)  # noqa: E731
        base, o = timed(dev, site, a.inner)
        ref = ttnn.to_torch(o); ttnn.deallocate(o)
        rl, ma = rel(ref)
        print(f"== {dname} site {1e3 * base:.4f} ms relL2_f64 {rl:.3e} max_abs_f64 {ma:.3e}", flush=True)
        res["rows"].append({"dtype": dname, "arm": "site", "ms": 1e3 * base, "rel_l2_f64": rl, "max_abs_f64": ma})
        cands = [(M, K, N, sh, sw) for M in (1, 2, 4, 8, 16) for K in (1, 2, 4, 6, 12) for N in (1, 2, 4, 8, 16)
                 for sh, sw in [max(((h, v) for h in (1, 2, 4) for v in (1, 2, 4)
                                     if M % h == 0 and N % v == 0 and h * v <= 4), key=lambda p: p[0] * p[1])]]
        if a.cands:
            want = {tuple(int(v) for v in c.split(":")) for c in a.cands.split(",")}
            cands = [c for c in cands if c[:3] in want]
        for M, K, N, sh, sw in cands:
            cfg = ttnn.MinimalMatmulConfig(M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh,
                                           subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
            row = {"dtype": dname, "arm": "minimal", "M": M, "K": K, "N": N, "sh": sh, "sw": sw}
            try:
                s, o = timed(dev, lambda: ttnn.experimental.minimal_matmul(
                    x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dt, compute_kernel_config=ckc, config=cfg),
                    a.inner)
                got = ttnn.to_torch(o).reshape(ref.shape); ttnn.deallocate(o)
                row.update(ms=1e3 * s, speedup=base / s, exact=bool(torch.equal(got, ref)),
                           max_abs=float((got.float() - ref.float()).abs().max()))
                row["rel_l2_f64"], row["max_abs_f64"] = rel(got)
                print(f"   M={M:2d} K={K:2d} N={N:2d} {row['ms']:.4f} ms {row['speedup']:.3f}x exact={row['exact']} "
                      f"relL2_f64 {row['rel_l2_f64']:.3e} max_abs_f64 {row['max_abs_f64']:.3e}", flush=True)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                print(f"   M={M:2d} K={K:2d} N={N:2d} ERROR {row['error'][:90]}", flush=True)
            res["rows"].append(row)
            Path(a.out).write_text(json.dumps(res, indent=1))
        ttnn.deallocate(x); ttnn.deallocate(w)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
