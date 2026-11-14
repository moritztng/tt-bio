#!/usr/bin/env python3
"""Dual-NOC output drain (`mm_dualnoc.in_proj`) vs stock `minimal_matmul` at BindCraft 2's pair shapes.

The WH synced Evoformer census on 887d3755b (.107 out/whevo2_887d3755b/devprof.json) has the K=128 pair
projections at 94-99 GB/s, ~0.66 s of a 6.42 s synced round:

    taped     [1,224,224,128] x [128,512]   192 calls 0.126 s
    taped     -> [.., 128]                   768 calls 0.208 s
    backward  -> [50176, 128]                480 calls 0.163 s
    backward  [1,224,224,128] x [128,512]    96 calls 0.063 s

Each writes 1-4x what it reads, so the output drain is the suspect. On BH the dual-NOC drain took
a quarter off the write (perf/esmbeat/s_d_dualnoc.json). This asks the same on Wormhole before any
tape entry is built: per shape, stock ms vs dualnoc ms, torch.equal between them, rel L2 vs float64.
"""
import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt, mm_dualnoc as DN  # noqa: E402
from af2_mm_sweep import timed, rel_l2  # noqa: E402

L = 224
SITES = [(128, 512), (128, 128), (512, 128), (384, 128), (128, 384)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=10)
    a = ap.parse_args()
    dev = tt.get_device()
    ckc_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
    ckc = ckc_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                  fp32_dest_acc_en=True, packer_l1_acc=True)
    DN.set_enabled(True)
    res = {"arch": str(dev.arch()), "grid": list(tt.COMPUTE_GRID_MAIN),
           "loadavg": open("/proc/loadavg").read().split()[:3], "inner": a.inner, "sites": []}
    torch.manual_seed(0)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for k, n in SITES:
        xt = torch.randn(1, L, L, k, dtype=torch.bfloat16)
        wt = (torch.randn(k, n, dtype=torch.bfloat16) / k ** 0.5).to(torch.bfloat16)
        ref64 = (xt.double().reshape(-1, k) @ wt.double()).reshape(1, L, L, n)
        x, w = up(xt), up(wt)
        stock = lambda: ttnn.experimental.minimal_matmul(  # noqa: E731
            x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        dual = lambda: DN.in_proj(x, w, ckc, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
        site = {"k": k, "n": n, "mb_moved": (L * L * (k + n) * 2 + k * n * 2) / 2**20}
        s_s, s_ts, so = timed(dev, stock, a.inner)
        st = ttnn.to_torch(so); ttnn.deallocate(so)
        site.update(stock_ms=1e3 * s_s, stock_reps_ms=[1e3 * t for t in s_ts], stock_rel_l2=rel_l2(st, ref64))
        probe = dual()
        if probe is None:
            site["dual"] = f"declined {DN.REJECTS}"
        else:
            ttnn.deallocate(probe)
            d_s, d_ts, do = timed(dev, dual, a.inner)
            dt = ttnn.to_torch(do); ttnn.deallocate(do)
            site.update(dual_ms=1e3 * d_s, dual_reps_ms=[1e3 * t for t in d_ts], dual_rel_l2=rel_l2(dt, ref64),
                        exact=bool(torch.equal(st, dt)), saving_ms=1e3 * (s_s - d_s))
        site["stock_gbs"] = site["mb_moved"] * 2**20 / (s_s * 1e9)
        print(json.dumps(site), flush=True)
        res["sites"].append(site)
        ttnn.deallocate(x); ttnn.deallocate(w)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
