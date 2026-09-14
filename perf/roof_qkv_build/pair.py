#!/usr/bin/env python3
"""The qkv->SDPA fold, built: one program doing the projection and the attention, against the two
programs it replaces, interleaved in one process on one card.

Arm A is what ships: `generic_minimal_matmul(x, w) -> q, k, v` then the persistent-mask fused SDPA.
Arm B is `triatt_sdpa.sdpa_fused_qkv(x, w, bias)`, which reads x and the head's weight slices and
makes q, k and v inside the SDPA's compute kernel. A/A is arm A run a second time in the same
round, so the floor is this session's own.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ttnn
import torch

from tt_bio import tenstorrent as TT
from tt_bio import triatt_qkv as TQ
from tt_bio import triatt_sdpa as TS

MB = 1e6
S = int(os.environ.get("ROOF_S", "512"))
H, DH = 4, 32
C = H * DH
REPS = int(os.environ.get("ROOF_REPS", "7"))
QC = int(os.environ.get("ROOF_QC", str(S)))
KC = int(os.environ.get("ROOF_KC", "256" if S >= 256 else str(S)))


def main():
    dev = TT.get_device()
    grid = tuple(TT.COMPUTE_GRID_MAIN)
    print(f"host={os.uname().nodename} grid={grid} arch={dev.arch()} S={S} q={QC} k={KC}",
          flush=True)

    torch.manual_seed(0)
    xt = torch.randn(S, S, C, dtype=torch.float32) * 0.1
    wt = torch.randn(C, 3 * H * DH, dtype=torch.float32) * 0.1
    bt = torch.randn(1, H, S, S, dtype=torch.float32) * 0.1
    mk = lambda t: ttnn.from_torch(  # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w, bias = mk(xt), mk(wt), mk(bt)
    scale = DH ** -0.5
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
        fp32_dest_acc_en=False, packer_l1_acc=False)
    cfg = TT._qkv_mm_config(x, w)

    def run_today():
        q, k, v = TQ.qkv_heads(x, w, ckc, H, DH, ttnn.bfloat16, cfg)
        o = TS.sdpa(q, k, v, bias, scale, QC, KC)
        assert o is not None, "arm A's SDPA declined"
        for t in (q, k, v):
            ttnn.deallocate(t)
        return o

    def run_fused():
        o = TS.sdpa_fused_qkv(x, w, bias, scale, H, DH, QC, KC, force=True)
        assert o is not None, f"the fold declined: {dict(TS.FUSE_REJECTS)}"
        return o

    ref = run_today()
    got = run_fused()
    a = ttnn.to_torch(ref).float()
    b = ttnn.to_torch(got).float()
    exact = bool(torch.equal(a, b))
    den = a.abs().max().item()
    max_abs = (a - b).abs().max().item()
    rel_rms = ((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item()
    print(f"parity vs today: torch.equal={exact}  max_abs={max_abs:.3e} "
          f"(ref max |.| {den:.3e})  rel_rms={rel_rms:.3e}", flush=True)
    ttnn.deallocate(ref)
    ttnn.deallocate(got)

    arms = [("today", run_today), ("fused", run_fused), ("AA:today", run_today)]
    for _n, f in arms:
        for _ in range(2):
            ttnn.deallocate(f())
    ttnn.synchronize_device(dev)

    samples = {n: [] for n, _ in arms}
    for r in range(REPS):
        for n, f in arms:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = f()
            ttnn.synchronize_device(dev)
            samples[n].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
        print("  rep %d/%d: " % (r + 1, REPS) + "  ".join(
            f"{n}={samples[n][-1]:.3f}" for n, _ in arms), flush=True)

    elem = 2
    cores = (grid[0] * grid[1] // H) * H
    mask_b = cores * S * S * elem
    b_today = (S * S * C * elem + 3 * S * H * S * DH * elem          # projection
               + 3 * S * H * S * DH * elem + mask_b + S * H * S * DH * elem)   # sdpa
    b_fused = H * S * S * C * elem + mask_b + S * H * S * DH * elem
    med = {n: statistics.median(v) for n, v in samples.items()}
    aa = abs(med["AA:today"] - med["today"])
    ratio = med["today"] / med["fused"]
    print()
    for n, mb in (("today", b_today / MB), ("fused", b_fused / MB)):
        print(f"  {n:6s} {med[n]:7.3f} ms   {mb:7.1f} MB   {mb / med[n]:6.1f} MB/ms")
    print(f"  A/A floor {aa:.3f} ms ({aa / med['today'] * 100:.2f} %)")
    print(f"  RATIO {ratio:.4f}x   delta {med['today'] - med['fused']:.3f} ms "
          f"= {(med['today'] - med['fused']) / max(aa, 1e-9):.0f}x the floor")

    out = dict(host=os.uname().nodename, grid=list(grid), arch=str(dev.arch()), S=S, q_chunk=QC,
               k_chunk=KC, reps=REPS, median_ms=med, samples=samples, aa_ms=aa, ratio=ratio,
               bytes_mb=dict(today=b_today / MB, fused=b_fused / MB),
               parity=dict(torch_equal=exact, max_abs=max_abs, rel_rms=rel_rms, ref_max=den))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pair_{S}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
