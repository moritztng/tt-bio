#!/usr/bin/env python3
"""The qkv->SDPA fold, built: one program doing the projection and the attention, against the two
programs it replaces, interleaved in one process on one card.

Arm `today` is what SHIPS, and the two halves do not share a compute kernel config: the projection
runs at the trunk's `af2.compute_kernel_config()` (HiFi4, approx off, fp32 accumulator, packer L1
acc) and the fused SDPA runs at the op's own default (HiFi2, approx on, no fp32 accumulator). A
fold puts both under ONE config, so there are two fused arms -- one at each of those -- and the
question the error columns answer is which of them the projection can afford to be under.

Every arm is scored against an fp64 CPU evaluation of the SAME bf16 operands, so only the kernel's
own error is left. The reference is evaluated on `ROOF_REF_B` batch rows, not all B: the full
[B, H, S, S] score tensor is 4.3 GB in fp64 at 512 aa.
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
REF_B = int(os.environ.get("ROOF_REF_B", "48"))
TAG = os.environ.get("ROOF_TAG", "s1")

HI = (ttnn.MathFidelity.HiFi4, False, True, False)     # the trunk's, minus packer_l1_acc
LO = (ttnn.MathFidelity.HiFi2, True, False, False)     # the fused SDPA's own default


def fp64_reference(xt, wt, bt, scale, nb):
    """Triangle attention in fp64 on the host, from the bf16 operands the device was given."""
    x = xt[:nb].double()                                   # [nb, S, C]
    w = wt.double()
    qkv = x @ w                                            # [nb, S, 3*H*DH]
    q, k, v = (qkv[..., i * H * DH:(i + 1) * H * DH].reshape(nb, S, H, DH).permute(0, 2, 1, 3)
               for i in range(3))
    sc = (q @ k.transpose(-1, -2)) * scale + bt.double()   # [nb, H, S, S]
    return (torch.softmax(sc, dim=-1) @ v)


def main():
    dev = TT.get_device()
    grid = tuple(TT.COMPUTE_GRID_MAIN)
    print(f"host={os.uname().nodename} grid={grid} arch={dev.arch()} S={S} q={QC} k={KC} "
          f"tag={TAG}", flush=True)

    torch.manual_seed(0)
    mk = lambda t: ttnn.from_torch(  # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = mk(torch.randn(S, S, C) * 0.1)
    w = mk(torch.randn(C, 3 * H * DH) * 0.1)
    bias = mk(torch.randn(1, H, S, S) * 0.1)
    # Read the bf16 operands BACK, so the reference sees exactly what the device holds.
    xt, wt, bt = (ttnn.to_torch(t).float() for t in (x, w, bias))
    scale = DH ** -0.5
    trunk_ckc = TT.af2.compute_kernel_config() if hasattr(TT, "af2") else None
    if trunk_ckc is None:
        from tt_bio import af2
        trunk_ckc = af2.compute_kernel_config()
    cfg = TT._qkv_mm_config(x, w)

    def run_today():
        """What ships: the projection at the trunk's ckc, the SDPA at the op's default."""
        q, k, v = TQ.qkv_heads(x, w, trunk_ckc, H, DH, ttnn.bfloat16, cfg)
        o = TS.sdpa(q, k, v, bias, scale, QC, KC)
        assert o is not None, "arm `today`'s SDPA declined"
        for t in (q, k, v):
            ttnn.deallocate(t)
        return o

    def fused(ckc):
        def f():
            o = TS.sdpa_fused_qkv(x, w, bias, scale, H, DH, QC, KC, ckc_default=ckc, force=True)
            assert o is not None, f"the fold declined: {dict(TS.FUSE_REJECTS)}"
            return o
        return f

    # The two halves on their own, so the fold's win can be split into the program it deletes and
    # the program it grows. `q0/k0/v0` are made once and kept, so `sdpa_only` reads what the
    # shipped projection wrote.
    q0, k0, v0 = TQ.qkv_heads(x, w, trunk_ckc, H, DH, ttnn.bfloat16, cfg)
    ttnn.synchronize_device(dev)

    def run_mm_only():
        q, k, v = TQ.qkv_heads(x, w, trunk_ckc, H, DH, ttnn.bfloat16, cfg)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return q

    def run_sdpa_only():
        o = TS.sdpa(q0, k0, v0, bias, scale, QC, KC)
        assert o is not None
        return o

    arms = [("today", run_today), ("fused_lo", fused(LO)), ("fused_hi", fused(HI)),
            ("mm_only", run_mm_only), ("sdpa_only", run_sdpa_only), ("AA:today", run_today)]

    # --- parity, before timing --------------------------------------------------------------
    ref = fp64_reference(xt, wt, bt, scale, REF_B)
    den = ref.pow(2).mean().sqrt()
    err, got = {}, {}
    for n, f in arms:
        o = f()
        got[n] = ttnn.to_torch(o)
        if n != "mm_only":
            t = got[n].double()[:REF_B]
            err[n] = ((t - ref).pow(2).mean().sqrt() / den).item()
        ttnn.deallocate(o)
    print(f"\nrel_rms vs an fp64 evaluation of the same bf16 operands, {REF_B} batch rows:")
    for n, _ in arms:
        if n == "mm_only":
            continue
        same = "  torch.equal(today)" if torch.equal(got[n], got["today"]) else ""
        print(f"  {n:9s} {err[n]:.6e}   {err[n] / err['today']:6.3f}x today{same}")

    # --- timing ------------------------------------------------------------------------------
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
    b_today = (S * S * C * elem + 3 * S * H * S * DH * elem
               + 3 * S * H * S * DH * elem + mask_b + S * H * S * DH * elem)
    b_fused = H * S * S * C * elem + mask_b + S * H * S * DH * elem
    med = {n: statistics.median(v) for n, v in samples.items()}
    aa = abs(med["AA:today"] - med["today"])
    print()
    for n, mb in (("today", b_today / MB), ("fused_lo", b_fused / MB),
                  ("fused_hi", b_fused / MB), ("mm_only", 0.0), ("sdpa_only", 0.0)):
        print(f"  {n:9s} {med[n]:7.3f} ms   {mb:7.1f} MB   {mb / med[n]:6.1f} MB/ms   "
              f"{med['today'] / med[n]:.4f}x")
    print(f"  A/A floor {aa:.3f} ms ({aa / med['today'] * 100:.2f} %)")

    out = dict(host=os.uname().nodename, grid=list(grid), arch=str(dev.arch()), S=S, q_chunk=QC,
               k_chunk=KC, reps=REPS, tag=TAG, median_ms=med, samples=samples, aa_ms=aa,
               ratio={n: med["today"] / med[n] for n, _ in arms},
               bytes_mb=dict(today=b_today / MB, fused=b_fused / MB),
               rel_rms_vs_fp64=err, ref_batch_rows=REF_B,
               equal_to_today={n: bool(torch.equal(got[n], got["today"]))
                               for n, _ in arms if n != "mm_only"})
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pair_{S}_{TAG}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
