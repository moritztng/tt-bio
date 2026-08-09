#!/usr/bin/env python3
"""W9: is the tri-attention SDPA off-roof, or is it AT the read roof moving 4.3x the bytes?

W6 left two named mechanisms. This probe attacks the second: the broadcast pair bias
[1,8,N,N] costs time proportional to the batch COUNT, not to the bias size.

Source evidence (ttnn 0.68.0 sdpa/device/kernels/dataflow/reader_interleaved.cpp:301-518):
the reader honours `broadcast_provided_mask_batch` only when computing the mask ADDRESS
(mask_batch_offset stays 0). It still issues a full Sq_chunk_t x Sk_chunk_t tile read of
the mask for every (batch, head, q_chunk, k_chunk) work unit, so the same N*N bias slice is
re-read from DRAM once per batch element per head:

    mask bytes = b * h * N * N * 2      (independent of chunk size)
    qkv  bytes = 3 * b * h * N * d * 2
    out  bytes =     b * h * N * d * 2

HYPOTHESIS: SDPA is not "at neither roof". It is at the DRAM read roof, and 77% of its
reads are redundant bias re-reads.

DESIGNED EXPERIMENT (this is the falsifiable one -- W6's sweep moved b and head_dim
together, so it could not separate them): hold N, h, d fixed and sweep b ALONE, with and
without the bias. Fit t(b) for each. The two slopes must differ by exactly the per-batch
bias bytes over the measured read roof:

    d(t)/d(b) [bias] - d(t)/d(b) [no bias]  ==  h*N*N*2 / BW_read
    at N=320, h=8: 1.638 MB per batch element = 4.3 us at 380 GB/s.

If the measured slope difference is ~0, the bias is NOT re-read per batch and the whole
mechanism is dead.
"""
import argparse
import json
import time

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="perf/sdpa_kernel/mask_traffic_qb2c1.json")
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--amort", type=int, default=4)
args = ap.parse_args()

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def timed(fn, iters=args.iters, amort=args.amort):
    """Amortised: `amort` issues inside one synchronize..synchronize region (W4's lesson)."""
    for _ in range(2):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(amort)]
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3 / amort)
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2]


res = {"card": "qb2-card1", "ttnn": "0.68.0", "roofs": {}, "sdpa": [], "batch_sweep": []}

# ---------------------------------------------------------------- roofs, this card
print("== roofs on this card ==", flush=True)
for mb in (26, 52, 105):
    rows = mb * 1_000_000 // (3200 * 2) // 32 * 32
    t = ttnn.from_torch(torch.randn(1, 1, rows, 3200), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
    nb = rows * 3200 * 2
    ms = timed(lambda: ttnn.clone(t, memory_config=DRAM), amort=2)
    rw = 2 * nb / ms * 1e-6
    print(f"clone DRAM->DRAM {nb/1e6:6.1f} MB: {ms:7.4f} ms  {rw:7.1f} GB/s read+write")
    res["roofs"][f"dram_copy_rw_gbs_{mb}mb"] = rw
    if nb < 60e6:
        ms_l1 = timed(lambda: ttnn.clone(t, memory_config=L1), amort=2)
        rd = nb / ms_l1 * 1e-6
        print(f"clone DRAM->L1   {nb/1e6:6.1f} MB: {ms_l1:7.4f} ms  {rd:7.1f} GB/s DRAM read only")
        res["roofs"][f"dram_read_into_l1_gbs_{mb}mb"] = rd
    ttnn.deallocate(t)


# ---------------------------------------------------------------- sdpa
def sdpa_case(b, h, n, d, bias_dtype, qc, kc, want_out=False, seed=7):
    q, k, v = (ttnn.from_torch(torch.randn(b, h, n, d) * 0.3, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
               for _ in range(3))
    bias = None
    if bias_dtype is not None:
        torch.manual_seed(seed)
        bias = ttnn.from_torch(torch.randn(1, h, n, n) * 0.3, dtype=bias_dtype,
                               layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=DEV.compute_with_storage_grid_size(),
        q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)

    def call():
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=d ** -0.5,
            program_config=prog, compute_kernel_config=CKC, memory_config=DRAM)

    ms = timed(call)
    out = None
    if want_out:
        o = call()
        out = ttnn.to_torch(o)
        ttnn.deallocate(o)
    for t in (q, k, v):
        ttnn.deallocate(t)
    if bias is not None:
        ttnn.deallocate(bias)
    return ms, out


B, H, D, N = 320, 8, 32, 320
print("\n== SDPA at the real 298-aa shape: q[320,8,320,32], bias[1,8,320,320] ==", flush=True)
print(f"{'bias':>8} {'qc':>4} {'kc':>4} {'ms':>9} {'qkv MB':>8} {'mask MB':>9} {'GB/s(r)':>9}")
outs = {}
for bias_name, bdt in (("bf16", ttnn.bfloat16), ("bfp8_b", ttnn.bfloat8_b), ("none", None)):
    for (qc, kc) in ((320, 320), (64, 64)):
        try:
            want = (qc, kc) == (320, 320)
            ms, o = sdpa_case(B, H, N, D, bdt, qc, kc, want_out=want)
        except Exception as e:
            print(f"{bias_name:>8} {qc:>4} {kc:>4}   FAILED: {str(e).splitlines()[0][:70]}")
            continue
        qkv = 3 * B * H * N * D * 2
        mask = B * H * N * N * (1 if bdt == ttnn.bfloat8_b else 2) if bdt is not None else 0
        print(f"{bias_name:>8} {qc:>4} {kc:>4} {ms:9.4f} {qkv/1e6:8.1f} "
              f"{mask/1e6:9.1f} {(qkv+mask)/ms*1e-6:9.1f}", flush=True)
        res["sdpa"].append(dict(bias=bias_name, qc=qc, kc=kc, ms=ms, qkv_mb=qkv / 1e6,
                                mask_mb=mask / 1e6, read_gbs=(qkv + mask) / ms * 1e-6))
        if want and o is not None:
            outs[bias_name] = o

if "bf16" in outs and "bfp8_b" in outs:
    a, c = outs["bf16"].float(), outs["bfp8_b"].float()
    rmsd = (a - c).pow(2).mean().sqrt().item()
    res["bfp8_bias_vs_bf16"] = dict(rmsd=rmsd, std=a.std().item(),
                                    rmsd_over_std=rmsd / a.std().item())
    print(f"\nbfp8_b bias vs bf16 bias on the SDPA output: rmsd/std = "
          f"{rmsd/a.std().item():.5f}")

# ------------------------------------------ the designed experiment: sweep b alone
print("\n== batch sweep at fixed N=320,h=8,d=32: does the bias cost track b? ==", flush=True)
print(f"{'b':>5} {'bias':>6} {'ms':>9} {'us/batch':>9}")
for b in (40, 80, 160, 320):
    row = {}
    for bias_name, bdt in (("bf16", ttnn.bfloat16), ("none", None)):
        ms, _ = sdpa_case(b, H, N, D, bdt, N, N)
        row[bias_name] = ms
        print(f"{b:>5} {bias_name:>6} {ms:9.4f} {ms*1e3/b:9.3f}", flush=True)
    res["batch_sweep"].append(dict(b=b, ms_bias=row["bf16"], ms_none=row["none"],
                                   delta_ms=row["bf16"] - row["none"]))

bs = res["batch_sweep"]
if len(bs) >= 2:
    db = bs[-1]["b"] - bs[0]["b"]
    slope_bias = (bs[-1]["ms_bias"] - bs[0]["ms_bias"]) / db
    slope_none = (bs[-1]["ms_none"] - bs[0]["ms_none"]) / db
    per_batch_mask_bytes = H * N * N * 2
    res["slope_us_per_batch_bias"] = slope_bias * 1e3
    res["slope_us_per_batch_none"] = slope_none * 1e3
    res["implied_mask_gbs"] = per_batch_mask_bytes / (slope_bias - slope_none) * 1e-6
    print(f"\nslope with bias {slope_bias*1e3:.3f} us/batch, without {slope_none*1e3:.3f} "
          f"us/batch, difference {(slope_bias-slope_none)*1e3:.3f} us/batch")
    print(f"per-batch mask bytes {per_batch_mask_bytes/1e6:.3f} MB -> implied bandwidth "
          f"{res['implied_mask_gbs']:.1f} GB/s")

json.dump(res, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
ttnn.close_device(DEV)
