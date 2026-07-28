"""p30: is `ttnn.gather` fast enough to replace the per-block `ttnn.scatter`?

The scatter at `rfd3.py:1280` places a [1,4,L,128] pair bias into a dense [1,4,L,Lpad]
-1e4 mask, once per atom block, nine times per step, and 05473c89b showed its cost is the
dense traversal (linear in dense size, flat in K), not the indirection.

`f["attn_indices"]` is STEP-FIXED and shared by all nine blocks -- only `pair_bias`
differs. So the indirection can be inverted once per step:

    ginv[b,h,i,j] = k   where idx[b,h,i,k] == j,  else K (a sentinel column of -1e4)
    dense_bias    = gather(pad(pair_bias, K -> K+32, -1e4), dim=3, index=ginv)

That converts nine scatters into one index build plus nine gathers, and is bit-exact by
construction: a gather is a copy, and the sentinel column holds exactly the -1e4 the
template held. The whole lever therefore reduces to one question this script answers:
**what bandwidth does `ttnn.gather` reach on a [1,4,L,Lpad] index?** If it is the same
~40 GB/s the scatter reaches, the lever is dead and so is Lever 3's precondition.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 scripts/rfd3_port/p30_gather_vs_scatter.py [L]
"""

from __future__ import annotations

import sys
import time

import torch

L = int(sys.argv[1]) if len(sys.argv) > 1 else 3359
H, K = 4, 128
TILE = 32
LPAD = -(-L // TILE) * TILE


def main() -> None:
    import ttnn

    device = ttnn.open_device(device_id=0)
    dt = ttnn.bfloat16

    g = torch.Generator().manual_seed(0)
    idx = torch.stack([
        torch.sort(torch.randperm(L, generator=g)[:K])[0] for _ in range(L)
    ])                                                       # [L,K]
    idx4 = idx.view(1, 1, L, K).expand(1, H, L, K).contiguous().to(torch.int32)

    # host-built inverse index, for the bandwidth question (the shipped path would build
    # it on device once per step; see _inverse_index below for that form)
    ginv_host = torch.full((L, LPAD), K, dtype=torch.int32)
    ginv_host.scatter_(1, idx.long(), torch.arange(K, dtype=torch.int32).expand(L, K))
    ginv4 = ginv_host.view(1, 1, L, LPAD).expand(1, H, L, LPAD).contiguous()

    def up(t, dtype=dt, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, layout=layout, device=device, dtype=dtype)

    def timed(label, fn, reps=6, quiet=False):
        try:
            for _ in range(2):
                out = fn()
            ttnn.synchronize_device(device)
        except Exception as exc:                                        # noqa: BLE001
            print(f"  {label:<50s}  UNSUPPORTED: {type(exc).__name__}: "
                  f"{str(exc).replace(chr(10), ' | ')[:150]}", flush=True)
            return None, None
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            out = fn()
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        samples.sort()
        ms = samples[len(samples) // 2]
        if not quiet:
            print(f"  {label:<50s} {ms:8.3f} ms", flush=True)
        return out, ms

    pair_bias = up(torch.randn(1, H, L, K, generator=g))
    dense_tmpl = up(torch.full((1, H, L, LPAD), -1e4))
    attn_idx = up(idx4, dtype=ttnn.uint32)
    dense_mb = H * L * LPAD * 2 / 1e6
    print(f"L={L} LPAD={LPAD} H={H} K={K}  dense bf16={dense_mb:.1f} MB  "
          f"index u32={H * L * LPAD * 4 / 1e6:.1f} MB")

    print("\n--- the op to beat, and the elementwise yardstick on the same tensor ---")
    ref, ms_scatter = timed("scatter(dense, 3, idx[K], pair_bias)  [SHIPPED]",
                            lambda: ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias))
    ref_host = ttnn.to_torch(ref).float()
    _, ms_add = timed("add(dense, dense)  bf16", lambda: ttnn.add(dense_tmpl, dense_tmpl))
    print(f"  scatter: {(2 * dense_mb) / ms_scatter:6.0f} MB/ms = "
          f"{(2 * dense_mb) / ms_scatter:.0f} GB/s      "
          f"add: {(3 * dense_mb) / ms_add:.0f} GB/s")

    print("\n--- the question: gather bandwidth on a dense [1,H,L,LPAD] index ---")
    pb_pad = ttnn.pad(pair_bias, [(0, 0), (0, 0), (0, 0), (0, TILE)], -1e4)
    print(f"  padded pair bias shape {tuple(pb_pad.shape)} (sentinel column {K} = -1e4)")
    for idt in (ttnn.uint32, ttnn.int32):
        gi = up(ginv4, dtype=idt)
        out, ms = timed(f"gather(pb_pad, 3, ginv:{str(idt).split('.')[-1]})",
                        lambda gi=gi: ttnn.gather(pb_pad, 3, gi))
        if ms is not None:
            d = (ttnn.to_torch(out).float() - ref_host).abs().max().item()
            gbs = (dense_mb + H * L * LPAD * 4 / 1e6) / ms
            print(f"      -> {gbs:.0f} GB/s, maxabs vs scatter = {d:.3e} "
                  f"{'BIT-EXACT' if d == 0.0 else 'NOT bit-exact'}")

    print("\n--- can the inverse index be built ON DEVICE once per step? ---")
    # scatter refuses int32/uint32 rows longer than 256, so build in bf16 (every value is
    # <= K = 128, exactly representable) and typecast.
    k_cols = up(torch.arange(K).view(1, 1, 1, K).expand(1, H, L, K).contiguous().float())
    sent = up(torch.full((1, H, L, LPAD), float(K)))
    gi_bf, ms_b1 = timed("build: scatter(full(K) bf16, 3, idx, arange(K) bf16)",
                         lambda: ttnn.scatter(sent, 3, attn_idx, k_cols))
    if gi_bf is not None:
        gi_u32, ms_b2 = timed("build: typecast(bf16 -> uint32)",
                              lambda: ttnn.typecast(gi_bf, ttnn.uint32))
        if gi_u32 is not None:
            exact = torch.equal(ttnn.to_torch(gi_u32).to(torch.int32), ginv4)
            print(f"      device-built inverse index == host-built: {exact}")
            print(f"      build cost {ms_b1 + ms_b2:.3f} ms ONCE per step")
            out, ms = timed("gather with the device-built index",
                            lambda: ttnn.gather(pb_pad, 3, gi_u32))
            if ms is not None:
                d = (ttnn.to_torch(out).float() - ref_host).abs().max().item()
                print(f"      -> maxabs vs scatter = {d:.3e} "
                      f"{'BIT-EXACT' if d == 0.0 else 'NOT bit-exact'}")
                print(f"\n  VERDICT per step (9 blocks): shipped 9 x {ms_scatter:.3f} = "
                      f"{9 * ms_scatter:.1f} ms   vs   "
                      f"build {ms_b1 + ms_b2:.3f} + 9 x {ms:.3f} = "
                      f"{ms_b1 + ms_b2 + 9 * ms:.1f} ms")

    print("\n--- does a preallocated output help either op? ---")
    gi = up(ginv4, dtype=ttnn.uint32)
    out_buf = up(torch.zeros(1, H, L, LPAD))
    timed("gather(..., out=preallocated)",
          lambda: ttnn.gather(pb_pad, 3, gi, out=out_buf))

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
