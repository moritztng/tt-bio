"""p30: price every way of getting `softmax(scores*scale + sparse_pair_bias)` at L=3359.

p29 named one op as 19.2% of RFD3's device time: `ttnn.scatter` of the [1,4,L,128] pair
bias into the dense [1,4,L,Lpad] -1e4 mask, at 41 GB/s where `add`/`softmax`/`multiply`
on the identical tensor reach 205-303. Commit 05473c89b already showed the cost is the
DENSE TRAVERSAL, not the scattering (linear in dense size, flat in K) and that no knob --
index layout, index dtype, sub_core_grids -- moves it, so this script does not re-price
knobs. It prices reformulations:

  * the shipped chain, as the bit-exactness reference and the time to beat;
  * `scale_mask_softmax`, which fuses scale+mask+softmax into one pass over the big tensor;
  * a per-step scatter + per-block `ttnn.gather` -- the indirection cost paid once per step
    instead of once per block, since `attn_indices` is step-fixed and shared by all 9 blocks;
  * `tosa_scatter`, a different kernel for the same semantics;
  * the no-scatter floor (a cached bias), which prices the recycle-caching idea's ceiling.

Every variant reports maxabs against the shipped chain's OUTPUT, because bit-exact is the
gate for this lineage, not a PCC threshold.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 scripts/rfd3_port/p30_bias_chain_variants.py [L]
"""

from __future__ import annotations

import sys
import time

import torch

L = int(sys.argv[1]) if len(sys.argv) > 1 else 3359
H, K, HEAD_DIM = 4, 128, 48
SCALE = HEAD_DIM**-0.5
TILE = 32
LPAD = -(-L // TILE) * TILE


def main() -> None:
    import ttnn

    device = ttnn.open_device(device_id=0)
    dt = ttnn.bfloat16
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    g = torch.Generator().manual_seed(0)
    idx = torch.stack([
        torch.sort(torch.randperm(L, generator=g)[:K])[0] for _ in range(L)
    ])                                             # [L,K] sorted neighbours, as in the model
    idx4 = idx.view(1, 1, L, K).expand(1, H, L, K).contiguous().to(torch.int32)

    def up(t, dtype=dt, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, layout=layout, device=device, dtype=dtype)

    def timed(label, fn, reps=6):
        """Median of `reps` sync-bracketed calls; None if the op refuses the shapes."""
        try:
            for _ in range(2):
                out = fn()
            ttnn.synchronize_device(device)
        except Exception as exc:                                        # noqa: BLE001
            msg = str(exc).split("\n")[0][:110]
            print(f"  {label:<52s}  UNSUPPORTED: {type(exc).__name__}: {msg}", flush=True)
            return None, None
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            out = fn()
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        samples.sort()
        return out, samples[len(samples) // 2]

    def report(label, fn, ref=None, reps=6):
        out, ms = timed(label, fn, reps)
        if ms is None:
            return None
        tag = ""
        if ref is not None and out is not None:
            a = ttnn.to_torch(out).float()
            d = (a - ref).abs().max().item()
            tag = f"  maxabs={d:.3e} {'BIT-EXACT' if d == 0.0 else 'NOT bit-exact'}"
        print(f"  {label:<52s} {ms:8.3f} ms{tag}", flush=True)
        return ms

    # ---- the tensors a real atom block hands the chain -------------------------------
    scores_bf = up(torch.randn(1, H, L, LPAD, generator=g) * 3.0)      # qq @ kk^T, bf16
    pair_bias = up(torch.randn(1, H, L, K, generator=g))               # linear(p), bf16
    dense_tmpl = up(torch.full((1, H, L, LPAD), -1e4))                 # _mask_template
    attn_idx = up(idx4, dtype=ttnn.uint32)

    mb_dense_bf = H * L * LPAD * 2 / 1e6
    print(f"L={L} LPAD={LPAD} H={H} K={K}  dense bf16={mb_dense_bf:.0f} MB, "
          f"fp32={mb_dense_bf * 2:.0f} MB  scale={SCALE:.6f}")

    # ---- V0: the shipped chain -------------------------------------------------------
    def shipped():
        bias = ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)
        s = ttnn.typecast(scores_bf, ttnn.float32)
        s = ttnn.multiply(s, SCALE)
        s = ttnn.add(s, ttnn.typecast(bias, ttnn.float32))
        s = ttnn.softmax(s, dim=-1)
        return ttnn.typecast(s, dt)

    print("\n--- V0 the shipped chain (reference for every maxabs below) ---")
    _, base = timed("scatter+2xtypecast+mul+add+softmax+typecast", shipped)
    REF = ttnn.to_torch(shipped()).float()
    print(f"  {'V0 SHIPPED':<52s} {base:8.3f} ms   (x9 blocks = {base * 9:.1f} ms/step)")

    # component split, so a variant's saving can be attributed
    print("\n--- V0 components ---")
    bias_ref = ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)
    ttnn.synchronize_device(device)
    bias_ref_host = ttnn.to_torch(bias_ref).float()
    for lbl, fn in (
        ("scatter -> dense bias bf16", lambda: ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)),
        ("typecast(scores) -> fp32", lambda: ttnn.typecast(scores_bf, ttnn.float32)),
        ("typecast(bias) -> fp32", lambda: ttnn.typecast(bias_ref, ttnn.float32)),
        ("add(fp32, fp32)", lambda s=ttnn.typecast(scores_bf, ttnn.float32),
         b=ttnn.typecast(bias_ref, ttnn.float32): ttnn.add(s, b)),
        ("softmax(fp32)", lambda s=ttnn.typecast(scores_bf, ttnn.float32): ttnn.softmax(s, dim=-1)),
        ("add(bf16, bf16)  [bandwidth yardstick]", lambda: ttnn.add(scores_bf, dense_tmpl)),
    ):
        report(lbl, fn)

    # ---- V1/V2/V3: fuse scale+mask+softmax ------------------------------------------
    print("\n--- V1-V3 scale_mask_softmax: one pass replaces mul+add+softmax+typecasts ---")

    def v1():
        bias = ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)
        return ttnn.scale_mask_softmax(scores_bf, SCALE, bias)

    def v2():
        bias = ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)
        return ttnn.scale_mask_softmax(
            scores_bf, SCALE, bias, compute_kernel_config=ckc, numeric_stable=True
        )

    def v3():
        bias = ttnn.scatter(dense_tmpl, 3, attn_idx, pair_bias)
        s = ttnn.typecast(scores_bf, ttnn.float32)
        b = ttnn.typecast(bias, ttnn.float32)
        return ttnn.typecast(
            ttnn.scale_mask_softmax(s, SCALE, b, compute_kernel_config=ckc,
                                    numeric_stable=True), dt)

    report("V1 scatter + scale_mask_softmax (bf16, default)", v1, REF)
    report("V2 scatter + scale_mask_softmax (bf16, fp32acc)", v2, REF)
    report("V3 scatter + scale_mask_softmax (fp32 chain)", v3, REF)

    # ---- V4: pay the indirection once per step, gather 9 times -----------------------
    # attn_indices is step-fixed and shared by all 9 atom blocks, so the inverse map
    # g[b,h,i,j] = k where idx[i,k]==j, else a sentinel column holding -1e4, can be built
    # ONCE per step with one scatter and then read by a plain gather per block.
    print("\n--- V4 per-step inverse index + per-block gather (bit-exact by construction) ---")
    SENT = K                                    # sentinel column of the padded pair bias
    k_cols = up(torch.arange(K).view(1, 1, 1, K).expand(1, H, L, K).contiguous().to(torch.int32),
                dtype=ttnn.uint32)
    sent_tmpl = up(torch.full((1, H, L, LPAD), float(SENT)).to(torch.int32), dtype=ttnn.uint32)
    ginv, ms_ginv = timed("  (once/step) build inverse index by scatter",
                          lambda: ttnn.scatter(sent_tmpl, 3, attn_idx, k_cols))
    if ginv is not None:
        print(f"  {'V4 index build (amortised over 9 blocks)':<52s} {ms_ginv:8.3f} ms")
        pb_pad = ttnn.pad(pair_bias, [(0, 0), (0, 0), (0, 0), (0, TILE)], -1e4)

        def v4_bias():
            return ttnn.gather(pb_pad, 3, ginv)

        out, ms = timed("V4 gather(pair_bias_padded, dim=3, inverse_index)", v4_bias)
        if ms is not None:
            d = (ttnn.to_torch(out).float() - bias_ref_host).abs().max().item()
            print(f"  {'V4 gather -> dense bias':<52s} {ms:8.3f} ms  "
                  f"maxabs(vs scatter bias)={d:.3e} "
                  f"{'BIT-EXACT' if d == 0.0 else 'NOT bit-exact'}")

    # ---- V5: a different scatter kernel ---------------------------------------------
    print("\n--- V5 tosa_scatter (rank-3 input, rank-2 index) ---")
    d3 = ttnn.reshape(dense_tmpl, (1, H * L, LPAD))
    s3 = ttnn.reshape(pair_bias, (1, H * L, K))
    i2 = up(idx4.reshape(H * L, K), dtype=ttnn.int32)
    report("V5 tosa_scatter", lambda: ttnn.tosa_scatter(d3, i2, s3))

    # ---- V6: the no-scatter floor ---------------------------------------------------
    print("\n--- V6 floor: the same chain with the scatter removed (cached bias) ---")

    def v6():
        s = ttnn.typecast(scores_bf, ttnn.float32)
        s = ttnn.multiply(s, SCALE)
        s = ttnn.add(s, ttnn.typecast(bias_ref, ttnn.float32))
        s = ttnn.softmax(s, dim=-1)
        return ttnn.typecast(s, dt)

    report("V6 cached bias + shipped chain", v6, REF)
    report("V6b cached bias + scale_mask_softmax(fp32)",
           lambda: ttnn.typecast(ttnn.scale_mask_softmax(
               ttnn.typecast(scores_bf, ttnn.float32), SCALE,
               ttnn.typecast(bias_ref, ttnn.float32),
               compute_kernel_config=ckc, numeric_stable=True), dt), REF)
    report("V6c cached fp32 bias + scale_mask_softmax(fp32)",
           lambda b=ttnn.typecast(bias_ref, ttnn.float32): ttnn.typecast(
               ttnn.scale_mask_softmax(ttnn.typecast(scores_bf, ttnn.float32), SCALE, b,
                                       compute_kernel_config=ckc, numeric_stable=True), dt), REF)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
