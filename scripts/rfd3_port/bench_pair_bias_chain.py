"""Split the two expensive shipped atom-block ops into their parts.

`bench_atom_attn_ops.py` measures "pair_bias linear+permute" as one 6.4 ms
lump at L=3359 and the fp32 score chain as another 5.8 ms. This separates
them and prices bit-exact alternatives:

* the pair-bias projection: is the cost the (1,L,128,16)@(16,4) linear
  (last dims 16 and 4, both sub-tile) or the (0,3,1,2) permute that follows?
* the fp32 score chain: can `typecast(bf16)->multiply(fp32)` collapse into one
  `multiply(bf16 in, fp32 out)` pass, which reads half the bytes and is
  bit-identical (bf16->fp32 is lossless, the product is computed in fp32 either
  way)?

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 -m scripts.rfd3_port.bench_pair_bias_chain [L]
"""

from __future__ import annotations

import sys
import time

import torch


def main() -> None:
    import ttnn

    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3359
    H, HD, K, C_PAIR = 4, 32, 128, 16
    dt = ttnn.bfloat16

    device = ttnn.open_device(device_id=0)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
    )

    def up(t, dtype=dt):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    p_sparse = up(torch.randn(1, L, K, C_PAIR))
    b_w = up(torch.randn(C_PAIR, H) * 0.05)
    pb_tok = up(torch.randn(1, L, K, H))
    scores = up(torch.randn(1, H, L, L))
    bias = up(torch.randn(1, H, L, L))
    scores_f = up(torch.randn(1, H, L, L), dtype=ttnn.float32)
    bias_f = up(torch.randn(1, H, L, L), dtype=ttnn.float32)

    def timed(label, fn, reps=3):
        out = fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000 / reps
        print(f"  {label:<50s} {ms:8.3f} ms", flush=True)
        return out, ms

    print(f"L={L}  pair-bias projection")
    timed("linear (1,L,128,16)@(16,4)",
          lambda: ttnn.linear(p_sparse, b_w, compute_kernel_config=ckc, dtype=dt))
    timed("permute (1,L,128,4)->(1,4,L,128)",
          lambda: ttnn.permute(pb_tok, (0, 3, 1, 2)))

    print(f"\nL={L}  fp32 score chain (shipped form)")
    _, a1 = timed("typecast scores bf16->fp32",
                  lambda: ttnn.typecast(scores, ttnn.float32))
    _, a2 = timed("multiply fp32 x scale",
                  lambda: ttnn.multiply(scores_f, HD ** -0.5))
    _, a3 = timed("typecast bias bf16->fp32",
                  lambda: ttnn.typecast(bias, ttnn.float32))
    _, a4 = timed("add fp32+fp32", lambda: ttnn.add(scores_f, bias_f))
    _, a5 = timed("softmax fp32", lambda: ttnn.softmax(scores_f, dim=-1))
    _, a6 = timed("typecast attention fp32->bf16",
                  lambda: ttnn.typecast(scores_f, dt))
    print(f"  {'shipped chain total':<50s} {a1+a2+a3+a4+a5+a6:8.3f} ms")

    print(f"\nL={L}  fused alternatives")
    try:
        _, b1 = timed("multiply(bf16 in, scale, dtype=fp32)",
                      lambda: ttnn.multiply(scores, HD ** -0.5, dtype=ttnn.float32))
    except Exception as exc:  # noqa: BLE001
        print(f"    multiply bf16->fp32 unsupported: {exc}")
        b1 = None
    try:
        _, b2 = timed("add(fp32, bf16) mixed",
                      lambda: ttnn.add(scores_f, bias))
    except Exception as exc:  # noqa: BLE001
        print(f"    mixed add unsupported: {type(exc).__name__}")
        b2 = None
    try:
        _, b3 = timed("softmax(fp32 in, dtype=bf16 out)",
                      lambda: ttnn.softmax(scores_f, dim=-1, dtype=dt))
    except Exception as exc:  # noqa: BLE001
        print(f"    softmax dtype-out unsupported: {type(exc).__name__}")
        b3 = None
    try:
        _, b4 = timed("add(bf16,bf16,dtype=fp32)",
                      lambda: ttnn.add(scores, bias, dtype=ttnn.float32))
    except Exception as exc:  # noqa: BLE001
        print(f"    add bf16->fp32 unsupported: {type(exc).__name__}")
        b4 = None

    # bit-exactness of the fused multiply
    if b1 is not None:
        ref = ttnn.to_torch(ttnn.multiply(ttnn.typecast(scores, ttnn.float32), HD ** -0.5))
        got = ttnn.to_torch(ttnn.multiply(scores, HD ** -0.5, dtype=ttnn.float32))
        print(f"  fused multiply bit-exact: {torch.equal(ref, got)} "
              f"maxabs={(ref - got).abs().max().item():.3e}")
    if b4 is not None:
        ref = ttnn.to_torch(ttnn.add(ttnn.typecast(scores, ttnn.float32),
                                     ttnn.typecast(bias, ttnn.float32)))
        got = ttnn.to_torch(ttnn.add(scores, bias, dtype=ttnn.float32))
        print(f"  fused add bit-exact: {torch.equal(ref, got)} "
              f"maxabs={(ref - got).abs().max().item():.3e}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
