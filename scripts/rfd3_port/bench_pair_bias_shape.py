"""Price the sub-tile pair-bias projection against a flattened 2D matmul.

The atom blocks project the gathered sparse pair stream with
`ttnn.linear((1,L,K,16), (16,4))`. Both trailing dims are sub-tile (16 and 4 pad
to 32), and the 4D form leaves the matmul with L*K/32 one-tile-wide rows, so at
L=3359/K=128 it costs 6.2 ms for 55 MFLOP. Collapsing (L,K) into a single row
dim gives the same reduction over the same 16 elements in the same order, so it
is bit-exact by construction -- this measures whether it is also fast, and
checks the same question for the token-level (1,I,I,128)@(128,16) projection.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 -m scripts.rfd3_port.bench_pair_bias_shape
"""

from __future__ import annotations

import time

import torch


def main() -> None:
    import ttnn

    dt = ttnn.bfloat16
    device = ttnn.open_device(device_id=0)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
    )

    def up(t, dtype=dt):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    def timed(label, fn, reps=5):
        out = fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000 / reps
        print(f"  {label:<52s} {ms:8.3f} ms", flush=True)
        return out, ms

    for L, K, C_PAIR, H, tag in ((3359, 128, 16, 4, "atom block"),
                                 (250, 250, 128, 16, "token DiT")):
        print(f"\n=== {tag}: ({1},{L},{K},{C_PAIR}) @ ({C_PAIR},{H}) ===")
        p = up(torch.randn(1, L, K, C_PAIR))
        w = up(torch.randn(C_PAIR, H) * 0.05)

        ref, ms_ref = timed("shipped: linear 4D",
                            lambda: ttnn.linear(p, w, compute_kernel_config=ckc, dtype=dt))
        _, ms_rs = timed("reshape (1,L,K,C)->(1,L*K,C)",
                         lambda: ttnn.reshape(p, (1, L * K, C_PAIR)))
        p2 = ttnn.reshape(p, (1, L * K, C_PAIR))
        out2, ms_mm = timed("linear 2D (1,L*K,C)@(C,H)",
                            lambda: ttnn.linear(p2, w, compute_kernel_config=ckc, dtype=dt))
        _, ms_rs2 = timed("reshape (1,L*K,H)->(1,L,K,H)",
                          lambda: ttnn.reshape(out2, (1, L, K, H)))

        def flat():
            f = ttnn.reshape(p, (1, L * K, C_PAIR))
            o = ttnn.linear(f, w, compute_kernel_config=ckc, dtype=dt)
            return ttnn.reshape(o, (1, L, K, H))

        got, ms_all = timed("FLAT TOTAL (reshape+linear+reshape)", flat)
        print(f"  -> {ms_ref / ms_all:.2f}x  ({ms_ref:.3f} -> {ms_all:.3f} ms)")
        rt, gt = ttnn.to_torch(ref), ttnn.to_torch(got)
        print(f"  bit-exact: {torch.equal(rt, gt)}  maxabs={(rt - gt).abs().max().item():.3e}")

        # can the permute that follows absorb the un-flatten?
        _, ms_p4 = timed("permute (1,L,K,H)->(1,H,L,K)",
                         lambda: ttnn.permute(got, (0, 3, 1, 2)))
        del p, w, p2, out2, ref, got

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
