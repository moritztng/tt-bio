"""p31: price every way of splitting [b,L,H*D] into [b,H,L,D] at RFD3's three head shapes.

p29 named `reshape [1,250,16,48]` (the DiT block's `heads()`, 144 calls/step) at 9.19 ms
device and 12 GB/s -- 20x off the elementwise rate the same bytes reach elsewhere. The
suspected cause is tile padding: head_dim=48 pads to 64 and the head axis (16) pads to 32,
so the 4D intermediate is 2.7x the bytes of the 2D source and every element is physically
moved twice (once by `reshape`, once by `permute`).

Shapes priced, all three head splits RFD3 actually runs:
  * DiT block          [1, 250, 768]  -> 16 heads x 48   (144 calls/step)
  * atom block         [1, L,   128]  -> 4 heads  x 32   (9 calls/step, tile-aligned d)
  * pairformer         [1, 250, 384]  -> 16 heads x 24   (4 blocks x 4 calls/step)

Variants: the shipped reshape+permute; the same with the two ops swapped; the fused
tt-metal head kernels (`nlp_create_qkv_heads`, `create_qkv_heads_from_separate_tensors`,
`transformer.split_query_key_value_and_split_heads`); and the concatenate side.

Every variant reports maxabs against the shipped form's output, because bit-exact is this
lineage's gate. Timing is median of `reps` sync-bracketed calls on a hot card.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       /home/moritz/tt-bio/env/bin/python3 scripts/rfd3_port/p31_heads_variants.py
"""

from __future__ import annotations

import time

import torch

# (label, batch, length, n_head, head_dim, calls per step)
CASES = [
    ("DiT      [1,250,768]->16x48", 1, 250, 16, 48, 144),
    ("atom     [1,3359,128]->4x32", 1, 3359, 4, 32, 36),
    ("pairform [1,250,384]->16x24", 1, 250, 16, 24, 16),
]


def main() -> None:
    import ttnn

    device = ttnn.open_device(device_id=0)
    dt = ttnn.bfloat16

    def timed(fn, reps=8):
        """Median of `reps` sync-bracketed calls; None if the op refuses the shapes."""
        try:
            for _ in range(2):
                out = fn()
            ttnn.synchronize_device(device)
        except Exception as exc:                       # noqa: BLE001 - probing an API
            return None, None, str(exc).split("\n")[0][:110]
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2], out, None

    for label, b, L, H, D, calls in CASES:
        C = H * D
        print(f"\n=== {label}   ({calls} calls/step) ===")
        x_host = torch.randn(b, L, C, dtype=torch.float32)
        x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)
        x4 = ttnn.reshape(x, (b, 1, L, C))
        bytes_mb = b * L * C * 2 / 1e6

        def shipped():
            t = ttnn.reshape(x, (b, L, H, D))
            return ttnn.permute(t, (0, 2, 1, 3))

        ms_ref, ref, err = timed(shipped)
        ref_host = ttnn.to_torch(ref).float()
        print(f"  {'shipped reshape+permute':<46} {ms_ref:7.3f} ms  "
              f"{2 * bytes_mb / ms_ref:6.1f} GB/s  -> step {ms_ref * calls:7.2f} ms")

        def report(name, fn, post=None):
            ms, out, err = timed(fn)
            if ms is None:
                print(f"  {name:<46} {'--':>7}      {err}")
                return
            got = post(out) if post else ttnn.to_torch(out).float()
            maxabs = (got - ref_host).abs().max().item() if got.shape == ref_host.shape \
                else float("nan")
            tag = "BIT-EXACT" if maxabs == 0.0 else f"maxabs {maxabs:.3e}"
            print(f"  {name:<46} {ms:7.3f} ms  {2 * bytes_mb / ms:6.1f} GB/s  "
                  f"-> step {ms * calls:7.2f} ms   {ms_ref / ms:5.2f}x   {tag}")

        # 1. same two ops, permuting the 4D-reshaped source instead.
        report("reshape(4D)+permute", lambda: ttnn.permute(
            ttnn.reshape(x4, (b, L, H, D)), (0, 2, 1, 3)))

        # 2. fused single-tensor head split. nlp_create_qkv_heads splits one packed
        #    [b,1,L,3*C] into q/k/v; feed it a 3x-wide tensor and take the first output.
        x3_host = x_host.repeat(1, 1, 3).reshape(b, 1, L, 3 * C)
        x3 = ttnn.from_torch(x3_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)
        report("nlp_create_qkv_heads(packed qkv) [q only]",
               lambda: ttnn.experimental.nlp_create_qkv_heads(
                   x3, num_heads=H, transpose_k_heads=False)[0])

        # 3. three separate tensors -> three head-split tensors, one kernel.
        report("create_qkv_heads_from_separate_tensors [q]",
               lambda: ttnn.experimental.create_qkv_heads_from_separate_tensors(
                   x4, ttnn.reshape(
                       ttnn.from_torch(x_host.repeat(1, 1, 2), layout=ttnn.TILE_LAYOUT,
                                       device=device, dtype=dt), (b, 1, L, 2 * C)),
                   num_q_heads=H, num_kv_heads=H, transpose_k_heads=False)[0])

        # 4. the ttnn.transformer front door.
        report("transformer.split_qkv_and_split_heads",
               lambda: ttnn.transformer.split_query_key_value_and_split_heads(
                   x3, num_heads=H)[0])

        # 5. the reverse direction, priced against its own reference.
        print(f"  --- concatenate side [b,H,L,D] -> [b,L,C] ---")
        h_host = torch.randn(b, H, L, D, dtype=torch.float32)
        h = ttnn.from_torch(h_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)

        def shipped_cat():
            t = ttnn.permute(h, (0, 2, 1, 3))
            return ttnn.reshape(t, (b, L, C))

        ms_c, refc, _ = timed(shipped_cat)
        refc_host = ttnn.to_torch(refc).float()
        print(f"  {'shipped permute+reshape':<46} {ms_c:7.3f} ms  "
              f"{2 * bytes_mb / ms_c:6.1f} GB/s  -> step {ms_c * calls // 4:7.2f} ms")

        def report_c(name, fn):
            ms, out, err = timed(fn)
            if ms is None:
                print(f"  {name:<46} {'--':>7}      {err}")
                return
            got = ttnn.to_torch(out).float().reshape(refc_host.shape) \
                if ttnn.to_torch(out).numel() == refc_host.numel() else ttnn.to_torch(out).float()
            maxabs = (got - refc_host).abs().max().item() if got.shape == refc_host.shape \
                else float("nan")
            tag = "BIT-EXACT" if maxabs == 0.0 else f"maxabs {maxabs:.3e}"
            print(f"  {name:<46} {ms:7.3f} ms  {2 * bytes_mb / ms:6.1f} GB/s  "
                  f"{ms_c / ms:5.2f}x   {tag}")

        report_c("experimental.nlp_concat_heads", lambda:
                 ttnn.experimental.nlp_concat_heads(h))
        report_c("transformer.concatenate_heads", lambda:
                 ttnn.transformer.concatenate_heads(h))

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
