"""Why does `ttnn.scatter` into the dense (1,4,L,L) attention bias cost 4.56 ms?

`bench_atom_block_sections.py` attributes 38% of a real atom block at L=3359 to
one op: scattering the sparse (1,4,L,128) pair bias into the dense -1e4 mask.
Its traffic is ~180 MB (read + write of a 90 MB bf16 tensor), which at the
p150a's measured 405 GB/s should take 0.45 ms. It takes 10x that.

This prices the knobs that could explain the gap -- index layout, index dtype,
`sub_core_grids`, output dtype, and how the cost scales with the scattered
column count -- so the next step is aimed at a cause rather than guessed.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 -m scripts.rfd3_port.bench_bias_scatter [L]
"""

from __future__ import annotations

import sys
import time

import torch


def main() -> None:
    import ttnn

    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3359
    H, K = 4, 128
    dt = ttnn.bfloat16
    device = ttnn.open_device(device_id=0)

    g = torch.Generator().manual_seed(0)
    idx = torch.stack([torch.sort(torch.randperm(L, generator=g)[:K])[0] for _ in range(L)])
    idx4 = idx.view(1, 1, L, K).expand(1, H, L, K).contiguous()

    def up(t, dtype=dt, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, layout=layout, device=device, dtype=dtype)

    def timed(label, fn, reps=6):
        try:
            for _ in range(2):
                out = fn()
            ttnn.synchronize_device(device)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<48s}  unsupported: {type(exc).__name__}", flush=True)
            return None
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            out = fn()
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        samples.sort()
        ms = samples[len(samples) // 2]
        print(f"  {label:<48s} {ms:8.3f} ms", flush=True)
        return out, ms

    dense_bf = up(torch.full((1, H, L, L), -1e4))
    dense_f32 = up(torch.full((1, H, L, L), -1e4), dtype=ttnn.float32)
    src_bf = up(torch.randn(1, H, L, K))
    src_f32 = up(torch.randn(1, H, L, K), dtype=ttnn.float32)
    i_tile_u32 = up(idx4.to(torch.int32), dtype=ttnn.uint32)
    i_rm_u32 = up(idx4.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    i_tile_i32 = up(idx4.to(torch.int32), dtype=ttnn.int32)

    print(f"L={L} H={H} K={K}  dense bf16={H*L*L*2/1e6:.0f} MB, fp32={H*L*L*4/1e6:.0f} MB")
    print("\n--- shipped form and index variants ---")
    ref, base = timed("scatter bf16, uint32 index, TILE  [SHIPPED]",
                      lambda: ttnn.scatter(dense_bf, 3, i_tile_u32, src_bf))
    timed("scatter bf16, uint32 index, ROW_MAJOR",
          lambda: ttnn.scatter(dense_bf, 3, i_rm_u32, src_bf))
    timed("scatter bf16, int32 index, TILE",
          lambda: ttnn.scatter(dense_bf, 3, i_tile_i32, src_bf))
    timed("scatter fp32 dense + fp32 src",
          lambda: ttnn.scatter(dense_f32, 3, i_tile_u32, src_f32))

    print("\n--- core-grid knob ---")
    grid = device.compute_with_storage_grid_size()
    full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                             ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    print(f"  (compute grid {grid.x}x{grid.y})")
    timed("scatter bf16, sub_core_grids=full grid",
          lambda: ttnn.scatter(dense_bf, 3, i_tile_u32, src_bf, sub_core_grids=full))

    print("\n--- how does it scale? (isolate per-column vs per-dense-element cost) ---")
    for k in (32, 64):
        sub = idx[:, :k].contiguous().view(1, 1, L, k).expand(1, H, L, k).contiguous()
        i_k = up(sub.to(torch.int32), dtype=ttnn.uint32)
        s_k = up(torch.randn(1, H, L, k))
        timed(f"scatter bf16, K={k}", lambda i_k=i_k, s_k=s_k:
              ttnn.scatter(dense_bf, 3, i_k, s_k))
    for h in (1, 2):
        d_h = up(torch.full((1, h, L, L), -1e4))
        i_h = up(idx.view(1, 1, L, K).expand(1, h, L, K).contiguous().to(torch.int32),
                 dtype=ttnn.uint32)
        s_h = up(torch.randn(1, h, L, K))
        timed(f"scatter bf16, H={h} (dense {h*L*L*2/1e6:.0f} MB)",
              lambda d_h=d_h, i_h=i_h, s_h=s_h: ttnn.scatter(d_h, 3, i_h, s_h))

    print("\n--- bandwidth reference on the same dense tensor ---")
    timed("ttnn.add(dense_bf, dense_bf)", lambda: ttnn.add(dense_bf, dense_bf))
    timed("ttnn.typecast(dense_bf -> fp32)",
          lambda: ttnn.typecast(dense_bf, ttnn.float32))
    timed("ttnn.softmax(dense_f32)", lambda: ttnn.softmax(dense_f32, dim=-1))

    print("\n--- can the separate fp32 typecast of the bias be dropped? ---")
    scores_f32 = up(torch.randn(1, H, L, L), dtype=ttnn.float32)
    bias_bf = up(torch.randn(1, H, L, L))
    timed("shipped: typecast(bias)->fp32 then add(fp32,fp32)",
          lambda: ttnn.add(scores_f32, ttnn.typecast(bias_bf, ttnn.float32)))
    timed("mixed:   add(fp32, bf16) directly",
          lambda: ttnn.add(scores_f32, bias_bf))
    a = ttnn.to_torch(ttnn.add(scores_f32, ttnn.typecast(bias_bf, ttnn.float32)))
    b = ttnn.to_torch(ttnn.add(scores_f32, bias_bf))
    print(f"  mixed add bit-exact vs shipped: {torch.equal(a, b)} "
          f"maxabs={(a - b).abs().max().item():.3e} dtype={b.dtype}")
    _ = ref, base

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
