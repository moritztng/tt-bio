#!/usr/bin/env python3
"""Device verdict for the one guard divergence: c_z=128 qkv at N=224.

Main's _tri_att_qkv_l1_config admits this shape (CB-only budget 1376256 <= 1532448).
The generalized helper rejects it, because it also charges the L1-resident output
tensor (per_core_M*n_tiles tiles per core) against the same per-core budget:
1376256 + 344064 = 1720320 > 1532448. If CBs and the output tensor share the
per-core unreserved pool, main's config must throw here; if the device runs it
cleanly and bit-exact, the helper is over-conservative at this one point.

Runs three arms on one device:
  MAIN   main's exact config (per_core_M=14, in0_block_w=k_tiles=4), output in L1
  NEW    the helper's answer (None -> the production minimal_matmul fallback)
  REF    minimal_matmul in DRAM, the reference both must match bit-exact
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import torch
import ttnn

from tt_bio.tenstorrent import get_device, cleanup
import tt_bio.tenstorrent as T

minimal_matmul = ttnn.experimental.minimal_matmul


def main():
    torch.set_grad_enabled(False)
    dev = get_device()
    assert T.COMPUTE_GRID_MAIN == (13, 10), f"expected the 13x10 grid, got {T.COMPUTE_GRID_MAIN}"

    N, C = 224, 128
    M, K, NN = N * N, C, 3 * C
    x_t = torch.randn(M, K, dtype=torch.float32).bfloat16()
    w_t = torch.randn(K, NN, dtype=torch.float32).bfloat16() * 0.05
    x = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    ref = minimal_matmul(x, w)
    ref_t = ttnn.to_torch(ref)

    # NEW arm: what the helper decides for this shape.
    new_cfg = T._l1_resident_linear_config(x, w, ttnn.bfloat16)
    print(f"helper decision at c_z=128 N=224 qkv: {new_cfg}")

    # MAIN arm: main's exact config, reconstructed from cd4b71e67.
    m_tiles, k_tiles, n_tiles = M // 32, K // 32, NN // 32
    per_core_M = 14  # main's search: smallest p >= ceil(1568/130)=13 dividing 1568
    assert m_tiles % per_core_M == 0
    main_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN,
        in0_block_w=k_tiles,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=per_core_M,
        out_block_w=n_tiles,
        per_core_M=per_core_M,
        per_core_N=n_tiles,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    l1_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    try:
        out = ttnn.linear(x, w, program_config=main_cfg, memory_config=l1_mc, dtype=ttnn.bfloat16)
        out_t = ttnn.to_torch(out)
        ok = bool(torch.equal(ref_t, out_t))
        print(f"MAIN config ran on device; bit_exact vs minimal_matmul: {ok}")
        print(f"output memory: {out.memory_config().buffer_type}")
    except Exception as e:
        print(f"MAIN config THREW on device: {type(e).__name__}: {str(e)[:300]}")
    cleanup()


if __name__ == "__main__":
    main()
