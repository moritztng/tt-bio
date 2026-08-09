"""Hardware test for the batched-matmul program config (tt_bio.tenstorrent.batched_matmul).

Locks three things the perfwar D3 leg had to measure the hard way:

  1. the chooser fires at the real 298 aa diffusion shapes, in both models and both dtypes -- a
     config that silently declines is a silent 1.7 s/fold regression;
  2. it never returns per_core_M=1 against a multi-tile M. That is not a tuning preference: on
     live operands it returns wrong results (up to 26.9 absolute on the DiT attention), and it
     does not reproduce on tensors built in isolation, so only this assertion catches it;
  3. every config it does return is bit-exact against the plain ttnn.matmul it replaces.

Run: TT_VISIBLE_DEVICES=<card> python3 tests/test_batched_matmul_hw.py
"""
import torch
import ttnn

from tt_bio import tenstorrent as T

F32, BF16 = ttnn.float32, ttnn.bfloat16

# (a shape, b shape, dtype, must the chooser take it?) at the real 298 aa dimensions.
CASES = [
    ((75, 4, 32, 128), (75, 4, 128, 32), F32, True),    # atom attention QK^T, protenix-v2
    ((75, 4, 32, 32), (75, 4, 32, 128), F32, True),     # atom attention AV, protenix-v2
    ((1, 16, 320, 320), (1, 16, 320, 64), F32, True),   # DiT AV, protenix-v2
    ((1, 16, 320, 64), (1, 16, 64, 320), F32, False),   # DiT QK^T: per_core_N would be 10, a loss
    ((75, 4, 32, 128), (75, 4, 128, 32), BF16, True),   # atom attention QK^T, opendde
    ((75, 4, 32, 32), (75, 4, 32, 128), BF16, True),    # atom attention AV, opendde
    ((1, 16, 608, 608), (1, 16, 608, 64), BF16, True),  # DiT AV, opendde
    ((1, 8, 608, 608), (1, 8, 608, 64), BF16, True),    # DiT AV, opendde tail chunk
    ((1, 16, 608, 64), (1, 16, 64, 608), BF16, False),  # DiT QK^T: 0.65x, must decline
]


def test_batched_matmul_config():
    dev = ttnn.open_device(device_id=0)
    try:
        T._configure_active_compute_grid(dev)
        ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        for sa, sb, dt, want in CASES:
            torch.manual_seed(0)
            a = ttnn.from_torch(torch.randn(sa), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = ttnn.from_torch(torch.randn(sb), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            m_tiles = -(-sa[-2] // 32)
            cfg = T._batched_reuse_config(sa[0] * sa[1], m_tiles, -(-sa[-1] // 32),
                                          -(-sb[-1] // 32), 4 if dt == F32 else 2)
            assert (cfg is not None) == want, f"{sa}x{sb} {dt}: expected applied={want}"
            if cfg is not None:
                assert not (cfg.per_core_M == 1 and m_tiles > 1), (
                    f"{sa}x{sb}: per_core_M=1 against Mt={m_tiles} returns wrong results on live "
                    f"operands")
                ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
                got = ttnn.to_torch(T.batched_matmul(a, b, compute_kernel_config=ckc))
                assert torch.equal(ref, got), f"{sa}x{sb} {dt}: not bit-exact"
            print(f"  ok  {str(sa):20s} x {str(sb):20s} {str(dt).split('.')[-1]:9s} "
                  + ("declined" if cfg is None else
                     f"per_core_M={cfg.per_core_M} per_core_N={cfg.per_core_N} "
                     f"in0_block_w={cfg.in0_block_w}, bit-exact"))
            ttnn.deallocate(a)
            ttnn.deallocate(b)
    finally:
        ttnn.close_device(dev)
    print("PASS")



if __name__ == "__main__":
    test_batched_matmul_config()
