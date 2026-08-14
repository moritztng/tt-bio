"""Hardware test for the batched-matmul program config (tt_bio.tenstorrent.batched_matmul).

Locks four things that were each measured the hard way:

  1. the chooser fires at the real 298 aa shapes, in every model and both dtypes -- a config that
     silently declines is a silent multi-second regression;
  2. it never emits a config where a core takes more than one output block while M is split within
     a batch element. That is not a tuning preference: both dataflow kernels advance by a whole
     batch stride once per BLOCK, so such a config returns wrong results (up to 26.9 absolute on
     the DiT attention). It does not reproduce on tensors built in isolation, so only this
     assertion catches it;
  3. every config it does return is bit-exact against the plain ttnn.matmul it replaces;
  4. per_core_M is a pure performance knob -- only in0_block_w can change the result -- so the
     sweep asserts exactness at every legal per_core_M, not just the one the chooser picks.

`in0_block_w` is the parity knob and this file is what pins it per shape: ttnn's 1D factories take
`Kt % 2 == 0 ? 2 : 1`, which the helper reproduces, but its 2D factory's width comes out of
`get_multi_dim_per_core_factor` and cannot be computed outside ttnn. **A new call site means a new
case here.**

Run: TT_VISIBLE_DEVICES=<card> python3 tests/test_batched_matmul_hw.py
"""
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.main import ensure_p300_mesh_descriptor

F32, BF16 = ttnn.float32, ttnn.bfloat16

# (a shape, b shape, dtype, must the chooser take it?) at the real 298 aa dimensions.
CASES = [
    ((75, 4, 32, 128), (75, 4, 128, 32), F32, True),    # atom attention QK^T, protenix-v2
    ((75, 4, 32, 32), (75, 4, 32, 128), F32, True),     # atom attention AV, protenix-v2
    ((1, 16, 320, 320), (1, 16, 320, 64), F32, True),   # DiT AV, protenix-v2 and openfold3
    ((1, 16, 320, 64), (1, 16, 64, 320), F32, True),    # DiT QK^T
    ((75, 4, 32, 128), (75, 4, 128, 32), BF16, True),   # atom attention QK^T, opendde
    ((75, 4, 32, 32), (75, 4, 32, 128), BF16, True),    # atom attention AV, opendde
    ((1, 16, 608, 608), (1, 16, 608, 64), BF16, True),  # DiT AV, opendde
    ((1, 8, 608, 608), (1, 8, 608, 64), BF16, True),    # DiT AV, opendde tail chunk
    ((1, 16, 608, 64), (1, 16, 64, 608), BF16, False),  # DiT QK^T at Nt=19: no config fits L1
    # openfold3. Its tri-attention is the shared raw-attention helper rather than the fused SDPA
    # the other models take, so these are the only trunk sites in the audit.
    ((298, 4, 298, 298), (298, 4, 298, 32), BF16, True),  # tri-attention AV
    ((298, 4, 298, 32), (298, 4, 32, 298), BF16, True),   # tri-attention QK^T
    ((1, 16, 298, 298), (1, 16, 298, 32), BF16, True),    # AttentionPairBias AV
    ((1, 64, 298, 298), (1, 64, 298, 298), BF16, True),   # trimul class: 2D branch, width 1
    # rank 5: the openfold3 atom transformer's windowed attention. ttnn accepts the config here
    # and returns bit-exact, so rank is not the gate -- every leading dim is one batch element.
    ((1, 75, 4, 32, 32), (1, 75, 4, 32, 128), F32, True),   # atom AV
    ((1, 75, 4, 32, 128), (1, 75, 4, 128, 32), F32, True),  # atom QK^T
]


def _legal_per_core_M(batch, m_tiles, n_tiles, block_w, elem_bytes, cores, l1):
    tile, acc = 1024 * elem_bytes, 4096
    return [p for p in range(1, m_tiles + 1)
            if m_tiles % p == 0
            and (p == m_tiles or batch * m_tiles // p <= cores)
            and 2 * (p + n_tiles) * block_w * tile + p * n_tiles * (tile + acc) <= l1]


def test_batched_matmul_config():
    # This opens the device directly rather than through T.get_device(), so it has to
    # apply the lone-P300 mesh-graph descriptor itself or open_device is a TT_FATAL.
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=0)
    try:
        T._configure_active_compute_grid(dev)
        gx, gy = T.COMPUTE_GRID_MAIN
        cores = gx * gy
        l1 = int(ttnn.get_max_worker_l1_unreserved_size())
        ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        for sa, sb, dt, want in CASES:
            torch.manual_seed(0)
            a = ttnn.from_torch(torch.randn(sa), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = ttnn.from_torch(torch.randn(sb), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            m_tiles, k_tiles = -(-sa[-2] // 32), -(-sa[-1] // 32)
            n_tiles = -(-sb[-1] // 32)
            batch = 1
            for d in sa[:-2]:
                batch *= d
            eb = 4 if dt == F32 else 2
            cfg = T._batched_matmul_config(batch, m_tiles, k_tiles, n_tiles, eb)
            assert (cfg is not None) == want, f"{sa}x{sb} {dt}: expected applied={want}"
            if cfg is not None:
                blocks = batch * m_tiles // cfg.per_core_M
                assert cfg.per_core_M == m_tiles or blocks <= cores, (
                    f"{sa}x{sb}: per_core_M={cfg.per_core_M} splits Mt={m_tiles} and puts "
                    f"{blocks} blocks on {cores} cores, so a core takes more than one block "
                    f"and the kernels stride by a whole batch per block -- wrong results")
                ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
                got = ttnn.to_torch(T.batched_matmul(a, b, compute_kernel_config=ckc))
                assert torch.equal(ref, got), f"{sa}x{sb} {dt}: not bit-exact"
                # per_core_M must not be able to change the result at this width.
                for p in _legal_per_core_M(batch, m_tiles, n_tiles, cfg.in0_block_w, eb, cores, l1):
                    alt = ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=cfg.in0_block_w,
                        out_subblock_h=cfg.out_subblock_h, out_subblock_w=cfg.out_subblock_w,
                        per_core_M=p, per_core_N=n_tiles)
                    if p % cfg.out_subblock_h:
                        continue  # subblock has to divide the block it tiles
                    alt_got = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                        program_config=alt))
                    assert torch.equal(ref, alt_got), (
                        f"{sa}x{sb} {dt}: per_core_M={p} at in0_block_w={cfg.in0_block_w} is not "
                        f"bit-exact -- per_core_M is supposed to be a performance knob only")
                    del alt_got
                del ref, got
            print(f"  ok  {str(sa):22s} x {str(sb):22s} {str(dt).split('.')[-1]:9s} "
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
