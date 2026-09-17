#!/usr/bin/env python3
"""Does MY generated dataflow pair work WITHOUT the gate?

gate_correct.py's reference arm uses the SHIPPED ttnn.experimental.minimal_matmul, so a hang in the
gated arm has two candidate causes that it cannot separate:
  (a) the MM_GATE output stage and the halved writer geometry, or
  (b) patch_mm_split.py's generated pair / the compute_dir override, which the gated arm is the
      first caller to exercise together.
This runs G.generic_minimal_matmul at the SAME shape with gate=False then gate=True and prints
between them, so whichever hangs names the cause.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, ttnn
from tt_bio import mm_generic as G

TILE = 32
REPO = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO / "tt_bio" / "kernels" / "mm_split"
ROLES = ("p_a", "g_a", "p_b", "g_b")
h, k, slice_c, group = 128, 128, 128, 4
C = slice_c // group
n_blocks = 4 * group
N = n_blocks * C

torch.manual_seed(0)
x_t = (torch.randn(1, h, h, k) * 0.5).bfloat16()
w_t = (torch.randn(k, N) * 0.1).bfloat16()

from tt_bio.tenstorrent import get_device, _MM_DEFAULT, COMPUTE_GRID_MAIN
dev = get_device()
print("STAGE open-ok", flush=True)
try:
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = (_MM_DEFAULT, tuple(COMPUTE_GRID_MAIN))
    dt = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w = dt(x_t), dt(w_t)

    def dests(n_out_tiles, count):
        return [ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, h, h, n_out_tiles * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
            ttnn.DRAM_MEMORY_CONFIG) for _ in range(count)]

    # (gate, n_chunks). N_chunks=1 takes write_block_sync / write_block_sync_granular where
    # N_chunks=2 takes the _split variants, so a pass on one and a hang on the other localises the
    # halving defect to the chunk mapping rather than to the gate's own geometry.
    for gate, nchunks in ((False, 2), (True, 1), (True, 2)):
        total = n_blocks // 2 if gate else n_blocks
        nw = total // nchunks
        outs = dests(nw, nchunks)
        print(f"STAGE gate={gate} nchunks={nchunks} n_widths={[nw]*nchunks} ENQUEUE", flush=True)
        t0 = time.perf_counter()
        G.generic_minimal_matmul(
            dev, x, w, outs, cfg, G.ckc_args(ckc), {"MM_DUAL_NOC": 1}, KERNEL_DIR, None,
            ttnn.NOC_MODE.DM_DYNAMIC_NOC, [nw] * nchunks, KERNEL_DIR, gate)
        print(f"STAGE gate={gate} nchunks={nchunks} ENQUEUED", flush=True)
        got = ttnn.to_torch(outs[0])
        print(f"STAGE gate={gate} nchunks={nchunks} SYNCED in {time.perf_counter()-t0:.3f}s "
              f"shape={tuple(got.shape)} finite={bool(torch.isfinite(got.float()).all())}", flush=True)
        for o in outs:
            ttnn.deallocate(o)
    print("STAGE ALL-DONE", flush=True)
finally:
    ttnn.close_device(dev)
