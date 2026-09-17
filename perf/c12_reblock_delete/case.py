#!/usr/bin/env python3
"""ONE MM_GATE case per process, so a hang costs only that case.

Axes, chosen to split the hang's cause rather than to explore:
  --gate       the epilogue on/off
  --nchunks    1 takes write_block_sync/_granular, 2 takes the _split variants
  --defines    dual  = MM_DUAL_NOC (production), the writes alternate NOC on mm_write_seq parity
               nowrite = MM_NOWRITE, every CB round trip kept and every NOC write stubbed out
               plain = neither, single-NOC writes
A hang under nowrite is a CB-protocol hang; a hang that only appears with writes enabled is an
addressing or transaction-retirement fault. That is the split the watcher could not make.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, ttnn
from tt_bio import mm_generic as G

TILE = 32
KERNEL_DIR = Path(__file__).resolve().parents[2] / "tt_bio" / "kernels" / "mm_split"

ap = argparse.ArgumentParser()
ap.add_argument("--gate", type=int, default=1)
ap.add_argument("--nchunks", type=int, default=1)
ap.add_argument("--defines", default="dual", choices=("dual", "nowrite", "plain"))
ap.add_argument("--h", type=int, default=128)
a = ap.parse_args()

h, k, group = a.h, 128, 4
C, n_blocks = 128 // group, 4 * group
N = n_blocks * C
gate = bool(a.gate)
total = n_blocks // 2 if gate else n_blocks
nw = total // a.nchunks

defines = {"dual": {"MM_DUAL_NOC": 1}, "nowrite": {"MM_NOWRITE": 1}, "plain": {}}[a.defines]
noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if a.defines == "dual" else None

torch.manual_seed(0)
x_t = (torch.randn(1, h, h, k) * 0.5).bfloat16()
w_t = (torch.randn(k, N) * 0.1).bfloat16()

from tt_bio.tenstorrent import get_device, _MM_DEFAULT, COMPUTE_GRID_MAIN
label = f"gate={a.gate} nchunks={a.nchunks} defines={a.defines} n_widths={[nw]*a.nchunks}"
dev = get_device()
try:
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = (_MM_DEFAULT, tuple(COMPUTE_GRID_MAIN))
    dt = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w = dt(x_t), dt(w_t)
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, h, h, nw * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG) for _ in range(a.nchunks)]
    print(f"CASE {label} ENQUEUE", flush=True)
    t0 = time.perf_counter()
    G.generic_minimal_matmul(dev, x, w, outs, cfg, G.ckc_args(ckc), defines, KERNEL_DIR, None,
                             noc_mode, [nw] * a.nchunks, KERNEL_DIR, gate)
    got = ttnn.to_torch(outs[0])
    fin = bool(torch.isfinite(got.float()).all())
    nz = int((got.float() != 0).sum())
    print(f"CASE {label} PASS {time.perf_counter()-t0:.3f}s shape={tuple(got.shape)} "
          f"finite={fin} nonzero={nz}", flush=True)
finally:
    ttnn.close_device(dev)
