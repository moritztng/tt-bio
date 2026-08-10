#!/usr/bin/env python3
"""Does the L1 guard actually fire at the fold's own shapes? (`tt-bio-l1-residency-guard-dead-in-real-folds`)"""
import sys
from pathlib import Path
import torch, ttnn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T

dev = T.get_device()
print("grid after device open:", T.COMPUTE_GRID_MAIN, flush=True)
print("get_max_worker_l1_unreserved_size:", int(ttnn.get_max_worker_l1_unreserved_size()), flush=True)
print("l1_bank_bytes:", T._l1_bank_bytes(), flush=True)
z = ttnn.from_torch(torch.randn(1, 298, 298, 256), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
for h in (1.5, 2.5):
    mc = T._l1_memory_config_if_it_fits(z, h)
    print(f"headroom {h}: {'L1' if mc is ttnn.L1_MEMORY_CONFIG else 'DRAM'}", flush=True)
w = ttnn.from_torch(torch.randn(256, 16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
zn = ttnn.layer_norm(z, epsilon=1e-5, memory_config=T._l1_memory_config_if_it_fits(z, 1.5))
print("layer_norm out buffer:", zn.memory_config().buffer_type, flush=True)
out = T._narrow_proj_linear(zn, w, None, ttnn.bfloat16, l1_out=True)
print("narrow proj out:", None if out is None else out.memory_config().buffer_type, flush=True)
wz = ttnn.from_torch(torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
p = T._pair_proj_linear(z, wz, None, ttnn.bfloat16, l1_out=True)
print("pair proj out:", p.memory_config().buffer_type, flush=True)
print("refused:", T._L1_OUT_REFUSED, flush=True)
