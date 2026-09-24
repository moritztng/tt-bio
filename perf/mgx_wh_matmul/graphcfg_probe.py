#!/usr/bin/env python3
"""Can the program config ttnn resolves for an auto-configured matmul be read back from Python?
Prints whatever ttnn.graph records for one ttnn.matmul call."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
ta = ttnn.from_torch(torch.randn(1, 1, 4096, 768).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
tb = ttnn.from_torch(torch.randn(768, 1536).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
o = ttnn.matmul(ta, tb, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
g = ttnn.graph.end_graph_capture()
for node in g:
    s = json.dumps(node, default=str)
    if "in0_block_w" in s or "program_config" in s or "Matmul" in s:
        print(s[:3000])
