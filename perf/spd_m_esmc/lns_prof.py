"""Device-profiler read of ln_split vs ttnn.layer_norm at [1, 544, 2560] with a weight: every analysis the profiler
reports per program (kernel duration, per-RISC spans where present), to place ln_split's ~40 us fixed cost.
Needs the Tracy build and TT_METAL_DEVICE_PROFILER=1 (the dc1 grid job's environment).

    python perf/spd_m_esmc/lns_prof.py --out prof.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import ttnn

from tt_bio import ln_split
from tt_bio.tenstorrent import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = get_device()
k = 2560
x = ttnn.from_torch(torch.randn(1, 544, k).to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
w = ttnn.from_torch(torch.randn(1, k).to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, packer_l1_acc=False)
out = {}
for name, fn in (("stock", lambda: ttnn.layer_norm(x, weight=w, epsilon=1e-5, compute_kernel_config=ck)),
                 ("split", lambda: ln_split.layer_norm(x, w, None, 1e-5))):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    for _ in range(5):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    rows = []
    for _chip, ps in ttnn.get_latest_programs_perf_data().items():
        for p in ps:
            rows.append({key: getattr(v, "duration", str(v)) for key, v in p.program_analyses_results.items()})
    out[name] = rows
    print(name, json.dumps(rows[-2:]), flush=True)
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
