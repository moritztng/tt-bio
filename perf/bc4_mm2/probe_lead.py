#!/usr/bin/env python3
"""lead_sum after rne_add's writer grew DUAL: compiles, and matches float64 at BindCraft 2's shapes."""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); A = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, lead_sum as LS
dev = T.get_device(); rows = []
g = torch.Generator().manual_seed(3)
for shape, dt in (([2, 224, 224, 128], torch.bfloat16), ([4, 224, 224, 128], torch.float32), ([2, 8, 224, 224], torch.bfloat16)):
    h = torch.randn(*shape, generator=g).to(dt)
    x = ttnn.from_torch(h, dtype=ttnn.bfloat16 if dt == torch.bfloat16 else ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = ttnn.to_torch(LS.lead_sum(x)).double()
    ref = h.double().sum(0, keepdim=True)
    rows.append(dict(shape=shape, dtype=str(dt), rel_f64=float((got - ref).norm() / ref.norm()), served=LS.STATS[0]))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(A.out, "w"), indent=1)
