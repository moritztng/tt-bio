"""Does ttnn's fill_pad reuse one shape's program for another? Run each shape through
ttnn.fill_implicit_tile_padding in one process with the program cache on, and report per call whether it added a cache
entry and whether the padding (and only the padding) came out as the fill value.

usage: TT_VISIBLE_DEVICES=<card> python fillpad_cache.py [--dtype bf16|fp32] [--mem dram|l1] SHAPE [SHAPE ...]
SHAPE is comma separated, e.g. 5,5941,128 1,6202,128. Every shape is run fresh, then the list again (warm).
"""
import argparse
import json

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
ap.add_argument("--mem", choices=("dram", "l1"), default="dram")
ap.add_argument("--fill", type=float, default=3.0)
ap.add_argument("shapes", nargs="+")
a = ap.parse_args()
ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()
dt = ttnn.bfloat16 if a.dtype == "bf16" else ttnn.float32
mem = ttnn.DRAM_MEMORY_CONFIG if a.mem == "dram" else ttnn.L1_MEMORY_CONFIG
shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes]

def padded(t):
    h = ttnn.from_device(t)
    return torch.as_tensor(h.to_torch_with_padded_shape()).float() if hasattr(h, "to_torch_with_padded_shape") else None


for rnd in ("fresh", "warm"):
    for sh in shapes:
        src = torch.randn(sh)
        x = ttnn.from_torch(src, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)
        n0 = dev.num_program_cache_entries()
        y = ttnn.fill_implicit_tile_padding(x, a.fill)
        n1 = dev.num_program_cache_entries()
        out, rec = padded(y), dict(round=rnd, shape=sh, new_cache_entries=n1 - n0)
        if out is not None:
            box = tuple(slice(0, d) for d in sh)
            mask = torch.ones(out.shape, dtype=torch.bool)
            mask[box] = False
            rec.update(data_maxerr=float((out[box] - src.to(out.dtype if a.dtype == "fp32" else torch.bfloat16).float())
                                         .abs().max()),
                       pad_bad=int((out[mask] != a.fill).sum()), pad_total=int(mask.sum()))
        print(json.dumps(rec), flush=True)
        ttnn.deallocate(y)
ttnn.synchronize_device(dev)
print("done")
