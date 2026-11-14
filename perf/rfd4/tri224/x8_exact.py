"""pair_x8's move back on the device: permute_back_pk on a bfp8 x equals torch's permute of x as stored, at the batched
shapes RFD4 runs (224 b10, 192 b14, 256 b4), and the bf16 program is unchanged. `pair_x8t` (bfp8 out, packer-truncated,
blocks along channels) cannot be bit-exact: its rel RMS against the exact move is recorded. Run under `timeout`.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/tri224/x8_exact.py --out <file.json>
"""
import argparse
import json
import statistics
import time

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import permute_back_pk as PK  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="224x10,192x14,256x4")
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timed(f):
    f()
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        f()
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


recs = []
gen = torch.Generator().manual_seed(1)
for spec in a.shapes.split(","):
    n, B = (int(v) for v in spec.split("x"))
    xt = torch.randn(B, 256, n, n, generator=gen) * torch.logspace(-2, 1, 256).reshape(1, 256, 1, 1)
    rec = {"n": n, "batch": B}
    for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        x = ttnn.from_torch(xt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, n, n, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, DRAM)
        PK.permute_back_pk(x, out, grid)
        stored = ttnn.to_torch(x).to(torch.bfloat16)
        rec[f"{name}_exact"] = bool(torch.equal(ttnn.to_torch(out), stored.permute(0, 2, 3, 1)))
        rec[f"{name}_stored_rel_rms"] = float((stored.double() - xt.double()).pow(2).mean().sqrt()
                                              / xt.double().pow(2).mean().sqrt())
        rec[f"{name}_move_ms"] = timed(lambda: PK.permute_back_pk(x, out, grid))
        if name == "bfp8":
            o8 = ttnn.allocate_tensor_on_device(ttnn.Shape([B, n, n, 256]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, DRAM)
            PK.permute_back_pk(x, o8, grid)
            ref = stored.permute(0, 2, 3, 1).double()
            rec["x8t_rel_rms_vs_exact"] = float((ttnn.to_torch(o8).double() - ref).pow(2).mean().sqrt()
                                                / ref.pow(2).mean().sqrt())
            rec["x8t_move_ms"] = timed(lambda: PK.permute_back_pk(x, o8, grid))
            ttnn.deallocate(o8)
        ttnn.deallocate(x)
        ttnn.deallocate(out)
    print(json.dumps(rec), flush=True)
    recs.append(rec)
json.dump(recs, open(a.out, "w"), indent=1)
