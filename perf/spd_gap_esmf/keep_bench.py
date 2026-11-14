#!/usr/bin/env python3
"""ESMFold2's per-loop LM dropout on the device at the page shape: the bit-packed keep mask (`_pack_keep`, 16 MB)
against the bf16 mask (128 MB). Same draw both arms; asserts the masked pairs are torch.equal and times each
arm end to end (host pack + upload + device unpack + multiply, synchronized), arms alternating.

    python perf/spd_gap_esmf/keep_bench.py OUT [--n 512] [--reps 6]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--reps", type=int, default=6)
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmfold2_runtime as R

dev = T.get_device()
g = torch.Generator().manual_seed(0)
z = torch.randn(1, A.n, A.n, A.cz, generator=g).to(torch.bfloat16)
zt = R._DevPair(ttnn.from_torch(z, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), A.n)
keep = torch.rand(z.shape, generator=g) >= 0.1
ref = z.float() * keep.float() / 0.9
times = {"packed": [], "bf16": []}
res = {"n": A.n, "cz": A.cz, "arch": str(dev.arch())}
for rep in range(A.reps + 1):
    for arm in ("packed", "bf16"):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        k = R._pack_keep(keep) if arm == "packed" else keep.to(torch.bfloat16)
        out = R._lm_dropout(zt, 0.1, k)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        o = ttnn.to_torch(out.t)
        ttnn.deallocate(out.t)
        if rep == 0:
            res[arm + "_max_abs_vs_ref"] = float((o.float() - ref).abs().max())
            res[arm + "_out"] = o
        else:
            times[arm].append(dt)
res["equal"] = bool(torch.equal(res.pop("packed_out"), res.pop("bf16_out")))
for arm, v in times.items():
    res[arm + "_ms"] = round(1e3 * statistics.median(v), 2)
    res[arm + "_all_ms"] = [round(1e3 * x, 2) for x in v]
print(json.dumps(res), flush=True)
(OUT / "result.json").write_text(json.dumps(res, indent=1))
