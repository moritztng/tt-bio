"""ESMFold2's packed LM keep mask: host-tilized vs chip-tilized int32 upload, and the unpacked mask against the host.

xf4 profile (p300c, -Fast --fast): 11 keep uploads of 16 MB took 0.84 s, ~210 MB/s, the speed of ttnn's host tilize.
Prints, per path, the median upload time and whether the device words and the unpacked bf16 mask equal the host's.

    TT_VISIBLE_DEVICES=<chip> python3 perf/spd_gap_esmf/keep_upload_check.py [--shape 1,512,512,256]
"""
import argparse
import json
import statistics
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import esmfold2_runtime as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="1,512,512,256", help="unpacked keep shape [B, L, L, C]")
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
dev = T.get_device()
torch.manual_seed(0)
keep = torch.empty(tuple(int(x) for x in a.shape.split(","))).bernoulli_(0.75).bool()
w = R._pack_keep(keep)


def timed(fn):
    ts = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(out)
    return statistics.median(ts)


host = lambda: ttnn.from_torch(w, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)
chip = lambda: T.upload(w, dtype=ttnn.int32, device=dev)
words_equal = torch.equal(ttnn.to_torch(chip()), w)
mask = ttnn.to_torch(R._unpack_keep_dev(w, dev))
print(json.dumps({"shape": list(keep.shape), "words_MB": w.numel() * 4 / 2**20,
                  "host_tilize_ms": 1e3 * timed(host), "chip_tilize_ms": 1e3 * timed(chip),
                  "words_equal": words_equal, "mask_equal": torch.equal(mask.bool(), keep)}))
