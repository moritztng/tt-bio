"""pair_swiglu_ws with the silu on the PACK thread (PACK_SILU) against the shipped MATH-thread silu: outputs must be
torch.equal, and the time of each. One JSON line per shape.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_packsilu.py [--n 512] [--reps 10]
"""
import argparse, json, os, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

dev = T.get_device()
# (c, hidden of one chunk, weight dtype, fidelity, l1acc): OpenDDE normal / fast chunks, Protenix-v2 / Boltz-2 c_z 128
CASES = ((384, 384, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, True), (384, 512, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True),
         (128, 512, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, False), (128, 512, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False))


def timed(fn):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return round(1e3 * statistics.median(ts), 3)


for c, h, dt, fid, acc in CASES:
    g = torch.Generator().manual_seed(0)
    w1, w2 = (torch.randn(c, h, generator=g) / c ** 0.5 for _ in range(2))
    w3 = torch.randn(h, c, generator=g) / h ** 0.5
    x = ttnn.from_torch(torch.randn(1, a.n, a.n, c, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev)
    parts = W.weights(w1, w2, w3, dev, dt)
    row = dict(n=a.n, c=c, hidden=h, wdtype=str(dt), fidelity=str(fid), l1acc=acc,
               card=os.environ.get("TT_VISIBLE_DEVICES"))
    if len(parts) > 1 or not W.eligible(x, *parts[0], l1acc=acc):
        print(json.dumps(dict(row, note="not one eligible chunk")), flush=True)
        continue
    outs, ms = {}, {}
    for ps in (False, True):
        W.PACK_SILU = ps
        call = lambda: W.pair_swiglu(x, *parts[0], fid, l1acc=acc)  # noqa: E731
        o = call()
        outs[ps] = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
        ms[ps] = timed(call)
    W.PACK_SILU = False
    d = (outs[True] - outs[False]).abs()
    print(json.dumps(dict(row, ms_math=ms[False], ms_pack=ms[True], equal=bool(torch.equal(outs[True], outs[False])),
                          max_abs=float(d.max()), n_diff=int((d > 0).sum()))), flush=True)
    ttnn.deallocate(x)
