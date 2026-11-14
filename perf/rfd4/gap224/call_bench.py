"""Wall time of one RFD4 trunk call on the engine's own trunk, eager and traced, under whatever ttnn is importable.

    TT_VISIBLE_DEVICES=<card> TT_BIO_TRACE_REGION_MB=2048 python perf/rfd4/gap224/call_bench.py \
        --n 224 --batch 5 --out <file.json> [--cycles 3] [--reps 10]

The call is exactly the one `perf/rfd4/e2e/elt_prof.py` profiles for the census (DesignEngine's trunk, all-ones
mask, fp32 s and z uploaded), so the same script run on the wheel and on the Tracy source build with the profiler
off says whether the census's kernel time and the timed call disagree because of the build or because of the call.
Uploads are outside the timed region. AICLK is sampled for the life of the run.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ttuser/rfd4-bundle")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import engine as E

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, trunk = engine._device, engine.trunk
    n, b = a.n, a.batch
    g = torch.Generator().manual_seed(0)
    tok, pair = int(trunk.blocks[0].mhba.mha.node_norm_w.shape[-1]), int(trunk.z_proj.shape[-1])
    cond = int(trunk.blocks[0].mhba.adaln.gamma_w.shape[0])
    masks = trunk.masks_for(torch.ones(b, n), dev)
    s = T.upload(torch.randn(b, n, tok, generator=g), dtype=ttnn.float32, device=dev)
    z = T.upload(torch.randn(b, n, n, pair, generator=g), dtype=ttnn.float32, device=dev)
    cs = [T.upload(torch.randn(b, n, cond, generator=g), dtype=ttnn.bfloat16, device=dev) for _ in range(a.cycles)]
    sync = lambda: ttnn.synchronize_device(dev)  # noqa: E731

    def call():
        return trunk(s, z, cs, masks)

    def free(ts):
        for t in ts:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    rec = {"n": n, "batch": b, "cycles": a.cycles, "ttnn": str(Path(ttnn.__file__).resolve().parent),
           "tt_bio": str(Path(T.__file__).resolve().parent), "card": os.environ.get("TT_VISIBLE_DEVICES")}
    with clocksample.during(period=0.25) as clk:
        free(call())
        free(call())
        sync()
        ms = []
        for _ in range(a.reps):
            sync()
            t = time.perf_counter()
            out = call()
            sync()
            ms.append(1e3 * (time.perf_counter() - t))
            free(out)
        rec["eager_ms"] = {"median": round(statistics.median(ms), 3), "min": round(min(ms), 3), "all": [round(x, 2) for x in ms]}
        tid, res = T.capture_trace(dev, call)
        sync()
        ms = []
        for _ in range(a.reps):
            t = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ms.append(1e3 * (time.perf_counter() - t))
        ttnn.release_trace(dev, tid)
        free(res)
        rec["traced_ms"] = {"median": round(statistics.median(ms), 3), "min": round(min(ms), 3), "all": [round(x, 2) for x in ms]}
    rec["aiclk"] = clk.summary()
    rec["load1"] = os.getloadavg()[0]
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
