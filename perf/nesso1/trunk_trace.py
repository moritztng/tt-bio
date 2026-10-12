"""Nesso-1 trunk alone: eager pass vs the same pass replayed from a ttnn trace.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=<tree> python perf/nesso1/trunk_trace.py --n 256 532 --report r.json

Real weights, a random z per size. Per size: eager wall per pass (what `Nesso1._trunk` costs
today, upload and download included), then one capture of the 48-block stack on a resident
input and replays of it, timed the same way (upload, execute, download). The replay output is
compared with torch.equal to an eager pass on the same input, so the number only counts if the
trace computes the same thing. Run it under a harness that
samples AICLK during the passes (ng_run.sh / ppl_run.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[256, 532])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.nesso1 import Nesso1

    dev = tt.get_device(trace="nesso1")
    model = Nesso1.from_pretrained(use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    pm = model.pairformer_module
    rep = {"host": socket.gethostname(), "visible": os.environ.get("TT_VISIBLE_DEVICES"),
           "load": os.getloadavg(), "sizes": []}

    for n in a.n:
        torch.manual_seed(n)
        z = torch.randn(1, n, n, 128)
        mask = torch.ones(1, n)
        pair = mask[:, :, None] * mask[:, None, :]
        pm.reset_static_cache()
        eager = []
        for _ in range(a.reps + 1):
            t = time.perf_counter()
            _, ze = pm(None, z, mask=mask, pair_mask=pair)
            eager.append(time.perf_counter() - t)

        # Same padding and masks the eager call just cached.
        pad = int(pm._cache_get("mask_tt").shape[-1]) - n
        zp = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
        z_in = pm._from_torch(zp)
        masks = (pm._cache_get("mask_tt"), pm._cache_get("attn_mask_start_tt"),
                 pm._cache_get("attn_mask_end_tt"))
        # The stack may free its input; the clone keeps z_in alive across replays.
        pm.module(None, ttnn.clone(z_in), *masks)  # warm with the exact capture call
        t = time.perf_counter()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        _, z_out = pm.module(None, ttnn.clone(z_in), *masks)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        capture_s = time.perf_counter() - t

        traced = []
        for _ in range(a.reps + 1):
            t = time.perf_counter()
            host = ttnn.from_torch(zp, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            ttnn.copy_host_to_device_tensor(host, z_in)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            zt = pm._to_torch(z_out)[:, :n, :n, :]
            traced.append(time.perf_counter() - t)
        same = bool(torch.equal(zt, ze))
        maxdiff = float((zt - ze).abs().max())
        ttnn.release_trace(dev, tid)

        row = {"n": n, "pad": pad, "eager_s": [round(x, 4) for x in eager],
               "traced_s": [round(x, 4) for x in traced], "capture_s": round(capture_s, 4),
               "eager_warm_median": round(statistics.median(eager[1:]), 4),
               "traced_median": round(statistics.median(traced[1:]), 4),
               "equal_to_eager": same, "max_abs_diff": maxdiff}
        rep["sizes"].append(row)
        print(json.dumps(row), flush=True)
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
