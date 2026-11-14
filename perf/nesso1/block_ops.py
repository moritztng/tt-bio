"""Where one Nesso-1 pairformer block spends its device time, per sub-op, at each size.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=<tree> python perf/nesso1/block_ops.py --n 256 532 --report r.json

Real weights (block 0), random z, under the model's lever set (TT_BIO_LEVERS, else LEVERS_NORMAL).
The sub-ops are called outside the model's own lever scope, so this script enters it itself. Each
sub-op is timed alone between two synchronize_device calls (on a clone of z, since some free their
input; `clone` is that cost alone), median of --reps after a warm call, on the inputs the trunk gives it (padded z, the
cached masks). The whole block is timed both ways the stack can run it: with the D174
transition mask (what the trunk does today, since its mask is 3-D) and without it (the fused
add_to_input path), so the cost of each path is a measurement, not an assumption.
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
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.nesso1 import Nesso1

    dev = tt.get_device()
    model = Nesso1.from_pretrained(use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    pm = model.pairformer_module
    rep = {"host": socket.gethostname(), "visible": os.environ.get("TT_VISIBLE_DEVICES"),
           "levers": sorted(model._levers), "load": os.getloadavg(), "sizes": []}

    def timed(fn, *args, **kw):
        fn(*args, **kw)  # warm: compile + program cache
        ts = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            out = fn(*args, **kw)
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t)
            for o in (out if isinstance(out, tuple) else (out,)):
                if isinstance(o, ttnn.Tensor):
                    ttnn.deallocate(o)
        return round(statistics.median(ts) * 1e3, 3)

    with tt.levers(model._levers):
        for n in a.n:
            torch.manual_seed(n)
            z = torch.randn(1, n, n, 128)
            mask = torch.ones(1, n)
            pm.reset_static_cache()
            pm(None, z, mask=mask, pair_mask=mask[:, :, None] * mask[:, None, :])  # builds the masks
            m, ms, me = (pm._cache_get(k) for k in ("mask_tt", "attn_mask_start_tt", "attn_mask_end_tt"))
            pad = int(m.shape[-1]) - n
            zt = pm._from_torch(torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad)))
            blk = pm.module.blocks[0]
            tmz = ttnn.unsqueeze(m, -1)
            row = {"n": n, "pad": pad, "ms": {
                "tri_mul_out": timed(lambda: blk.triangle_multiplication_start(ttnn.clone(zt), m)),
                "tri_mul_in": timed(lambda: blk.triangle_multiplication_end(ttnn.clone(zt), m)),
                "tri_att_start": timed(lambda: blk.triangle_attention_start(ttnn.clone(zt), ms)),
                "tri_att_end": timed(lambda: blk.triangle_attention_end(ttnn.clone(zt), me)),
                "transition": timed(lambda: blk.transition_z(ttnn.clone(zt))),
                "transition_masked": timed(lambda: blk.transition_z(ttnn.clone(zt), mask=tmz)),
                "residual_add": timed(lambda: ttnn.add(ttnn.clone(zt), zt)),
                "block_d174": timed(lambda: blk(None, ttnn.clone(zt), m, ms, me, None, tmz, None)),
                "block_fused": timed(lambda: blk(None, ttnn.clone(zt), m, ms, me)),
                "clone": timed(ttnn.clone, zt),
            }}
            rep["sizes"].append(row)
            print(json.dumps(row), flush=True)
            ttnn.deallocate(tmz)
            ttnn.deallocate(zt)
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
