"""Where a trunk call's time goes at width 1, 2 and 4: host prep + upload, device, readback.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/batch/callsplit.py --weights <bundle> --out <json>

One network call (three recycles, as a step at depth 3) on random inputs of the design's shapes,
synchronized between phases, median of five after a warm-up per width. Per-design ms that rises
with width in a phase names where batching loses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="1,2,4")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()

    import ttnn

    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.trunk import download, upload

    eng = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, tr = eng._device, eng.trunk
    cfg = eng.config["nn"]
    D, P, C, n = cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"], a.n
    rows = []
    for B in (int(w) for w in a.widths.split(",")):
        g = torch.Generator().manual_seed(0)
        s0, z0 = torch.randn(B, n, D, generator=g), torch.randn(B, n, n, P, generator=g)
        cs0 = [torch.randn(B, n, C, generator=g) for _ in range(a.depth)]
        masks = tr.masks_for(torch.ones(B, n), dev)
        phases = {"upload": [], "device": [], "readback": []}
        for rep in range(6):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            s = upload(s0.contiguous(), dtype=tr.residual_dtype, device=dev)
            z = upload(z0.contiguous(), dtype=tr.residual_dtype, device=dev)
            cs = [upload(c.contiguous(), dtype=tr.dtype, device=dev) for c in cs0]
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            s_out, z_out = tr(s, z, cs, masks)
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            download(s_out)
            t3 = time.perf_counter()
            for t in (s, z, s_out, z_out, *cs):
                if t.is_allocated():
                    ttnn.deallocate(t)
            if rep:  # the first call at a width warms it
                for k, dt in zip(phases, (t1 - t0, t2 - t1, t3 - t2)):
                    phases[k].append(dt * 1e3)
        row = {"batch": B, **{f"{k}_ms": round(sorted(v)[len(v) // 2], 2) for k, v in phases.items()}}
        row.update({f"{k}_ms_per_design": round(row[f"{k}_ms"] / B, 2) for k in phases})
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps({"n": n, "depth": a.depth, "rows": rows}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
