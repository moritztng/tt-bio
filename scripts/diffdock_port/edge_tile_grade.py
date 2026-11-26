"""Grade the fused edge-tile kernel against the float64 host tensor product, and time it beside
the GEMM form (`device.DeviceEdgeTP`, fp32 and bf16) on the same card in the same process.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:dfd-perf \
        timeout 900 python scripts/diffdock_port/edge_tile_grade.py --out et.json

Inputs and weights as `device_tp_grade.py` (real v1.1 FC weights per signature). AICLK is
sampled during the whole run.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from device_tp_grade import CKPT, SIGNATURES, grade, inputs, layer  # noqa: E402
from tt_bio.diffdock import config as dcfg  # noqa: E402
from tt_bio.diffdock import device as ddev  # noqa: E402
from tt_bio.diffdock.edge_tile import DeviceEdgeTile, emulate  # noqa: E402


def timed(fn, dev, ttnn, reps):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--layers", default=",".join(SIGNATURES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.diffdock.capture import clock_summary, sample_aiclk

    cfg = dcfg.load(CKPT / "score_model", old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    gen = torch.Generator().manual_seed(0)
    E = ddev.pad32(args.edges)
    dev = get_device()
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_aiclk,
                          args=(stop, samples, os.environ.get("TT_VISIBLE_DEVICES", "0")),
                          daemon=True)
    th.start()

    rows = {}
    for name in args.layers.split(","):
        tp, fc = layer(cfg, sd, name)
        ea, x1, x2 = inputs(tp, fc, E, gen)
        try:
            mod = DeviceEdgeTile(dev, tp, fc)
        except NotImplementedError as e:
            rows[name] = dict(skipped=str(e))
            print(name, "skipped:", e, flush=True)
            continue
        with torch.no_grad():
            ref = tp(x1, x2, fc(ea))
        row = dict(edges=E, emulate_fp64=grade(ref, emulate(mod.plan, ea, x1, x2)),
                   groups=[g.terms for g in mod.plan.groups])
        ins = [ddev.to_device(dev, t, ttnn.bfloat16) for t in (ea, x1, x2)]
        got = ttnn.to_torch(mod(*ins)).double()[:E] @ mod.plan.perm
        row["card"] = grade(ref, got)
        row["ms_fused"] = timed(lambda: mod(*ins), dev, ttnn, args.reps)
        if not args.no_baseline:
            for dtn in ("float32", "bfloat16"):
                dt = getattr(ttnn, dtn)
                base = ddev.DeviceEdgeTP(dev, tp, fc.float(), dtype=dt)
                bins = [ddev.to_device(dev, t, dt) for t in (ea, x1, x2)]
                row[f"ms_gemm_{dtn}"] = timed(lambda: base(*bins), dev, ttnn, args.reps)
                fc.double()
        rows[name] = row
        print(name, json.dumps(row), flush=True)
    stop.set()
    th.join(timeout=30)
    res = dict(edges=E, rows=rows, chip=os.environ.get("TT_VISIBLE_DEVICES"))
    res.update(clock_summary(samples))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
