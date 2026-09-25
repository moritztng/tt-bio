#!/usr/bin/env python3
"""One of3t-tapeattn device arm: `of3t-bwdaccum/dev_cot.py`, optionally with the verb census.

    TAPEATTN_CENSUS=0   shipped path, nothing wrapped. This is the arm the cotangent ladder
                        and the A/A floor are read on.
    TAPEATTN_CENSUS=1   `tapecensus.install()`: every taped verb counted from inside
                        `autograd._tape`, and every single-track node's backward differenced
                        against its own float64 vector-Jacobian product.

The census arm must produce a BIT-IDENTICAL cotangent file to the shipped arm at the same width,
and `arm.sh` checks that by sha256. That is the inertness proof the brief asks for: the wrapper
counts, snapshots and pins, and pinning moves bytes between L1 and DRAM without moving a number.

Device gradients are at most fp32, so the cotangent payload is stored as float32 here. The cast is
CHECKED elementwise, not asserted: `float32_cast_lossy` counts any tensor the round trip moved, and
it has to read 0. At padded 384 the float64 form is 151 MB per rung per pair track and the box has
61 GB free.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_tapeattn"))

CENSUS = os.environ.get("TAPEATTN_CENSUS", "0") == "1"
HEADS = int(os.environ.get("TAPEATTN_HEADS", "16"))
SIDECAR = os.environ.get("TAPEATTN_SIDECAR", "")

import torch                                                          # noqa: E402
import tt_bio.tenstorrent as T                                        # noqa: E402

CAST = {"tensors": 0, "float32_cast_lossy": 0}
_real_save = torch.save


def _shrink(o):
    if isinstance(o, torch.Tensor):
        CAST["tensors"] += 1
        if o.dtype == torch.float64:
            f = o.float()
            if not torch.equal(f.double(), o):
                CAST["float32_cast_lossy"] += 1
                return o
            return f
        return o
    if isinstance(o, dict):
        return {k: _shrink(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_shrink(v) for v in o)
    return o


def _save(obj, path, *a, **kw):
    if isinstance(obj, dict) and "cot" in obj:
        obj = dict(obj)
        obj["cot"] = _shrink(obj["cot"])
        obj["stored_dtype"] = "float32, checked lossless against the float64 device read"
    return _real_save(obj, path, *a, **kw)


def main():
    torch.save = _save
    census = None
    if CENSUS:
        import tapecensus
        census = tapecensus.install(heads=HEADS, capture=True)
    import dev_cot
    t0 = time.perf_counter()
    rc = 1
    try:
        rc = dev_cot.main()
    finally:
        if SIDECAR:
            import tapecensus as tc
            body = {
                "census": CENSUS, "heads": HEADS, "rc": rc,
                "seconds": round(time.perf_counter() - t0, 1),
                "cast": CAST,
                "env": {k: os.environ.get(k, "") for k in
                        ("TT_BIO_SOFTMAX_BW_RENORM", "TT_BIO_TRIATT_SDPA_HIFI_AB",
                         "TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TAPEATTN_CENSUS")},
                "fp32_softmax_stats": dict(T.FP32_SOFTMAX_STATS),
                "triatt_fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
                "softmax_bw_renorm_stats": dict(
                    __import__("tt_bio.autograd", fromlist=["x"]).SOFTMAX_BW_RENORM_STATS),
            }
            if census is not None:
                body.update(tc.summary())
            json.dump(body, open(SIDECAR, "w"), indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
