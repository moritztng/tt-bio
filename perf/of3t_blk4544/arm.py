#!/usr/bin/env python3
"""One of3t-blk4544 device arm: `of3t-bwdaccum/dev_cot.py`, optionally with a verb pinned.

    BLK_PIN=""            shipped path, nothing wrapped. The ladder baseline and the A/A floor.
    BLK_PIN=identity      the pinning wrapper installed and firing, writing the device's OWN
                          values back. Must reproduce the shipped cotangent byte for byte.
    BLK_PIN=sm16          the single track's softmax backward pinned to its float64 VJP.
    BLK_PIN=triatt        the pair track's TriangleAttention backward pinned.
    BLK_PIN=paircontract  TriangleMultiplication's contraction pinned.

    BLK_BLOCKS=45,44      restrict the pin to those blocks; empty means every block.

The cotangent payload is stored as float32, and the cast is CHECKED elementwise rather than
asserted: `float32_cast_lossy` must read 0.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_blk4544"))

PIN = os.environ.get("BLK_PIN", "")
BLOCKS = os.environ.get("BLK_BLOCKS", "")
CHUNK = int(os.environ.get("BLK_CHUNK", "16"))
SIDECAR = os.environ.get("BLK_SIDECAR", "")

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
    pin = None
    if PIN:
        import pinvjp
        blocks = [int(x) for x in BLOCKS.split(",") if x.strip() != ""] or None
        pin = pinvjp.install(pin=PIN, blocks=blocks, chunk=CHUNK)
    import dev_cot
    t0 = time.perf_counter()
    rc = 1
    try:
        rc = dev_cot.main()
    finally:
        if SIDECAR:
            body = {"pin": PIN, "blocks": BLOCKS, "chunk": CHUNK, "rc": rc,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "cast": CAST,
                    "env": {k: os.environ.get(k, "") for k in
                            ("TT_BIO_SOFTMAX_BW_RENORM", "TT_VISIBLE_DEVICES",
                             "TT_BIO_LEASE_CARDS", "BLK_PIN", "BLK_BLOCKS")},
                    "fp32_softmax_stats": dict(T.FP32_SOFTMAX_STATS),
                    "softmax_bw_renorm_stats": dict(
                        __import__("tt_bio.autograd", fromlist=["x"]).SOFTMAX_BW_RENORM_STATS)}
            if pin is not None:
                import pinvjp as pv
                body.update(pv.summary())
            json.dump(body, open(SIDECAR, "w"), indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
