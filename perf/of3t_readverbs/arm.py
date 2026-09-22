#!/usr/bin/env python3
"""One of3t-readverbs device arm: `of3t-bwdaccum/dev_cot.py` under a route pin.

    RV_PIN=""           shipped path, nothing wrapped. The A/A floor and the baseline.
    RV_PIN=census       every taped node recorded per BLOCK, nothing substituted, plus a capped
                        float64 scoring of the route verbs' own contributions in RV_ERRBLOCKS.
                        Doubles as an inertness control: its cotangent must be bit-identical.
    RV_PIN=identity3    the route selector in identity mode. Must be bit-exact.
    RV_PIN=route        `_identity_grad` and `_sliced` pinned to their float64 VJPs, written
                        back in the parent's own dtype. The protocol `allref`/`allref2` used.
    RV_PIN=routef32     the same, written back in float32. The arm that removes the rounding.
    RV_PIN=nocast       `_identity_grad` without its narrowing typecast: the exact VJP on
                        device, at every block, no host round trip.

    RV_BLOCKS=44,4,0    restrict a pin to those blocks; empty means every block.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_blk4544"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_readverbs"))

PIN = os.environ.get("RV_PIN", "")
BLOCKS = os.environ.get("RV_BLOCKS", "")
ERRBLOCKS = os.environ.get("RV_ERRBLOCKS", "44,4,0")
CAP = int(os.environ.get("RV_CAP", "2"))
SIDECAR = os.environ.get("RV_SIDECAR", "")

import torch                                                          # noqa: E402
import tt_bio                                                         # noqa: E402
import tt_bio.tenstorrent as T                                        # noqa: E402

# D190: `python3 perf/<ns>/script.py` puts the SCRIPT's directory on sys.path and the shared env
# carries an editable tt_bio pointed at /home/ttuser/tt-bio-dev, which sits on main.
_WT = os.path.realpath(os.getcwd())
if not os.path.realpath(tt_bio.__file__).startswith(_WT):
    raise SystemExit("tt_bio resolved to %s, not this worktree %s" % (tt_bio.__file__, _WT))

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


def main() -> int:
    torch.save = _save
    mode = None
    if PIN == "census":
        import pinroute
        pinroute.install_census(
            err_blocks=tuple(int(x) for x in ERRBLOCKS.split(",") if x.strip() != ""),
            cap=CAP)
        mode = "census"
    elif PIN in ("nocast", "double"):
        import pinroute
        pinroute.install_nocast(mode=PIN)
        pinroute.install_census(err_blocks=(), cap=0)
        mode = "census"
    elif PIN:
        import pinroute
        blocks = [int(x) for x in BLOCKS.split(",") if x.strip() != ""] or None
        pinroute.install(pin=PIN, blocks=blocks)
        mode = "pin"
    import dev_cot
    t0 = time.perf_counter()
    rc = 1
    try:
        rc = dev_cot.main()
    finally:
        if SIDECAR:
            _short = socket.gethostname().split(".")[0]
            _host = {"tt-quietbox": "qb1", "tt-quietbox2": "qb2"}.get(_short, _short)
            _card = os.environ.get("TT_VISIBLE_DEVICES", "")
            body = {"pin": PIN, "blocks": BLOCKS, "err_blocks": ERRBLOCKS, "cap": CAP, "rc": rc,
                    "host": _host, "hostname": _short,
                    "card": "%s (%s) card %s, p150a Blackhole" % (_host, _short, _card),
                    "seconds": round(time.perf_counter() - t0, 1),
                    "cast": CAST, "tt_bio": os.path.realpath(tt_bio.__file__),
                    "env": {k: os.environ.get(k, "") for k in
                            ("TT_BIO_SOFTMAX_BW_RENORM", "TT_VISIBLE_DEVICES",
                             "TT_BIO_LEASE_CARDS", "RV_PIN", "RV_BLOCKS", "RV_ERRBLOCKS")},
                    "fp32_softmax_stats": dict(T.FP32_SOFTMAX_STATS),
                    "softmax_bw_renorm_stats": dict(
                        __import__("tt_bio.autograd", fromlist=["x"]).SOFTMAX_BW_RENORM_STATS)}
            import pinroute
            if mode == "census":
                body.update(pinroute.census_summary())
            elif mode == "pin":
                body.update(pinroute.summary())
            json.dump(body, open(SIDECAR, "w"), indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
