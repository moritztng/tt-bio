#!/usr/bin/env python3
"""One of3t-vjpln device arm: `of3t-bwdaccum/dev_cot.py` under the WIDENED pin.

    VJP_PIN=""          shipped path, nothing wrapped. The A/A floor.
    VJP_PIN=census      every taped node recorded, nothing substituted. Answers `CENSUS:` and
                        doubles as an inertness control -- its cotangent must be bit-identical
                        to the shipped one.
    VJP_PIN=identity2   the widened selector firing in identity mode: every node round-tripped
                        through host float64 and written back UNCHANGED. Must be bit-exact.
    VJP_PIN=allref2     the arm. Every `_v_matmul`, `_v_softmax`, `triangle_attention`,
                        `_taped_layer_norm` and `_taped_linear` pinned to its float64 VJP.

    VJP_BLOCKS=45,44    restrict the pin to those blocks; empty means every block.
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
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_vjpln"))

PIN = os.environ.get("VJP_PIN", "")
BLOCKS = os.environ.get("VJP_BLOCKS", "")
CHUNK = int(os.environ.get("VJP_CHUNK", "16"))
SIDECAR = os.environ.get("VJP_SIDECAR", "")

import torch                                                          # noqa: E402
import tt_bio                                                         # noqa: E402
import tt_bio.tenstorrent as T                                        # noqa: E402

# D190's trap, paid for once already on this campaign: `python3 perf/<ns>/script.py` puts the
# SCRIPT's directory on sys.path, and the shared env carries an EDITABLE install of tt_bio
# pointed at /home/ttuser/tt-bio-dev, which sits on main. An arm that measures main while
# running from this worktree announces it only in a config field nobody re-reads.
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
        import pinvjp2
        pinvjp2.install_census()
        mode = "census"
    elif PIN:
        import pinvjp2
        blocks = [int(x) for x in BLOCKS.split(",") if x.strip() != ""] or None
        pinvjp2.install(pin=PIN, blocks=blocks, chunk=CHUNK)
        mode = "pin"
    import dev_cot
    t0 = time.perf_counter()
    rc = 1
    try:
        rc = dev_cot.main()
    finally:
        if SIDECAR:
            # D155: a card number without a host is not an identification -- card 0 is a
            # different piece of hardware on pc, qb1 and qb2 -- and an artifact whose value
            # could depend on which card produced it has to carry that card itself, not leave
            # it in a log. `socket.gethostname()` rather than a flag, because a transcribed
            # host is the same failure one level up.
            _short = socket.gethostname().split(".")[0]
            _host = {"tt-quietbox": "qb1", "tt-quietbox2": "qb2"}.get(_short, _short)
            _card = os.environ.get("TT_VISIBLE_DEVICES", "")
            body = {"pin": PIN, "blocks": BLOCKS, "chunk": CHUNK, "rc": rc,
                    "host": _host, "hostname": _short,
                    "card": "%s (%s) card %s, p150a Blackhole" % (_host, _short, _card),
                    "seconds": round(time.perf_counter() - t0, 1),
                    "cast": CAST, "tt_bio": os.path.realpath(tt_bio.__file__),
                    "env": {k: os.environ.get(k, "") for k in
                            ("TT_BIO_SOFTMAX_BW_RENORM", "TT_VISIBLE_DEVICES",
                             "TT_BIO_LEASE_CARDS", "VJP_PIN", "VJP_BLOCKS")},
                    "fp32_softmax_stats": dict(T.FP32_SOFTMAX_STATS),
                    "softmax_bw_renorm_stats": dict(
                        __import__("tt_bio.autograd", fromlist=["x"]).SOFTMAX_BW_RENORM_STATS)}
            if mode == "census":
                import pinvjp2
                body.update(pinvjp2.census_summary())
            elif mode == "pin":
                import pinvjp2
                body.update(pinvjp2.summary())
            json.dump(body, open(SIDECAR, "w"), indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
