#!/usr/bin/env python3
"""One of3t-shapekey device arm: `of3t-bwdaccum/dev_cot.py` with one shape-keyed pick pinned.

Nothing in `tt_bio/` is edited. The pin is installed on the imported module here, so the arm runs
the shipped code with exactly one dispatch decision replaced, and a reach counter says whether the
replacement was ever asked for. D121: a pin that never fires and a pin that fires and is inert are
different results, and no output comparison separates them.

    SHAPEKEY_PIN=none            shipped
    SHAPEKEY_PIN=single_plan64   `_fp32_softmax_l1_plan` returns the width-64 answer (330, 110)
                                 for the single track's shape class at 384, (height_per_row 6144,
                                 width 384). Every other shape class is untouched.
    SHAPEKEY_PIN=pair_plan64     the same for the pair track's class at 384, (1536, 384), pinned to
                                 (0, 0) -- no block and no shard, which is the geometry the 64 case
                                 actually runs.

The pair-track route pin needs no patch: `TT_BIO_TRIATT_SDPA_HIFI_AB=-openfold3.trunk` is the
shipped A/B grammar and `arm.sh` sets it.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))

PIN = os.environ.get("SHAPEKEY_PIN", "none")
SIDECAR = os.environ.get("SHAPEKEY_SIDECAR", "")

import tt_bio.tenstorrent as T  # noqa: E402

REACH = {"pin": PIN, "queried": 0, "replaced": 0, "returned": {}}

# Route census, always on and bit-neutral: it counts and delegates. `FP32_SOFTMAX_STATS["calls"]`
# reads 384 at BOTH widths, which is 96 triangle-attention calls over 4 passes and leaves no room
# for the 48-per-pass AttentionPairBias site. Whether the single track calls the function the brief
# names at all is not something a pure shape lookup can answer, so it is counted here by shape.
CENSUS: dict = {}


def _census(name, shape):
    k = f"{name} {list(shape)}"
    CENSUS[k] = CENSUS.get(k, 0) + 1


_real_fp32_attn = T._fp32_softmax_attention


def _counted_fp32_attn(q, *a, **kw):
    _census("_fp32_softmax_attention q", q.shape)
    return _real_fp32_attn(q, *a, **kw)


T._fp32_softmax_attention = _counted_fp32_attn

_real_sdpa_masked = T._sdpa_masked


def _counted_sdpa_masked(fn, q, *a, **kw):
    _census(f"_sdpa_masked site={kw.get('site', '?')} q", q.shape)
    return _real_sdpa_masked(fn, q, *a, **kw)


T._sdpa_masked = _counted_sdpa_masked

# The shape classes, as (height_per_row, width) -> the answer the width-64 case produces.
#   single track at 384 is (16 heads * 384, k_len 384); the 64 case answers (330, 110)
#   pair track   at 384 is ( 4 heads * 384, k_len 384); the 64 case answers no blocking and no
#     shard at all, because at 64 `run()` takes the unblocked branch and `_fp32_softmax_shard`
#     refuses (16384 % 3520). (0, 0) reproduces that: no plan, so `blk` stays at the byte budget
#     3616, which is above the 384 leading rows, so ONE unblocked interleaved call.
PINS = {
    "single_plan64": {(6144, 384): (330, 110)},
    "pair_plan64": {(1536, 384): (0, 0)},
}

if PIN in PINS:
    _table = PINS[PIN]
    _real_plan = T._fp32_softmax_l1_plan

    def _pinned_plan(per_row, height_per_row, width, cap=None, bmm=None, free_cap=None):
        shipped = _real_plan(per_row, height_per_row, width, cap, bmm, free_cap)
        REACH["queried"] += 1
        key = f"{height_per_row}x{width}"
        want = _table.get((height_per_row, width))
        if want is not None:
            REACH["replaced"] += 1
            REACH["returned"][key] = {"shipped": list(shipped), "pinned": list(want)}
            return want
        REACH["returned"].setdefault(key, {"shipped": list(shipped), "pinned": None})
        return shipped

    T._fp32_softmax_l1_plan = _pinned_plan
elif PIN != "none":
    raise SystemExit(f"unknown SHAPEKEY_PIN {PIN!r}")


def _sidecar(rc, seconds):
    if not SIDECAR:
        return
    picks = {f"{q}x{k}": v for (q, k), v in T.TRIATT_FUSED_HIFI_PICKS.items()}
    json.dump({
        "pin": PIN,
        "rc": rc,
        "seconds": round(seconds, 1),
        "pin_reach": REACH,
        "route_census": dict(sorted(CENSUS.items())),
        "env": {k: os.environ.get(k, "")
                for k in ("TT_BIO_TRIATT_SDPA_HIFI_AB", "TT_BIO_SOFTMAX_BW_RENORM",
                          "TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS")},
        "site_flags_triatt_hifi_on": T.site_flags_on("TT_BIO_TRIATT_SDPA_HIFI_AB"),
        "triatt_fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
        "triatt_fused_hifi_picks_q_k_kvbf": picks,
        "fp32_softmax_stats": dict(T.FP32_SOFTMAX_STATS),
    }, open(SIDECAR, "w"), indent=1)


def main():
    import dev_cot
    t0 = time.perf_counter()
    rc = 1
    try:
        rc = dev_cot.main()
    finally:
        _sidecar(rc, time.perf_counter() - t0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
