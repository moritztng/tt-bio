#!/usr/bin/env python3
"""Count which dtype branch each layer-norm backward takes in a BindCraft 2 round.

    count_ln.py --out DIR [duo_round.py args...]

Wraps `tt_bio.autograd._layer_norm_bw` so each backward call records (x, g, gamma) dtypes and K,
then runs `perf/bcx_p10_duotraj/duo_round.py` unchanged. Writes DIR/ln_bw_counts.json when the
rounds finish. An instrument, not a timed arm: the wrapper is a dict increment per call.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import autograd as ag  # noqa: E402

COUNTS = collections.Counter()
_orig = ag._layer_norm_bw


def _counting(x, gamma, beta, eps, bwcfg):
    bw = _orig(x, gamma, beta, eps, bwcfg)

    def wrapped(g):
        xd, gd = x.value.dtype, g.dtype
        gmd = None if gamma is None else gamma.value.dtype
        fp32 = ttnn.float32 in (xd, gd)
        casts = sum(d is not None and fp32 and d != ttnn.float32 for d in (xd, gd, gmd))
        COUNTS[f"x={xd} g={gd} gamma={gmd} K={int(x.value.shape[-1])} "
               f"branch={'fp32' if fp32 else 'native'} casts={casts}"] += 1
        return bw(g)
    return wrapped


ag._layer_norm_bw = _counting

from perf.bcx_p10_duotraj import duo_round  # noqa: E402

out = sys.argv[sys.argv.index("--out") + 1]
try:
    duo_round.main()
finally:
    tot = sum(COUNTS.values())
    f32 = sum(v for k, v in COUNTS.items() if "branch=fp32" in k)
    casts = sum(v * int(k.rsplit("casts=", 1)[1]) for k, v in COUNTS.items())
    pathlib.Path(out, "ln_bw_counts.json").write_text(json.dumps(
        {"calls": tot, "fp32_branch": f32, "typecasts": casts,
         "row_mean_stats": getattr(ag, "ROW_MEAN_STATS", None),
         "by_key": dict(COUNTS.most_common())}, indent=1, default=str))
    print(f"ln_bw: {f32}/{tot} calls took the fp32 branch, {casts} typecasts", flush=True)
