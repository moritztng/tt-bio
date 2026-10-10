#!/usr/bin/env python3
"""Grade the query-chunked `triatt_bw` backward against float64 on a Wormhole chip.

`serving_plan` serves the fused triangle-attention backward only where the whole-query plan fits
L1. On Wormhole that is 288 tokens, so above it every call declines and the round falls back to the
chunked recompute: a BindCraft 2 ladder measured 432 of 432 calls served at 224 and 0 of 432 at
480, 512 and 576. Blackhole takes the query-chunked plan at those sizes and serves, and the chunk
fits Wormhole's own CB budget at every bucket to 1024. What has never existed is a float64 grade of
that loop on a Wormhole chip. This is that grade; it passed at 288, 480, 512 and 576 and the
Wormhole gate was removed.

Three arms per size, each a teacher-forced Evoformer VJP against the float64 reference
(`perf/bcx_afgrad`), everything else held at the shipped Wormhole fast round:
  fallback  the fused backward off: the chunked recompute, what Wormhole runs today
  chunked   the fused backward on, query-chunked above 288: what ships
A size where the whole-query form already fits (288) is run as a control: there the flag must not
change the plan, so the two arms differ only by the fused kernel itself and the pair is a read on
the harness rather than on the chunk.

The grade is `fallback` vs `chunked`, both against float64. The chunked loop passes if it is no
further from float64 than the path it replaces.
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=480)
ap.add_argument("--evo", type=int, default=8)
ap.add_argument("--blocks", default="0,3,7")
ap.add_argument("--arms", default="fallback,chunked")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--params", default=A.DEFAULT_PARAMS)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor      # noqa: E402
ensure_p300_mesh_descriptor()
from tt_bio import bindcraft2, triatt_bw as TB           # noqa: E402

ns = argparse.Namespace(params=a.params, card=0, n=a.n, extra=0, evo=a.evo,
                        blocks=a.blocks, controls_all=False, controls_only=False, seed=a.seed,
                        msa_mask=False, threads=8)

# The round arms the Wormhole kernel set exactly as `bindcraft2.fast_round()` ships it, so the only
# thing moving between arms is the fused backward and its chunk.
with bindcraft2.fast_round():
    print(f"== plan n={a.n} whole_fits_wh="
          f"{TB.fits_l1(TB.plan(a.n, 4, a.n, 32, (8, 8)), True)} "
          f"chunked_Qt={(TB.largest_fitting_q_chunk(a.n, 4, a.n, 32, (8, 8), True) or {}).get('Qt')}",
          flush=True)
    for arm in a.arms.split(","):
        TB.FUSED = arm == "chunked"
        before = dict(TB.STATS)
        ns.tag = f"whchunk_{a.n}_{arm}"
        print(f"== arm {arm} n={a.n} FUSED={TB.FUSED}", flush=True)
        A.cmd_vjp(ns)
        print(f"== arm {arm} n={a.n} reach triatt_bw "
              f"{ {k: v - before.get(k, 0) for k, v in TB.STATS.items()} }", flush=True)
