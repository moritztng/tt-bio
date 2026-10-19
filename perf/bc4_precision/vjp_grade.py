#!/usr/bin/env python3
"""Grade bfp8 storage placements on the BC2 design round against float64, block by block.

`perf/bcx_afgrad/afgrad.py vjp` per arm, under `bindcraft2.fast_round()` (the round BindCraft 2 runs):
each Evoformer block teacher-forced (float64 inputs rounded to bf16, the float64 cotangent at its
output), its device VJP compared with the float64 VJP on the same inputs, beside torch bf16's own
distance from the same reference. An arm is a set of module flags, each a bfp8 storage placement:

  off     main
  b8in    the trimul in-projection written bfp8 (`tenstorrent._TRIMUL_INPROJ_B8`)
  tri8    triangle attention's q/k/v/gate interior bfp8 (`tenstorrent._TRIATT_B8`)
  both    b8in + tri8
  b8b     b8in with AF2's in-projection bias allowed (`_TRIMUL_INPROJ_B8_BIAS`), the form that fires on AF2

    TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python perf/bc4_precision/vjp_grade.py --arms off,b8in,tri8,both
"""
import argparse, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402

ARMS = {"off": {}, "b8in": {"_TRIMUL_INPROJ_B8": True}, "tri8": {"_TRIATT_B8": True},
        "both": {"_TRIMUL_INPROJ_B8": True, "_TRIATT_B8": True},
        "b8b": {"_TRIMUL_INPROJ_B8": True, "_TRIMUL_INPROJ_B8_BIAS": True}}
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--evo", type=int, default=8)
ap.add_argument("--blocks", default="0,3,7")
ap.add_argument("--arms", default="off,b8b")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import bindcraft2, tenstorrent as TN
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=0, evo=a.evo, blocks=a.blocks,
                        controls_all=False, controls_only=False, seed=a.seed, msa_mask=False, threads=8)
with bindcraft2.fast_round():
    for arm in a.arms.split(","):
        flags = {k: False for f in ARMS.values() for k in f}
        flags.update(ARMS[arm])
        for k, v in flags.items():
            setattr(TN, k, v)
        ns.tag = f"bc4p_{arm}_s{a.seed}"
        print(f"== arm {arm} {flags}", flush=True)
        A.cmd_vjp(ns)
