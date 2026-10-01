#!/usr/bin/env python3
"""Grade TT_BIO_COTANGENT_B8 against float64: `perf/bcx_afgrad` `vjp`, lever off then on.

Same teacher-forced block VJP as `perf/bcp_device/vjp_grade.py` (float64 inputs rounded to bf16,
the float64 cotangent at each block output, torch bf16 and fp32 arms as the envelope), under
`bindcraft2.fast_round()`, with `autograd.COTANGENT_B8` flipped between the passes.
"""
import argparse, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--evo", type=int, default=8)
ap.add_argument("--blocks", default="0,3,7")
ap.add_argument("--arms", default="off,on")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import autograd as ag, bindcraft2
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=0, evo=a.evo,
                        blocks=a.blocks, controls_all=False, controls_only=False, seed=0,
                        msa_mask=False, threads=8)
with bindcraft2.fast_round():
    for arm in a.arms.split(","):
        ag.COTANGENT_B8 = arm == "on"
        ns.tag = f"bcw_ct8_{arm}_n{a.n}"
        print(f"== arm {arm} COTANGENT_B8={ag.COTANGENT_B8}", flush=True)
        A.cmd_vjp(ns)
