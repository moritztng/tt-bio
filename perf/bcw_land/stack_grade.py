#!/usr/bin/env python3
"""The round program's Evoformer VJP against float64, every lever at its shipped default.

`perf/bcx_afgrad` `vjp` under `bindcraft2.fast_round()`: each block teacher-forced (float64 inputs
rounded to bf16, the float64 cotangent at its output), the device VJP compared with the float64 VJP
on the same inputs. Run from the tree being graded (`.base/` or this one), so the two read alike.
    stack_grade.py <tag> [--n 288] [--evo 8] [--blocks 0,3,7]
"""
import argparse, pathlib, sys

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--n", type=int, default=288)
ap.add_argument("--evo", type=int, default=8)
ap.add_argument("--blocks", default="0,3,7")
a = ap.parse_args()
ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from perf.bcx_afgrad import afgrad as A     # noqa: E402
from tt_bio import bindcraft2, triatt_bw as TB
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=0, evo=a.evo,
                        blocks=a.blocks, controls_all=False, controls_only=False, seed=0,
                        msa_mask=False, threads=8, tag=a.tag)
print(f"== tree {ROOT} tag {a.tag}", flush=True)
with bindcraft2.fast_round():
    A.cmd_vjp(ns)
print(f"== triatt_bw {TB.STATS}", flush=True)
