#!/usr/bin/env python3
"""Grade the gated channel move's tape entry against float64: `perf/bcx_afgrad` `vjp`, by arm.

From `perf/bcp_device/vjp_grade.py`. Each Evoformer block is teacher-forced (float64 inputs
rounded to bf16, the float64 cotangent at its output) and its device VJP compared with the float64
VJP on the same inputs. Under `bindcraft2.fast_round()`, the round's program, with:
  off    the entry not installed (the four-way split, as on main)
  entry  the entry installed, its composed backward
  on     the entry installed, the fused backward (`reblock_permute_gated_bw`)
`--lever gate_bw` grades stage 3 instead: both arms with stage 1 on, `gate_bw` off / on.
"""
import argparse, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402

BASE = "tri_att_sdpa_hifi,rne_add"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=288)
ap.add_argument("--evo", type=int, default=8)
ap.add_argument("--blocks", default="0,3,7")
ap.add_argument("--arms", default=None)
ap.add_argument("--lever", choices=("gated", "gate_bw"), default="gated")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import bindcraft2, gate_bw as GB, reblock_permute as R, taped_ttnn as T
arms = a.arms or ("off,entry,on" if a.lever == "gated" else "off,on")
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=0, evo=a.evo,
                        blocks=a.blocks, controls_all=False, controls_only=False, seed=0,
                        msa_mask=False, threads=8)
with bindcraft2.fast_round():
    for arm in arms.split(","):
        stage1 = a.lever == "gate_bw" or arm != "off"
        os.environ["TT_BIO_TAPED_KERNELS"] = BASE + (",reblock_permute_gated" if stage1 else "")
        R.GATED_BW_FUSED = a.lever == "gate_bw" or arm == "on"
        GB.FUSED = a.lever == "gate_bw" and arm == "on"
        s0 = list(GB.STATS)
        g0, b0 = list(R.STATS_GATED), list(R.STATS_GATED_BW)
        e0 = list(T.KERNEL_STATS.get("reblock_permute_gated", [0, 0]))
        ns.tag = f"bcp_evo_{a.lever}_{arm}"
        print(f"== arm {arm} kernels={os.environ['TT_BIO_TAPED_KERNELS']} "
              f"GATED_BW_FUSED={R.GATED_BW_FUSED}", flush=True)
        A.cmd_vjp(ns)
        e1 = T.KERNEL_STATS.get("reblock_permute_gated", [0, 0])
        print(f"== arm {arm} reach gated_move {[x - y for x, y in zip(R.STATS_GATED, g0)]} "
              f"entry {[x - y for x, y in zip(e1, e0)]} "
              f"gated_bw {[x - y for x, y in zip(R.STATS_GATED_BW, b0)]} "
              f"gate_bw {[x - y for x, y in zip(GB.STATS, s0)]}", flush=True)
