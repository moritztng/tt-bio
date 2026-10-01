#!/usr/bin/env python3
"""Grade the gated channel move's tape entry against float64: `perf/bcx_afgrad` `vjp`, by arm.

From `perf/bcp_device/vjp_grade.py`. Each Evoformer block is teacher-forced (float64 inputs
rounded to bf16, the float64 cotangent at its output) and its device VJP compared with the float64
VJP on the same inputs. Under `bindcraft2.fast_round()`, the round's program, with:
  off    the entry not installed (the four-way split, as on main)
  entry  the entry installed, its composed backward
  on     the entry installed, the fused backward (`reblock_permute_gated_bw`)
`--lever gate_bw|g_bias|lead_sum` grades a later stage instead: arms off / on with every earlier
stage on and every later one off (stage order: LATER below).
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
LATER = ("gate_bw", "g_bias", "lead_sum", "qkv_packed", "fanin_cast", "pair_transpose",
         "gated_packed", "exp21f", "pair_mm")
ap.add_argument("--lever", choices=("gated",) + LATER, default="gated")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import af2, bindcraft2, gate_bw as GB, lead_sum as LS, reblock_permute as R
from tt_bio import autograd as AG, pair_transpose as PT, triatt_bw as TB
from tt_bio import taped_ttnn as T, tenstorrent as TN, pair_mm as PM
SWITCH = {"gate_bw": (GB, "GATE_BW_FUSED"), "g_bias": (af2.AF2PairBlock, "tri_att_g_in_matmul"),
          "lead_sum": (LS, "LEAD_SUM_FUSED"), "qkv_packed": (TB, "QKV_PACKED"),
          "fanin_cast": (AG, "FANIN_CAST_FUSED"), "pair_transpose": (PT, "PAIR_TRANSPOSE_FUSED"),
          "gated_packed": (R, "GATED_GRAD_PACKED"), "exp21f": (TB, "EXP_21F"), "pair_mm": (PM, "PAIR_MM_FUSED")}
arms = a.arms or ("off,entry,on" if a.lever == "gated" else "off,on")
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=0, evo=a.evo,
                        blocks=a.blocks, controls_all=False, controls_only=False, seed=0,
                        msa_mask=False, threads=8)
with bindcraft2.fast_round():
    for arm in arms.split(","):
        stage1 = a.lever in LATER or arm != "off"
        os.environ["TT_BIO_TAPED_KERNELS"] = BASE + (",reblock_permute_gated" if stage1 else "") + (
            ",pair_transpose" if a.lever in LATER else "")
        R.GATED_BW_FUSED = a.lever in LATER or arm == "on"
        for lv, (obj, attr) in SWITCH.items():
            mine = a.lever in LATER and LATER.index(lv) <= LATER.index(a.lever)
            setattr(obj, attr, mine and (lv != a.lever or arm == "on"))
        s0, l0, p0 = list(GB.STATS), list(LS.STATS), dict(TN.PAIR_BIAS_STATS)
        x0 = (dict(TB.STATS), dict(AG.FANIN_CAST_STATS), list(PT.STATS), dict(AG.SLAB_STATS))
        g0, b0, m0 = list(R.STATS_GATED), list(R.STATS_GATED_BW), list(PM.STATS)
        e0 = list(T.KERNEL_STATS.get("reblock_permute_gated", [0, 0]))
        ns.tag = f"bcp_evo_{a.lever}_{arm}"
        print(f"== arm {arm} kernels={os.environ['TT_BIO_TAPED_KERNELS']} "
              f"GATED_BW_FUSED={R.GATED_BW_FUSED} "
              f"{ {lv: getattr(o, at) for lv, (o, at) in SWITCH.items()} }", flush=True)
        A.cmd_vjp(ns)
        e1 = T.KERNEL_STATS.get("reblock_permute_gated", [0, 0])
        print(f"== arm {arm} reach gated_move {[x - y for x, y in zip(R.STATS_GATED, g0)]} "
              f"entry {[x - y for x, y in zip(e1, e0)]} "
              f"gated_bw {[x - y for x, y in zip(R.STATS_GATED_BW, b0)]} "
              f"gate_bw {[x - y for x, y in zip(GB.STATS, s0)]} "
              f"lead_sum {[x - y for x, y in zip(LS.STATS, l0)]} "
              f"pair_bias {({k: v - p0.get(k, 0) for k, v in TN.PAIR_BIAS_STATS.items()})} "
              f"triatt_bw {({k: v - x0[0].get(k, 0) for k, v in TB.STATS.items()})} "
              f"fanin_cast {({k: v - x0[1].get(k, 0) for k, v in AG.FANIN_CAST_STATS.items()})} "
              f"pair_transpose {[x - y for x, y in zip(PT.STATS, x0[2])]} "
              f"slab {({k: v - x0[3].get(k, 0) for k, v in AG.SLAB_STATS.items()})} "
              f"pair_mm {[x - y for x, y in zip(PM.STATS, m0)]}",
              flush=True)
