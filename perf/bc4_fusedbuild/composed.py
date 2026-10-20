"""Baseline for bc4-bet-fusedbuild: today's AF2 trimul forward as BindCraft 2 runs it, both directions, per call.

tt_bio.tenstorrent.TriangleMultiplication under bindcraft2.fast_round() (GIN_BIASED + T2CM armed), offered the
residual the way AF2PairBlock._pair_op offers it, so a call is z + mask * update in one new tensor. The served
counters of gin_ragged and T2 are recorded: a row whose levers never ran measures nothing.

    python perf/bc4_fusedbuild/composed.py --out <jsonl>
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=21)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, bindcraft2, bmm_res
from tt_bio.af2 import compute_kernel_config
from perf.bc4_fusedbuild.common import N, C, inputs, ref_out, grade, Clock, Out, traced

dev = T.get_device(trace="bc2")
clk = Clock()
out = Out(A.out, clk)
z, mask, w = inputs()
sd = {k: v.float() for k, v in w.items()}
Z = ttnn.from_torch(z.reshape(1, N, N, C).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
MK = ttnn.from_torch(mask.reshape(1, N, N).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
with bindcraft2.fast_round() as armed:
    ckc = compute_kernel_config()
    for ending in (False, True):
        r = {"arm": "composed_fast_round", "ending": ending, "armed": {k: str(v) for k, v in armed.items()
                                                                       if "T2CM" in k or "GIN" in k}}
        try:
            m = T.TriangleMultiplication(ending, sd, ckc)

            def f():
                o = m(Z, MK, fused_residual=(None,))
                if not m.took_residual:      # tail declined the residual: add it as the block would
                    o2 = ttnn.add(Z, o); ttnn.deallocate(o); o = o2
                return [o]
            s0 = (list(TT.GIN_RAGGED_STATS), list(TT.T2CM_STATS), list(bmm_res.STATS))
            o = f()[0]
            r.update(grade(ref_out(z, mask, w, ending), ttnn.to_torch(o)))
            ttnn.deallocate(o)
            r["served"] = {"gin_ragged": [a - b for a, b in zip(TT.GIN_RAGGED_STATS, s0[0])],
                           "t2cm": [a - b for a, b in zip(TT.T2CM_STATS, s0[1])],
                           "bmm_res": [a - b for a, b in zip(bmm_res.STATS, s0[2])], "took_residual": m.took_residual,
                           "fused_gout_env": os.environ.get("TT_BIO_TRIMUL_FUSED_GOUT")}
            r.update(traced(dev, f, reps=A.reps))
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out.emit(r)
out.emit({"arm": "clock", **clk.summary()})
