#!/usr/bin/env python3
"""C4 float64 grade: `perf/bcx_afgrad` vjp (teacher-forced blocks, float64 VJP on the same inputs)
under fast_round with the real MSA mask, so the masked OPM's `_sum_rows` is on the path. Arms:
off = rows summed after S products, on = rows joined along the contraction."""
import argparse, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import af2, bindcraft2
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=288, extra=0, evo=8, blocks="0,3,7",
                        controls_all=False, controls_only=False, seed=0, msa_mask=True, threads=8)
# afgrad embeds ONE sequence, a 1-row MSA, where `rows_in_k` never acts. BindCraft 2 runs 2 rows;
# the second row here is the embedding of a second random sequence, concatenated on the row axis.
_embed = A.embed
def embed2(mod, logits, ridx):
    m1, z = _embed(mod, logits, ridx)
    g = torch.Generator().manual_seed(1)
    m2, _ = _embed(mod, (torch.randn(logits.shape, generator=g) * 2.0).to(logits.dtype), ridx)
    return torch.cat([m1, m2], dim=-3), z
A.embed = embed2
calls = [0]
orig = af2.AF2MaskedOuterProductMean._sum_rows
def counted(self, a, b):
    calls[0] += 1
    return orig(self, a, b)
af2.AF2MaskedOuterProductMean._sum_rows = counted
with bindcraft2.fast_round():
    for arm in ("off", "on"):
        af2.AF2MaskedOuterProductMean.rows_in_k = arm == "on"
        calls[0] = 0
        ns.tag = f"bcw_callcut_opm2_{arm}"
        print(f"== arm {arm} rows_in_k={af2.AF2MaskedOuterProductMean.rows_in_k}", flush=True)
        A.cmd_vjp(ns)
        print(f"== arm {arm} reach _sum_rows calls {calls[0]}", flush=True)
