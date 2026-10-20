"""The whole AF2 trimul forward with the fused kernel's phase 1, measured end to end (bc4-bet-fusedbuild).

Three programs, every intermediate through DRAM: phase 1 (tt_bio/kernels/trimul_fused: LN in, gated projections,
mask, rotation to channel-major a, b), the per-channel contraction (bmm_res, a and b read once) and the output tail
with the residual (T2CM: LN out, out-projection, g_out gate, z + update), plus ttnn.layer_norm for the gate's input
norm, which T2CM reads. Graded against float64 on the real block; timed traced whole and per stage.

    python perf/bc4_fusedbuild/chain.py --out <jsonl> [--sig 1 3] [--fid HiFi4 HiFi2]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--sig", type=int, nargs="*", default=[1, 3])
ap.add_argument("--fid", nargs="*", default=["HiFi4", "HiFi2"])
ap.add_argument("--reps", type=int, default=21)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, bmm_res, mm_generic
from perf.bc4_fusedbuild.common import N, C, inputs, ref_out, grade, Clock, Out, traced
from perf.bc4_fusedbuild import fused

dev = T.get_device()
clk = Clock()
out = Out(A.out, clk)
z, mask, w = inputs()
Z = fused._dram(z.reshape(1, N, N, C).float(), dev)
MK = fused.mask_rows(mask, dev)
W = fused.weights(w, dev)
grid = tuple(T.COMPUTE_GRID_MAIN)
for fname in A.fid:
    fid = getattr(ttnn.MathFidelity, fname)
    ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                                  packer_l1_acc=True)
    m = T.TriangleMultiplication(False, {k: v.float() for k, v in w.items()}, ckc)
    rows = (TT.bias_rows(m.p_out_bias), TT.bias_rows(m.g_out_bias))
    cargs = mm_generic.ckc_args(ckc)
    for sig in A.sig:
        for ending in (False, True):
            r = {"arm": "chain_p1_bres_t2cm", "fidelity": fname, "sig": sig, "ending": ending}
            try:
                bufs = {}

                def p1():
                    bufs["ab"] = fused.phase1(Z, W, MK, N, fid, sig=sig, out=bufs.get("ab"))
                    return []

                def contract():
                    a, b = bufs["ab"]
                    x = bmm_res.matmul(a, b, ckc, transpose_a=ending, transpose_b=not ending)
                    assert x is not None, "bmm_res declined"
                    return x

                def norm():
                    return ttnn.layer_norm(Z, weight=m.in_norm_weight, bias=m.in_norm_bias, epsilon=1e-5,
                                           compute_kernel_config=ckc)

                def tail(x, xb):
                    o = ttnn.allocate_tensor_on_device(Z.shape, Z.dtype, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
                    s0 = list(TT.T2CM_STATS)
                    res = TT.fused_tail_ln_cm(x, m.out_norm_weight, m.out_norm_bias, m.out_p_weight, xb, Z, cargs,
                                              grid, wg=m.g_out_weight, bp=rows[0], bg=rows[1], mask=None, out=o)
                    assert res is not None, f"T2CM declined {TT.REJECTS}"
                    return res

                def whole():
                    p1()
                    x = contract(); xb = norm()
                    o = tail(x, xb)
                    return [x, xb, o]

                ts = whole()
                r.update(grade(ref_out(z, mask, w, ending), ttnn.to_torch(ts[2])))
                for t in ts:
                    ttnn.deallocate(t)
                r["whole"] = traced(dev, whole, reps=A.reps)
                r["p1"] = traced(dev, p1, reps=A.reps)
                r["contract"] = traced(dev, lambda: [contract()], reps=A.reps)
                r["ln_in"] = traced(dev, lambda: [norm()], reps=A.reps)
                x, xb = contract(), norm()
                r["tail"] = traced(dev, lambda: [tail(x, xb)], reps=A.reps)
                ttnn.deallocate(x); ttnn.deallocate(xb)
                r["ms"], r["tflops"] = r["whole"]["ms"], r["whole"]["tflops"]
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            out.emit(r)
out.emit({"arm": "clock", **clk.summary()})
