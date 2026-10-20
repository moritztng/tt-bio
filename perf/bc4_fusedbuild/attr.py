"""Attribution of the two stages after phase 1 (bc4-bet-fusedbuild): bmm_res contraction and the T2CM tail, each
timed traced with its own bench-only stage-skip bits, so the part a single program could not shed (compute) is
separated from the DRAM hand-off it could. Outgoing direction, HiFi4.  python perf/bc4_fusedbuild/attr.py --out f"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--fid", default="HiFi4")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, bmm_res, mm_generic
from perf.bc4_fusedbuild.common import N, C, inputs, Clock, Out, traced
from perf.bc4_fusedbuild import fused

dev = T.get_device()
clk = Clock()
out = Out(A.out, clk)
z, mask, w = inputs()
Z = fused._dram(z.reshape(1, N, N, C).float(), dev)
MK = fused.mask_rows(mask, dev)
W = fused.weights(w, dev)
fid = getattr(ttnn.MathFidelity, A.fid)
ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                              packer_l1_acc=True)
m = T.TriangleMultiplication(False, {k: v.float() for k, v in w.items()}, ckc)
rows = (TT.bias_rows(m.p_out_bias), TT.bias_rows(m.g_out_bias))
a, b = fused.phase1(Z, W, MK, N, fid)
for abl in (0, 1, 2, 4, 6):
    r = {"arm": "contract_abl", "abl": abl, "fidelity": A.fid}
    try:
        f = lambda: [bmm_res.matmul(a, b, ckc, transpose_b=True, abl=abl)]
        [t.deallocate() for t in f()]
        ttnn.synchronize_device(dev)
        r.update(traced(dev, f))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    out.emit(r)
x = bmm_res.matmul(a, b, ckc, transpose_b=True)
xb = ttnn.layer_norm(Z, weight=m.in_norm_weight, bias=m.in_norm_bias, epsilon=1e-5, compute_kernel_config=ckc)
for abl in (0, 1, 2, 4, 8, 15):
    TT.T2_ABL = abl
    r = {"arm": "tail_abl", "abl": abl, "fidelity": A.fid}
    try:
        def tail():
            o = ttnn.allocate_tensor_on_device(Z.shape, Z.dtype, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
            res = TT.fused_tail_ln_cm(x, m.out_norm_weight, m.out_norm_bias, m.out_p_weight, xb, Z,
                                      mm_generic.ckc_args(ckc), tuple(T.COMPUTE_GRID_MAIN), wg=m.g_out_weight,
                                      bp=rows[0], bg=rows[1], mask=None, out=o)
            assert res is not None, TT.REJECTS
            return [res]
        [t.deallocate() for t in tail()]
        ttnn.synchronize_device(dev)
        r.update(traced(dev, tail))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    out.emit(r)
TT.T2_ABL = 0
out.emit({"arm": "clock", **clk.summary()})
