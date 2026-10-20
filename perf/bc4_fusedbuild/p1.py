"""Build step 1 check: the fused kernel's phase 1 alone (LN in, gated projections, rotation) against float64 a, b,
and its traced time in each sigmoid mode and fidelity.  python perf/bc4_fusedbuild/p1.py --out <jsonl>"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--abl", type=int, nargs="*", default=[0])
ap.add_argument("--sig", type=int, nargs="*", default=[1, 3])
ap.add_argument("--dest16", action="store_true")
ap.add_argument("--fid", nargs="*", default=["HiFi4", "HiFi2"])
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from perf.bc4_fusedbuild.common import N, C, VALID, inputs, Clock, Out, traced
from perf.bc4_fusedbuild import fused

dev = T.get_device()
clk = Clock()
out = Out(A.out, clk)
z, mask, w = inputs()
zn = torch.nn.functional.layer_norm(z, (C,), w["norm_in.weight"], w["norm_in.bias"], 1e-5)
lin = lambda x, k: x @ w[f"{k}.weight"].T + w[f"{k}.bias"]  # noqa: E731
x = torch.sigmoid(lin(zn, "g_in")) * lin(zn, "p_in")
ref_a = (x[..., :C] * mask[..., None]).permute(2, 0, 1)
ref_b = x[..., C:].permute(2, 0, 1)
Z = fused._dram(z.float(), dev)
MK = fused.mask_rows(mask, dev)
W = fused.weights(w, dev)
P1_FLOP = 2 * N * N * C * 512
for fid in [getattr(ttnn.MathFidelity, f) for f in A.fid]:
    ckc = fid
    for sig in A.sig:
        for abl in A.abl:
            r = {"arm": "phase1", "fidelity": str(fid), "sig": sig, "abl": abl, "dest32": not A.dest16}
            try:
                a, b = fused.phase1(Z, W, MK, N, ckc, sig=sig, abl=abl, dest32=not A.dest16)
                ga, gb = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
                for nm, g, ref in (("a", ga, ref_a), ("b", gb, ref_b)):
                    d = (g - ref)[:, :VALID, :VALID]
                    r[f"{nm}_rel_l2"] = round(float(d.norm() / ref[:, :VALID, :VALID].norm()), 6)
                    r[f"{nm}_max_abs"] = round(float(d.abs().max()), 5)
                bufs = [a, b]
                t = traced(dev, lambda: [] if fused.phase1(Z, W, MK, N, ckc, sig=sig, abl=abl, out=bufs, dest32=not A.dest16) else [])
                t["tflops_p1"] = round(P1_FLOP / t["ms"] / 1e9, 2)
                r.update(t)
                ttnn.deallocate(a); ttnn.deallocate(b)
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            out.emit(r)
out.emit({"arm": "clock", **clk.summary()})
