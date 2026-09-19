#!/usr/bin/env python3
"""The patched OuterProductMean against a float64 reference, and against its own legacy arm.

Runs the REAL engine class both ways (TT_BIO_OPM_LEGACY_LAYOUT=1 and the shipped default), so what
is scored is the production path, not a hand-rolled copy of it. Size reduced to I=J=128 only
because the float64 reference has to fit on the host; every other dimension is the 512 aa cell's
own (MSA depth 64, c_m 64, C=D=32, c_z=128).
"""
import json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/c14-opm-layout")
import ttnn
from tt_bio import tenstorrent as T

S, I, J, CM, C, D, CZ = 64, 128, 128, 64, 32, 32, 128
torch.manual_seed(7)
sd = {
    "norm.weight": torch.randn(CM) / 4 + 1,
    "norm.bias": torch.randn(CM) / 8,
    "proj_a.weight": torch.randn(C, CM) / 8,
    "proj_b.weight": torch.randn(D, CM) / 8,
    "proj_o.weight": torch.randn(CZ, C * D) / 32,
    "proj_o.bias": torch.randn(CZ) / 8,
}
x_t = (torch.randn(1, S, I, CM) / 2).to(torch.bfloat16)

# ---- float64 reference, from the bf16 input, exact arithmetic ----
x64 = x_t.to(torch.float64)[0]
mu = x64.mean(-1, keepdim=True)
var = x64.var(-1, unbiased=False, keepdim=True)
m64 = (x64 - mu) / torch.sqrt(var + 1e-5) * sd["norm.weight"].double() + sd["norm.bias"].double()
a64 = m64 @ sd["proj_a.weight"].double().t()          # (S,I,C)
b64 = m64 @ sd["proj_b.weight"].double().t()          # (S,J,D)
z64 = torch.einsum("sic,sjd->ijcd", a64, b64).reshape(I, J, C * D) / S
ref = z64 @ sd["proj_o.weight"].double().t() + sd["proj_o.bias"].double()

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)

def run(legacy: bool):
    os.environ["TT_BIO_OPM_LEGACY_LAYOUT"] = "1" if legacy else "0"
    opm = T.OuterProductMean(dict(sd), ckc)
    x = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = opm(x, None, None)
    t = ttnn.to_torch(out)
    ttnn.deallocate(out)
    return t

leg = run(True)
new = run(False)
os.environ.pop("TT_BIO_OPM_LEGACY_LAYOUT", None)

def err(t):
    o = t.to(torch.float64).reshape(I, J, CZ)
    return {"max_abs": float((o - ref).abs().max()), "mean_abs": float((o - ref).abs().mean()),
            "ref_absmax": float(ref.abs().max()),
            "rel_max": float((o - ref).abs().max() / ref.abs().max())}

res = {"shape": {"S": S, "I": I, "J": J, "c_m": CM, "C": C, "D": D, "c_z": CZ},
       "legacy_vs_float64": err(leg), "shipped_vs_float64": err(new),
       "bit_exact_legacy_vs_shipped": bool(torch.equal(leg, new)),
       "max_abs_legacy_vs_shipped": float((leg.to(torch.float64) - new.to(torch.float64)).abs().max()),
       "out_shape": list(new.shape)}
print(json.dumps(res, indent=1))
Path("/home/ttuser/.coworker/wt/c14-opm-layout/perf/c14_opm_layout/numerics_qb1c2.json").write_text(
    json.dumps(res, indent=1))
