"""Numerical error of whichever silu LLK the current TT_METAL_RUNTIME_ROOT supplies.

Exercises the fused-activation path with protenix own ckc (HiFi4, fp32_dest_acc_en=True,
math_approx_mode=True) by multiplying through an identity weight, so the value the SFPU sees is
the input and the comparison is against torch silu in fp32.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, ttnn
from tt_bio.tenstorrent import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

K = 32
dev = get_device()
rec = {"tag": a.tag, "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT", "<unset>")}
try:
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    rows = 512
    xt = torch.empty(rows, K)
    # half a realistic post-layernorm draw, half a wide sweep so the tails are covered
    xt[: rows // 2] = torch.randn(rows // 2, K)
    xt[rows // 2:] = torch.linspace(-12, 12, (rows - rows // 2) * K).reshape(-1, K)
    xt = xt.to(torch.bfloat16)

    x = ttnn.from_torch(xt.reshape(1, 1, rows, K), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.eye(K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(o).reshape(rows, K).float()
    ref = torch.nn.functional.silu(xt.float())

    err = (got - ref).abs()
    den = ref.abs().clamp(min=1e-3)
    g, r = got.flatten(), ref.flatten()
    pcc = float(torch.corrcoef(torch.stack([g, r]))[0, 1])
    rec.update({
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "rms_err": float((err ** 2).mean().sqrt()),
        "max_rel_err": float((err / den).max()),
        "pcc": pcc,
        "ref_absmax": float(ref.abs().max()),
        # the band that matters: a post-layernorm draw, |x| <= 4
        "max_abs_err_core": float(err[xt.abs() <= 4].max()),
    })
    ttnn.deallocate(o); ttnn.deallocate(x); ttnn.deallocate(w)
except Exception as e:
    rec["error"] = repr(e)
json.dump(rec, open(a.out, "w"), indent=2)
