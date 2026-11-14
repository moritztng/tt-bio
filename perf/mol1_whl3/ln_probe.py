"""Device micro-test: WH's trunk block 43 single-Transition LayerNorm on its captured input, under several configs.

    TT_VISIBLE_DEVICES=N python ln_probe.py TAP.pt LNW.pt OUT.json
Per arm: per-token scale error std of the normalized output (affine fit vs float64 on the same input), the token part of
the output error against the token part of the float64 output, and whether the arm reproduces the fold's own LN."""
import json, sys
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.tenstorrent import get_device
tap = torch.load(sys.argv[1], weights_only=False); lnw = torch.load(sys.argv[2], weights_only=False)
x = tap["in"][0].float().reshape(1, 1, -1, 384); fold = tap["layer_norm"][0].double().reshape(-1, 384)
lw, lb = lnw["w"].double(), lnw["b"].double()
xd = x.double().reshape(-1, 384)
xh = (xd - xd.mean(-1, keepdim=True)) / (xd.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
ref = xh * lw + lb
dev = get_device()
wh = dev.arch() == ttnn.Arch.WORMHOLE_B0
Cfg = ttnn.types.WormholeComputeKernelConfig if wh else ttnn.types.BlackholeComputeKernelConfig
def grade(o):
    y = (o - lb) / lw
    A = torch.stack([xh, torch.ones_like(xh)], -1)
    a = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)[:, 0]
    d = o - ref
    return {"scale_std": float(a.std()), "tok_err_rms": float((d - d.mean(0)).pow(2).mean().sqrt()),
            "tok_ref_rms": float((ref - ref.mean(0)).pow(2).mean().sqrt()), "eq_fold": bool(torch.equal(o.float(), fold.float())),
            "max_vs_fold": float((o - fold).abs().max())}
res = {"arch": "wh" if wh else "bh"}
up = lambda t, dt: ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, 1, 1, -1),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
for name, xdt, wdt, fid, acc, l1 in (("hifi4_acc", ttnn.bfloat16, ttnn.bfloat16, "HiFi4", True, False),
                                     ("hifi4_acc_l1", ttnn.bfloat16, ttnn.bfloat16, "HiFi4", True, True),
                                     ("hifi4_noacc", ttnn.bfloat16, ttnn.bfloat16, "HiFi4", False, False),
                                     ("hifi3_acc", ttnn.bfloat16, ttnn.bfloat16, "HiFi3", True, False),
                                     ("hifi2_acc", ttnn.bfloat16, ttnn.bfloat16, "HiFi2", True, False),
                                     ("x_fp32_hifi4_acc", ttnn.float32, ttnn.bfloat16, "HiFi4", True, False),
                                     ("x_fp32_w_fp32_hifi4_acc", ttnn.float32, ttnn.float32, "HiFi4", True, False)):
    try:
        ckc = Cfg(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=l1)
        o = ttnn.layer_norm(up(x, xdt), weight=up(lw.float(), wdt), bias=up(lb.float(), wdt), epsilon=1e-5,
                            compute_kernel_config=ckc)
        o = ttnn.typecast(o, ttnn.bfloat16) if o.dtype != ttnn.bfloat16 else o
        res[name] = grade(ttnn.to_torch(o).double().reshape(-1, 384)[: xd.shape[0]])
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
