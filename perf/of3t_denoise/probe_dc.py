#!/usr/bin/env python3
"""of3t-denoise D257 root cause: the diffusion conditioning's dW, in isolation, against float64.

    probe_dc.py --out F.json [--tokens N]

Builds `OF3DiffusionConditioning` from the checkpoint exactly as `OpenFold3` does (fp32 under
`device_dtype_override`), feeds it random bf16 trunk-like inputs through the sampler's own dtype
boundary, runs `single` and `pair` on the tape and seeds random cotangents. Each dW is compared
with float64 X^T G computed on host from the SAME device X (read back) and the same seed. Then the
bare matmul the linear backward issues is re-run on fresh operands to find which operand property
produces the garbage: dtype of X, dtype of G, and whether X came from ttnn.layer_norm/concat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def rel(a, b):
    a, b = a.double(), b.double()
    bad = int((~torch.isfinite(a) | (a.abs() > 1e30)).sum())
    af = torch.where(torch.isfinite(a) & (a.abs() <= 1e30), a, torch.zeros_like(a))
    return {"bad": bad, "n": a.numel(), "rel_finite_part": float((af - b).norm() / b.norm())}


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    n = int(argv[argv.index("--tokens") + 1]) if "--tokens" in argv else 64
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion import OF3DiffusionConditioning
    from tt_bio.openfold3_weights import _sub

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(Path.home() / "of3-weights/of3-p2-155k.pt", map_location="cpu",
                    weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    with device_dtype_override(ttnn.float32):
        dc = OF3DiffusionConditioning(_sub(sd, "diffusion_module.diffusion_conditioning"), ckc)
    rec = {"tokens": n, "weight_dtype": str(dc.w_lin_s.dtype)}
    g = torch.Generator().manual_seed(0)
    bf = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa
    import tt_bio.openfold3_sample_diffusion as sdm
    # the sampler module's own `ttnn` name, which the tape swaps for its shim
    c = lambda x: sdm.ttnn.typecast(x, ttnn.float32) if x.dtype == ttnn.bfloat16 else x  # noqa
    s_trunk = torch.randn(1, n, 384, generator=g)
    s_input = torch.randn(1, n, 449, generator=g)
    n_emb = torch.randn(1, 1, 256, generator=g)
    z_trunk = torch.randn(1, n, n, 128, generator=g)
    relpos = (torch.rand(1, n, n, 139, generator=g) > 0.9).float()
    tok = torch.ones(1, n, 1)
    names = {"w_lin_s": dc.w_lin_s, "w_lin_n": dc.w_lin_n, "w_lin_z": dc.w_lin_z}
    leaves = {k: ag.parameter(v) for k, v in names.items()}
    seen = {}
    orig_ln = ttnn.layer_norm

    with ag.tape():
        si = dc.single(c(ag.Tensor(bf(s_trunk))), c(ag.Tensor(bf(s_input))),
                       c(ag.Tensor(bf(n_emb))), c(bf(tok)))
        zij = dc.pair(c(ag.Tensor(bf(z_trunk))), c(ag.Tensor(bf(relpos))),
                      c(bf(tok.reshape(1, n, 1, 1) * tok.reshape(1, 1, n, 1))))
    gs = torch.randn(1, n, 384, generator=g)
    gz = torch.randn(1, n, n, 128, generator=g)
    ag.backward([si, zij], [ttnn.from_torch(gs, layout=ttnn.TILE_LAYOUT, device=dev,
                                            dtype=si.value.dtype),
                            ttnn.from_torch(gz, layout=ttnn.TILE_LAYOUT, device=dev,
                                            dtype=zij.value.dtype)])
    rec["si_dtype"], rec["zij_dtype"] = str(si.value.dtype), str(zij.value.dtype)
    for k, t in leaves.items():
        gd = ttnn.to_torch(t.grad).double()
        v = gd.reshape(-1)
        bad = (~torch.isfinite(v)) | (v.abs() > 1e30)
        gd2 = gd.reshape(tuple(t.value.shape))
        badrows = ((~torch.isfinite(gd2)) | (gd2.abs() > 1e30)).any(-1).nonzero().flatten()
        rec[k] = {"grad_dtype": str(t.grad.dtype), "shape": list(t.value.shape),
                  "bad": int(bad.sum()), "n": v.numel(),
                  "bad_rows_head": badrows[:12].tolist(), "n_bad_rows": len(badrows)}
    print(json.dumps(rec), flush=True)

    # The bare dW matmul, on fresh operands of the dc shapes.
    from tt_bio.autograd import _flat2d
    bare = {}
    for K, M in ((833, 384), (256, 384), (267, 128), (768, 384)):
        for xd in (ttnn.float32, ttnn.bfloat16):
            for gd_ in (ttnn.float32, ttnn.bfloat16):
                X = torch.randn(n, K, generator=g)
                G = torch.randn(n, M, generator=g)
                Xd = ttnn.from_torch(X, layout=ttnn.TILE_LAYOUT, device=dev, dtype=xd)
                Gd = ttnn.from_torch(G, layout=ttnn.TILE_LAYOUT, device=dev, dtype=gd_)
                Xh = ttnn.to_torch(Xd).double()
                Gh = ttnn.to_torch(Gd).double()
                dW = ttnn.matmul(_flat2d(Xd), _flat2d(Gd), transpose_a=True,
                                 compute_kernel_config=ag.precise_config(), dtype=ttnn.float32)
                r = rel(ttnn.to_torch(dW), Xh.T @ Gh)
                bare[f"K{K}_M{M}_x{str(xd).split('.')[-1]}_g{str(gd_).split('.')[-1]}"] = r
    rec["bare_matmul_transpose_a"] = bare
    for k, v in bare.items():
        print(k, v, flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
