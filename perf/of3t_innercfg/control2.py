#!/usr/bin/env python3
"""Sharpen CONTROL.json: is `compute_kernel_config` HONOURED by `ttnn.sum` and already at its
ceiling, or IGNORED?

CONTROL.json showed `precise_config()` bit-identical to no config on every `ttnn.sum` arm while
the same object moved a matmul 4.1x. That is consistent with two things. This decides it by
handing the reduction a config that should make it WORSE. A config that cannot make a reduction
worse is not being read.
"""
from __future__ import annotations

import json, pathlib, platform, subprocess, sys, time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio  # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT)

import torch, ttnn  # noqa: E402
from tt_bio.autograd import precise_config  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

nrm = torch.linalg.vector_norm


def cfgs():
    F = ttnn.MathFidelity
    return {
        "none": None,
        "precise (HiFi4+fp32dest+packerl1)": precise_config(),
        "HiFi4, fp32_dest_acc OFF": ttnn.WormholeComputeKernelConfig(
            math_fidelity=F.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=False),
        "LoFi, fp32_dest_acc OFF, approx ON": ttnn.WormholeComputeKernelConfig(
            math_fidelity=F.LoFi, math_approx_mode=True, fp32_dest_acc_en=False,
            packer_l1_acc=False),
        "HiFi2, fp32_dest_acc OFF": ttnn.WormholeComputeKernelConfig(
            math_fidelity=F.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=False),
    }


def main():
    dev = get_device()
    rep = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=platform.node(),
               commit=subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip())

    # the real operands: an exact softmax y and a matmul-shaped g, the trunk's own axis
    torch.manual_seed(17)
    s = torch.randn(8, 4, 384, 384) * 3.0
    y = torch.softmax(s.to(torch.bfloat16).double(), dim=-1).to(torch.bfloat16)
    g = torch.randn(8, 4, 384, 384).to(torch.bfloat16)
    yt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    prod = ttnn.multiply(gt, yt)
    ref = (ttnn.to_torch(gt).double() * ttnn.to_torch(yt).double()).sum(-1, keepdim=True)

    base = None
    out = {}
    for name, c in cfgs().items():
        kw = {} if c is None else {"compute_kernel_config": c}
        r = ttnn.to_torch(ttnn.sum(prod, dim=-1, keepdim=True, **kw)).double()
        if base is None:
            base = r
        out[name] = dict(rel_l2_vs_f64=float(nrm(r - ref) / nrm(ref)),
                         bitwise_equal_to_none=bool(torch.equal(r, base)))
    rep["numerator_sum_dim_-1_shape_8x4x384x384"] = out

    # the same sweep on a matmul, so the sweep itself is shown to be able to separate configs
    torch.manual_seed(5)
    a = (torch.randn(1, 1, 256, 512) * 0.5).to(torch.bfloat16)
    b = (torch.randn(1, 1, 512, 256) * 0.5).to(torch.bfloat16)
    at = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    bt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mref = a.double() @ b.double()
    mbase = None
    mout = {}
    for name, c in cfgs().items():
        kw = {} if c is None else {"compute_kernel_config": c}
        r = ttnn.to_torch(ttnn.matmul(at, bt, **kw)).double()
        if mbase is None:
            mbase = r
        mout[name] = dict(rel_l2_vs_f64=float(nrm(r - mref) / nrm(mref)),
                          bitwise_equal_to_none=bool(torch.equal(r, mbase)))
    rep["BREAK_CONTROL_matmul_same_sweep"] = mout

    # other reduction axes, in case dim=-1 is a special-cased kernel
    axis = {}
    for dim in (-1, -2, 0):
        r0 = ttnn.to_torch(ttnn.sum(prod, dim=dim, keepdim=True)).double()
        r1 = ttnn.to_torch(ttnn.sum(prod, dim=dim, keepdim=True,
                                    compute_kernel_config=precise_config())).double()
        axis[f"dim={dim}"] = dict(bitwise_equal=bool(torch.equal(r0, r1)),
                                  max_abs_diff=float((r0 - r1).abs().max()))
    rep["axis_sweep_precise_vs_none"] = axis

    p = pathlib.Path(_ROOT) / "perf/of3t_innercfg/CONTROL2.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
