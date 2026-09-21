#!/usr/bin/env python3
"""D116, part 4: is the row-sum deficit a property of `ttnn.softmax`, or of calling it
without a `compute_kernel_config`?

parts 1-3 measured the no-config call and found rows summing to 0.9934. The
triangle-attention backward passes `precise_config()` to its softmax and the repair bought
nothing there. Those two facts only fit together one way, and this is the control that
decides it. Everything is scored against a float64 softmax on the same bf16 input values.
"""
import argparse, json, os, pathlib, sys, time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT), (
    f"tt_bio came from {_tt_bio.__file__}, not {_ROOT}")

import torch
import ttnn

from tt_bio.autograd import precise_config
from tt_bio.tenstorrent import get_device


def rel_l2(a, b):
    return float(torch.linalg.vector_norm((a - b).double()) / torch.linalg.vector_norm(b.double()))


def one(dev, shape, std, seed, cfg_name, cfg, dtype=ttnn.bfloat16):
    torch.manual_seed(seed)
    x = (torch.randn(*shape) * std).to(torch.bfloat16)
    g = torch.randn(*shape).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    g_tt = ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    kw = {} if cfg is None else {"compute_kernel_config": cfg}
    y_tt = ttnn.softmax(x_tt, dim=-1, **kw)
    y = ttnn.to_torch(y_tt).double()
    x64, g64 = x.double(), g.double()
    p = torch.softmax(x64, dim=-1)
    rows = y.sum(-1, keepdim=True)

    inner = (g64 * y).sum(-1, keepdim=True)
    dx_ref = p * (g64 - (g64 * p).sum(-1, keepdim=True))
    dx_ship = y * (g64 - inner)
    dx_rn = y * (g64 - inner / rows)

    # the device arms too, so the rule and the arithmetic are separated at each config
    i_tt = ttnn.sum(ttnn.multiply(g_tt, y_tt), dim=-1, keepdim=True)
    dx_ship_dev = ttnn.to_torch(ttnn.multiply(y_tt, ttnn.subtract(g_tt, i_tt))).double()
    r_tt = ttnn.sum(y_tt, dim=-1, keepdim=True, compute_kernel_config=precise_config())
    dx_rn_dev = ttnn.to_torch(
        ttnn.multiply(y_tt, ttnn.subtract(g_tt, ttnn.divide(i_tt, r_tt)))).double()

    return dict(
        config=cfg_name, shape=list(shape), std=std,
        storage="float32" if dtype == ttnn.float32 else "bfloat16",
        rowsum_mean=float(rows.mean()), rowsum_rmsdev=float((rows - 1).pow(2).mean().sqrt()),
        rowsum_min=float(rows.min()), rowsum_max=float(rows.max()),
        y_rel_l2=rel_l2(y, p),
        dx=dict(shipped_rule_f64=rel_l2(dx_ship, dx_ref), renorm_rule_f64=rel_l2(dx_rn, dx_ref),
                shipped_device=rel_l2(dx_ship_dev, dx_ref), renorm_device=rel_l2(dx_rn_dev, dx_ref)),
        dx_rowsum_rms=dict(
            reference=float(dx_ref.sum(-1).pow(2).mean().sqrt()),
            shipped_rule_f64=float(dx_ship.sum(-1).pow(2).mean().sqrt()),
            renorm_rule_f64=float(dx_rn.sum(-1).pow(2).mean().sqrt()),
            shipped_device=float(dx_ship_dev.sum(-1).pow(2).mean().sqrt()),
            renorm_device=float(dx_rn_dev.sum(-1).pow(2).mean().sqrt())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_d116/ckc.json")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    dev = get_device()
    hifi2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                             math_approx_mode=False, fp32_dest_acc_en=False,
                                             packer_l1_acc=False)
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                            math_approx_mode=False, fp32_dest_acc_en=False,
                                            packer_l1_acc=False)
    hifi4_nofp32 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                    math_approx_mode=False,
                                                    fp32_dest_acc_en=False, packer_l1_acc=False)
    configs = [("none (as _v_softmax ships)", None), ("precise_config()", precise_config()),
               ("HiFi4, no fp32_dest_acc", hifi4_nofp32), ("HiFi2", hifi2), ("LoFi", lofi)]
    rows, t0 = [], time.time()
    for std in (3.0, 12.0):
        for nm, cfg in configs:
            rows.append(one(dev, (1, 16, 384, 384), std, a.seed, nm, cfg))
    # and the same two, in fp32 storage, to separate storage from fidelity
    for nm, cfg in (("none (as _v_softmax ships)", None), ("precise_config()", precise_config())):
        rows.append(one(dev, (1, 16, 384, 384), 3.0, a.seed, nm + " [fp32 store]", cfg,
                        dtype=ttnn.float32))
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=os.uname().nodename, seconds=round(time.time() - t0, 1), cases=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, out["seconds"], "s")


if __name__ == "__main__":
    main()
