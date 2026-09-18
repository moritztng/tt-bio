#!/usr/bin/env python3
"""The full-shape arms disagreed by MORE than the signal. Which variant is wrong, against a host reference.

`opm_transpose_a.py` found `transpose_a=True` bit-exact against `permute + matmul` at a small
shape (K=64) and then, at the production shape (K=1024) with `core_grid` pinned on both arms,
found a max_abs of 0.609 against a signal whose own absmax is 0.539. A deviation larger than the
signal is not rounding. Two suspects, and they are not equivalent:

  * `transpose_a=True` combined with an explicit `core_grid` picks a program config that
    contracts the wrong thing. That is a CORRECTNESS bug and a hard stop.
  * the bias differs: arm A passes `bias=` into `ttnn.linear`, arm B does a separate `ttnn.add_`.

A device-vs-device comparison cannot separate those, which is why the small-shape check passed
and still told us nothing about this. So every variant here is scored against a host float64
reference of the same contraction, at a shape small enough to compute on the host but with
K=1024 KEPT, because K is where it broke.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]

import torch                                                                   # noqa: E402
import ttnn                                                                    # noqa: E402
import tt_bio.tenstorrent as T                                                 # noqa: E402

ROWS, CD, J, COUT = 64, 1024, 512, 128
dev = T.get_device()
kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                 fp32_dest_acc_en=True, packer_l1_acc=True)
grid = T.CORE_GRID_MAIN

torch.manual_seed(0)
zh = torch.randn(ROWS, CD, J) * 0.05
wh = torch.randn(CD, COUT) * 0.05
bh = torch.randn(1, COUT) * 0.05
# Round the operands to bf16 FIRST, so the reference is the reference for these operands and the
# residual reported is the device's own, not float32-vs-bf16 input rounding.
zb = zh.to(torch.bfloat16).float()
wb = wh.to(torch.bfloat16).float()
bb = bh.to(torch.bfloat16).float()
ref = torch.einsum("rcj,co->rjo", zb.double(), wb.double()) + bb.double()
den = ref.abs().max().item()


def dev_t(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


z, w, b = dev_t(zh), dev_t(wh), dev_t(bh)
R = {"shape": [ROWS, CD, J, COUT], "ref_absmax": den, "variants": {}}


def score(name, fn):
    try:
        o = fn()
        ttnn.synchronize_device(dev)
        h = ttnn.to_torch(o).float().double()
        ttnn.deallocate(o)
    except Exception as e:                                                     # noqa: BLE001
        R["variants"][name] = {"error": repr(e), "verdict": "REFUSED"}
        print("%-44s REFUSED %r" % (name, e), flush=True)
        return
    rel = (h - ref).abs().max().item() / den
    R["variants"][name] = {"rel_max_err": rel, "absmax": h.abs().max().item(),
                           "verdict": "OK" if rel < 0.05 else "WRONG"}
    print("%-44s rel_max_err %.6f  %s" % (name, rel, R["variants"][name]["verdict"]), flush=True)


def perm_linear(cg, with_bias):
    def f():
        p = ttnn.permute(z, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kw = {"core_grid": cg} if cg else {}
        o = ttnn.linear(p, w, bias=(b if with_bias else None),
                        compute_kernel_config=ckc, **kw)
        ttnn.deallocate(p)
        return o
    return f


def ta_matmul(cg, with_bias):
    def f():
        kw = {"core_grid": cg} if cg else {}
        o = ttnn.matmul(z, w, transpose_a=True, compute_kernel_config=ckc, **kw)
        return ttnn.add_(o, b) if with_bias else o
    return f


score("A  permute+linear   core_grid  bias", perm_linear(grid, True))
score("A2 permute+linear   no_grid    bias", perm_linear(None, True))
score("A3 permute+linear   core_grid  nobias", perm_linear(grid, False))
score("B  transpose_a      core_grid  bias", ta_matmul(grid, True))
score("B2 transpose_a      no_grid    bias", ta_matmul(None, True))
score("B3 transpose_a      core_grid  nobias", ta_matmul(grid, False))
score("B4 transpose_a      no_grid    nobias", ta_matmul(None, False))
(HERE / "runs").mkdir(parents=True, exist_ok=True)
(HERE / "runs/opm_isolate.json").write_text(json.dumps(R, indent=1))
print("wrote runs/opm_isolate.json")
