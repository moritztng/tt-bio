#!/usr/bin/env python3
"""Correctness of the fused SwiGLU kernel against a float64 reference, at the fold's own shape.

Three things are scored, in this order, because they answer different questions:

  * fused vs a float64 reference  -- is the transform RIGHT. This is the hard stop: a fused kernel
    that is wrong rather than imprecise is a NO-GO regardless of what it measures.
  * production vs the same float64 reference -- the known-answer control. Without it a fused error
    of 1e-2 means nothing, because bf16 in and bf16 out has an error of that order anyway.
  * fused vs production, torch.equal -- bit-exactness, kept because it is free and it is the
    cheapest regression signal. NOT a gate: the standing bar is accuracy, not bit-exactness.

The production arm is production's own three ops with silu FUSED into fc1, which is what
`Transition.swiglu` issues unless TT_BIO_UNFUSED_SILU is set. c14-radical's ladder used the
unfused `ttnn.silu`, so the unfused arm is scored too and both are reported.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402

from tt_bio import mm_generic as MG                                   # noqa: E402
from tt_bio import swiglu_fused as SW                                 # noqa: E402

B, M, K, H = 16, 512, 128, 512
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
CG = ttnn.CoreGrid(y=g.y, x=g.x)
GRID = (g.x, g.y)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
x_t = torch.randn(1, B, M, K, dtype=torch.bfloat16)
w1_t = torch.randn(K, H, dtype=torch.bfloat16)
w2_t = torch.randn(K, H, dtype=torch.bfloat16)


def to_tt(x, mc=L1):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


xn = to_tt(x_t)
w1, w2 = to_tt(w1_t, DRAM), to_tt(w2_t, DRAM)

# float64 reference on the exact bf16 operand values the device sees.
x64, w164, w264 = x_t.double(), w1_t.double(), w2_t.double()
ref = (torch.nn.functional.silu(x64 @ w164) * (x64 @ w264)).squeeze(0)


def prod(fused_silu: bool):
    a = ttnn.linear(xn, w1, activation="silu" if fused_silu else None,
                    compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.bfloat16,
                    core_grid=CG)
    if not fused_silu:
        a = ttnn.silu(a, memory_config=L1, output_tensor=a)
    b = ttnn.linear(xn, w2, compute_kernel_config=ckc, memory_config=L1,
                    dtype=ttnn.bfloat16, core_grid=CG)
    out = ttnn.multiply_(a, b)
    ttnn.deallocate(b)
    return out


print(f"eligible: {SW.eligible(xn, w1, w2)!r}  grid={GRID}  block={SW._block(w1)}", flush=True)

fused = SW.fused_swiglu(xn, w1, w2, MG.ckc_args(ckc), GRID)
assert fused is not None, SW.REJECTS
f_t = ttnn.to_torch(fused).squeeze(0).float()
p_fused_silu = prod(True)
pf_t = ttnn.to_torch(p_fused_silu).squeeze(0).float()
p_unfused = prod(False)
pu_t = ttnn.to_torch(p_unfused).squeeze(0).float()


def score(name, got):
    d = (got.double() - ref).abs()
    scale = ref.abs().mean().item()
    return {"arm": name, "max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "mean_abs_over_mean_ref": d.mean().item() / scale,
            "rel_l2": (d.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()}


out = {
    "shape": [1, B, M, K], "hidden": H, "grid": list(GRID),
    "block": list(SW._block(w1)), "round": SW.ROUND,
    "ref_mean_abs": ref.abs().mean().item(), "ref_max_abs": ref.abs().max().item(),
    "vs_float64": [score("fused", f_t), score("prod_fused_silu", pf_t),
                   score("prod_unfused_silu", pu_t)],
    "bit_exact_fused_vs_prod_fused_silu": bool(torch.equal(f_t, pf_t)),
    "bit_exact_fused_vs_prod_unfused_silu": bool(torch.equal(f_t, pu_t)),
    "bit_exact_prod_fused_vs_prod_unfused": bool(torch.equal(pf_t, pu_t)),
    "elems_differing_fused_vs_prod_fused_silu": int((f_t != pf_t).sum().item()),
    "n_elems": f_t.numel(),
    "fused_out_memcfg": str(fused.memory_config()),
}
ttnn.close_device(dev)
(HERE / "parity_qb1n0.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
