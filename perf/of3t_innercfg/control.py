#!/usr/bin/env python3
"""The control for VJP.json's zero: does a compute kernel config change `ttnn.sum` AT ALL?

`vjp.py` measured the fix arm bit-identical to the shipped arm on every shape and on the
whole triangle-attention VJP. Two things produce that reading and they have opposite
conclusions: the config makes no difference to this reduction on these operands, or the
config never reached the reduction. An arm that cannot move is not evidence about the
variable, so this moves it on purpose.

  A  a reduction built so bf16 accumulation MUST lose: 4096 equal addends, exact sum known.
  B  the same reduction at the trunk's own axis length, 384.
  C  `precise_config()` on a matmul, where HiFi4 is known to change the answer. This is the
     break control for the CONFIG OBJECT: if C does not move either, the object is the
     defect and nothing about `ttnn.sum` has been shown.
  D  the DENOMINATOR sum of `softmax_bw_inner`, the one the shipped code does configure,
     with and without. If D is also flat the whole `config` parameter is cosmetic, not
     just the numerator's missing one.
"""
from __future__ import annotations

import json, pathlib, platform, subprocess, sys, time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio  # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT)

import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.autograd import precise_config  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def d(t):
    return ttnn.to_torch(t).double()


def main():
    dev = get_device()
    cfg = precise_config()
    rep = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=platform.node(),
               commit=subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               ttnn_sum_doc_has_ckc="compute_kernel_config" in (ttnn.sum.__doc__ or ""))
    cases = {}

    # A / B: equal addends. Exact sum is N; bf16 sequential accumulation cannot reach it.
    for tag, N in (("A_ones_4096", 4096), ("B_ones_384", 384)):
        x = torch.ones(1, 1, 32, N, dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        off = d(ttnn.sum(xt, dim=-1, keepdim=True))
        on = d(ttnn.sum(xt, dim=-1, keepdim=True, compute_kernel_config=cfg))
        cases[tag] = dict(exact=float(N), off=float(off.flatten()[0]), on=float(on.flatten()[0]),
                          bitwise_equal=bool(torch.equal(off, on)),
                          max_abs_off_minus_on=float((off - on).abs().max()))
        ttnn.deallocate(xt)

    # A2: graded magnitudes, the shape an inner product of a gradient and a probability has --
    # one big term and many small ones, which is exactly what swallows in bf16.
    torch.manual_seed(3)
    x = (torch.randn(1, 1, 32, 4096) * 0.01).to(torch.bfloat16)
    x[..., 0] = 64.0
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    off = d(ttnn.sum(xt, dim=-1, keepdim=True))
    on = d(ttnn.sum(xt, dim=-1, keepdim=True, compute_kernel_config=cfg))
    ref = x.double().sum(-1, keepdim=True)
    cases["A2_one_big_many_small"] = dict(
        exact=float(ref.flatten()[0]), off=float(off.flatten()[0]), on=float(on.flatten()[0]),
        bitwise_equal=bool(torch.equal(off, on)),
        off_rel_err=float((off - ref).abs().max() / ref.abs().max()),
        on_rel_err=float((on - ref).abs().max() / ref.abs().max()))
    ttnn.deallocate(xt)

    # C: the config object on a matmul. This one is expected to MOVE.
    torch.manual_seed(5)
    a = (torch.randn(1, 1, 256, 512) * 0.5).to(torch.bfloat16)
    b = (torch.randn(1, 1, 512, 256) * 0.5).to(torch.bfloat16)
    at = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    bt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    m_off = d(ttnn.matmul(at, bt))
    m_on = d(ttnn.matmul(at, bt, compute_kernel_config=cfg))
    ref = a.double() @ b.double()
    nrm = torch.linalg.vector_norm
    cases["C_matmul_break_control"] = dict(
        bitwise_equal=bool(torch.equal(m_off, m_on)),
        off_rel_l2=float(nrm(m_off - ref) / nrm(ref)),
        on_rel_l2=float(nrm(m_on - ref) / nrm(ref)))
    ttnn.deallocate(at); ttnn.deallocate(bt)

    # D: the denominator the shipped code DOES configure, on a real exact-softmax y.
    torch.manual_seed(17)
    s = torch.randn(8, 4, 384, 384) * 3.0
    y64 = torch.softmax(s.to(torch.bfloat16).double(), dim=-1)
    yt = ttnn.from_torch(y64.to(torch.bfloat16), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    r_off = d(ttnn.sum(yt, dim=-1, keepdim=True))
    r_on = d(ttnn.sum(yt, dim=-1, keepdim=True, compute_kernel_config=cfg))
    r_ref = ttnn.to_torch(yt).double().sum(-1, keepdim=True)
    cases["D_shipped_denominator"] = dict(
        bitwise_equal=bool(torch.equal(r_off, r_on)),
        off_rel_l2=float(nrm(r_off - r_ref) / nrm(r_ref)),
        on_rel_l2=float(nrm(r_on - r_ref) / nrm(r_ref)),
        off_mean=float(r_off.mean()), on_mean=float(r_on.mean()))
    ttnn.deallocate(yt)

    # E: does the OUTPUT DTYPE of the reduction move it? The config sets the accumulator,
    # `dtype` sets what is written back and what the product was rounded to.
    torch.manual_seed(17)
    g = torch.randn(8, 4, 384, 384).to(torch.bfloat16)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    yt = ttnn.from_torch(y64.to(torch.bfloat16), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    prod_bf = ttnn.multiply(gt, yt)
    prod_f32 = ttnn.multiply(gt, yt, dtype=ttnn.float32)
    ref_inner = (d(gt) * d(yt)).sum(-1, keepdim=True)
    e = {}
    for pname, pt in (("product_bf16", prod_bf), ("product_fp32", prod_f32)):
        for cname, kw in (("no_cfg", {}), ("precise", {"compute_kernel_config": cfg})):
            r = d(ttnn.sum(pt, dim=-1, keepdim=True, **kw))
            e[f"{pname}/{cname}"] = float(nrm(r - ref_inner) / nrm(ref_inner))
    cases["E_product_dtype_vs_config"] = e
    rep["cases"] = cases
    p = pathlib.Path(_ROOT) / "perf/of3t_innercfg/CONTROL.json"
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
