#!/usr/bin/env python3
"""The fused score scale, against a float64 reference that finite differences validated first.

`_fp32_softmax_tail` (tenstorrent.py:3766) rides the attention score scale on the add's
input-a activation rather than paying a pass for it:

    attn = ttnn.add_(sc, bias_f, input_tensor_a_activations=[MUL_UNARY_SFPU(1/sqrt(d))])

which is `sc * c + bias_f`. openfold3's trunk, template and MSA stacks all set
`fp32_softmax=True`, so this is on the shipped path for the whole model, and until this row
the tape had no entry for a parameterised fused activation and refused. Protenix never takes
this path, which is why `ptx-fastpath`'s 331/331 never met it.

Two things are checked, and the second is the one that could have gone wrong quietly:

1. The gradient. `dL/dsc` carries the factor `c` and `dL/dbias_f` does not. Dropping `c`
   would leave the forward exactly right -- it comes from the shipped verb either way -- and
   scale one gradient by 1/c, which at OF3's head_dim 32 is 5.66x. That is the same shape of
   defect as the fused sigmoid gate `ptx-fastpath` measured at 4.88x high.
2. The constant itself. ttnn binds `op_type` on `UnaryWithParam` and not `params`, so the
   tape reads the scalar out of the repr. The negative control below is a reference built
   with c = 1, which is exactly what a parse returning the wrong thing would produce.

Method, PROTOCOL §3c: the reference is float64 and is validated against float64 central
finite differences BEFORE anything is compared to it, and the forward is checked first --
a reference that disagrees with the forward can still agree with a wrong gradient, which is
how the fused SDPA's mask-before-scale ordering cost `ptx-fastpath` a verification pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "ptx_fastpath"))

BAR = 5.0e-2          # PROTOCOL 3d, per-tensor relative L2
FD_BAR = 1.0e-8       # the reference against its own central difference, in float64


def ref_loss(a, b, w, c):
    """L = sum(w * (a*c + b)^2), in float64. The square is what makes the gradient depend
    on the value, so a wrong `c` shows up in the magnitude as well as in the slope."""
    return float((w * (a * c + b) ** 2).sum())


def ref_grads(a, b, w, c):
    t = 2.0 * w * (a * c + b)
    return t * c, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import tenstorrent as tt
    from tt_bio.taped_ttnn import taped_ttnn
    from modulecheck import clocks

    c = 1.0 / math.sqrt(a.head_dim)
    rng = np.random.default_rng(a.seed)
    shape = (1, a.heads, a.tokens, a.tokens)
    a64 = rng.standard_normal(shape)
    b64 = rng.standard_normal(shape) * 0.3
    w64 = rng.standard_normal(shape)

    dev = tt.get_device()
    D = lambda x: ttnn.from_torch(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    H = lambda t: np.asarray(ttnn.to_torch(t).to(torch.float64))

    # The operands as the DEVICE sees them, so the reference differentiates the same numbers
    # production does rather than the fp64 ones it was handed.
    a_dev, b_dev, w_dev = D(a64), D(b64), D(w64)
    ad, bd, wd = H(a_dev), H(b_dev), H(w_dev)

    stop, clk = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, clk), daemon=True)
    th.start()
    out = {"c": c, "shape": list(shape)}
    try:
        # -- 0. the reference against central finite differences, in float64 --------------
        ga_ref, gb_ref = ref_grads(ad, bd, wd, c)
        da = rng.standard_normal(shape)
        da /= np.linalg.norm(da)
        eps = 1e-5
        num = (ref_loss(ad + eps * da, bd, wd, c)
               - ref_loss(ad - eps * da, bd, wd, c)) / (2 * eps)
        ana = float((ga_ref * da).sum())
        fd_rel = abs(num - ana) / max(abs(ana), 1e-300)
        out["reference_vs_fd"] = fd_rel
        print(f"reference vs float64 central difference: rel {fd_rel:.3e} "
              f"against a {FD_BAR:.0e} bar  {'ok' if fd_rel <= FD_BAR else 'FAIL'}")
        if fd_rel > FD_BAR:
            print("VERDICT: FAIL -- the reference is not validated, nothing below counts")
            return 1

        # -- 1. the FORWARD first ----------------------------------------------------------
        op = ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, c)
        fwd = ttnn.add(a_dev, b_dev, input_tensor_a_activations=[op])
        fh = H(fwd)
        want = ad * c + bd
        f_rel = float(np.linalg.norm(fh - want) / (np.linalg.norm(want) + 1e-300))
        wrong = ad + bd                       # the ordering a dropped scale would compute
        f_rel_wrong = float(np.linalg.norm(fh - wrong) / (np.linalg.norm(wrong) + 1e-300))
        out["forward_rel"] = f_rel
        out["forward_rel_if_scale_dropped"] = f_rel_wrong
        print(f"forward: kernel vs `a*c + b` rel {f_rel:.3e}; vs `a + b` rel "
              f"{f_rel_wrong:.3e}  -- the reference models the kernel")
        ttnn.deallocate(fwd)
        if f_rel > 1e-2 or f_rel_wrong < f_rel * 10:
            print("VERDICT: FAIL -- the forward does not match the reference's function")
            return 1

        # -- 2. the gradient, through the tape over the shipped verb -----------------------
        A = ag.Tensor(a_dev, requires_grad=True)
        B = ag.Tensor(b_dev, requires_grad=True)
        with ag.tape():
            tnn = taped_ttnn()
            y = tnn.add(A, B, input_tensor_a_activations=[op])
            y2 = tnn.multiply(y, y)
        y2.backward(seed=w_dev)
        ga, gb = H(A.grad), H(B.grad)

        def rel(got, want):
            return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300))

        ra, rb = rel(ga, ga_ref), rel(gb, gb_ref)
        out["d_scores"], out["d_bias"] = ra, rb
        print(f"d(scores) rel L2 {ra:.3e}    d(bias) rel L2 {rb:.3e}   "
              f"against a {BAR:.1e} bar")

        # -- 3. the control: a reference that ignored the fused scale ----------------------
        ga_bad, gb_bad = ref_grads(ad, bd, wd, 1.0)
        rab, rbb = rel(ga, ga_bad), rel(gb, gb_bad)
        out["control_d_scores"], out["control_d_bias"] = rab, rbb
        rejected = rab > BAR
        print(f"negative control (the same reference with c = 1, which is what a mis-read "
              f"scalar gives): d(scores) rel {rab:.3e}, d(bias) rel {rbb:.3e} "
              f"{'-- rejected' if rejected else '-- CHECK IS BLIND'}")

        ok = ra <= BAR and rb <= BAR and rejected
    finally:
        stop.set()
        th.join(timeout=3)
    if clk:
        out["aiclk"] = [min(clk), max(clk), len(clk)]
        print(f"aiclk DURING: min {min(clk)} max {max(clk)} MHz ({len(clk)} samples)")
    out["verdict"] = "PASS" if ok else "FAIL"
    json.dump(out, open("perf/of3t_tape/fusedscale.json", "w"), indent=1)
    print("VERDICT:", out["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
