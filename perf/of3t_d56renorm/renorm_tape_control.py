#!/usr/bin/env python3
"""The positive control for D56's zeros: make the renorm counters fire.

`renorm_blast_radius.py` claims a fold leaves `SOFTMAX_BW_RENORM_STATS` at 0/0. A counter
that is simply never incremented anywhere would report the same thing, so that zero is
uninformative until something drives it non-zero on the same tree, in the same build.

This does it in the smallest way that is still the real path: open the tape, softmax through
the shim, `backward()`, read the counters. Both flag values, because `applied` and `declined`
are different branches and each has to be shown reachable.

Run one flag value per process: the flag is read at import time, so an in-process A/B would
read whatever the first import saw (`eligibility-firing-condition-is-not-a-code-fact`).
"""
import json
import os
import sys

import torch
import ttnn


def main():
    out = sys.argv[1]
    want = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM")

    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as TT

    # Read through BOTH names. `taped_ttnn._SOFTMAX_BW_RENORM` is an alias of
    # `autograd.SOFTMAX_BW_RENORM` now, and recording them separately is how a future second
    # parse of the variable shows up here instead of being silently tolerated.
    rep = {"env_flag": want, "module_read": bool(ag.SOFTMAX_BW_RENORM),
           "alias_agrees": bool(TT._SOFTMAX_BW_RENORM) == bool(ag.SOFTMAX_BW_RENORM),
           "before": dict(ag.SOFTMAX_BW_RENORM_STATS)}

    dev = ttnn.open_device(device_id=0)
    try:
        # Seeded BEFORE the input exists, so the two flag arms are the same values and the
        # residual below is a matched A/B rather than two draws.
        torch.manual_seed(20260921)
        t = torch.randn(1, 2, 64, 64)
        v = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        x = ag.Tensor(v, requires_grad=True)
        with TT.tape():
            shim = TT.taped_ttnn()
            y = shim.softmax(x, dim=-1)
        # A seed of ones makes this gradient identically zero -- sum(softmax(x)) is the
        # constant 1, so d/dx of it vanishes and the run would report a fired counter on an
        # arithmetically empty backward. Seed non-uniformly so the row-sum invariant the
        # repair restores is actually exercised.
        seed = ttnn.from_torch(torch.randn(1, 2, 64, 64), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
        y.backward(seed)
        rep["after"] = dict(ag.SOFTMAX_BW_RENORM_STATS)
        rep["grad_is_present"] = x.grad is not None
        if x.grad is not None:
            g = ttnn.to_torch(x.grad).double()
            rep["grad_absmax"] = float(g.abs().max())
            # The invariant the repair exists to restore: `d_logits = y (g - sum g y)` has
            # rows summing to zero exactly when the softmax row sums to one. On bf16 it does
            # not, and this is the residual, in units of the gradient it rides on.
            rep["row_sum_absmax"] = float(g.sum(dim=-1).abs().max())
            rep["row_sum_rel"] = float(g.sum(dim=-1).abs().max() / g.abs().max())
            y64 = ttnn.to_torch(y.value if hasattr(y, "value") else y).double()
            rep["softmax_row_sum_dev_from_one"] = float((y64.sum(dim=-1) - 1.0).abs().max())
    finally:
        ttnn.close_device(dev)

    key = "applied" if ag.SOFTMAX_BW_RENORM else "declined"
    rep["counter_fired"] = rep.get("after", {}).get(key, 0) > rep["before"].get(key, 0)
    rep["verdict"] = "PASS" if (rep["counter_fired"] and rep["alias_agrees"]) else "FAIL"
    print(json.dumps(rep, indent=1, sort_keys=True))
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    return 0 if rep["counter_fired"] else 1


if __name__ == "__main__":
    sys.exit(main())
