#!/usr/bin/env python3
"""Is this card fit to host a gradient-correctness campaign?

pc's p150a is a known-faulty card: it computes some ttnn matmuls wrong at a low,
location-keyed, data-independent rate, at every size (memory
``pc-card0-512aa-fold-nondeterminism``, root-caused 2026-08-17). A gradcheck run on such
a card can fail for the card rather than for the formula, and -- worse -- can pass while
a real error hides under a tolerance that was set for bf16 quantisation.

So before any weight gradient is believed, the card gets measured: run the SAME backward
N times on the SAME inputs and compare the repeats to each other, not to a reference. A
correct card is bit-identical across repeats, because nothing here is nondeterministic.
Any nonzero spread is the card's own noise floor, and a gradient difference below it is
not evidence of anything.

The probe targets ``dW`` on purpose. dW is ``X^T @ dY`` over a flattened activation
(autograd.py:243), which reduces over the token axis rather than the channel axis, so it
is a different matmul shape from the forward -- and the recorded fault is shape-keyed.
"""

import argparse
import math
import sys

import numpy as np
import torch
import ttnn

sys.path.insert(0, "/home/moritz/.coworker/wt/ptxft-build")
from tt_bio import autograd as ag  # noqa: E402


def one_pass(device, rounded, dt):
    t = {k: ag.Tensor(ttnn.from_torch(v.to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
                                      dtype=dt, layout=ttnn.TILE_LAYOUT, device=device),
                      requires_grad=True)
         for k, v in rounded.items()}
    out = ag.linear(t["x"], t["w"], t["b"])
    out.backward(seed=ttnn.from_torch(
        rounded["seed"].to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
        dtype=dt, layout=ttnn.TILE_LAYOUT, device=device))
    return {k: ttnn.to_torch(t[k].grad).to(torch.float64).numpy() for k in ("x", "w", "b")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--cin", type=int, default=512)
    ap.add_argument("--cout", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dt = ttnn.bfloat16 if a.dtype == "bfloat16" else ttnn.float32

    rng = np.random.default_rng(a.seed)
    raw = {
        "x": rng.standard_normal((a.tokens, a.cin)),
        "w": rng.standard_normal((a.cin, a.cout)) / math.sqrt(a.cin),
        "b": rng.standard_normal((1, a.cout)) * 0.1,
        "seed": rng.standard_normal((a.tokens, a.cout)),
    }
    torch_dt = torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32
    rounded = {k: torch.from_numpy(v).to(torch_dt).to(torch.float64) for k, v in raw.items()}

    device = ttnn.open_device(device_id=0)
    try:
        runs = [one_pass(device, rounded, dt) for _ in range(a.repeats)]
    finally:
        ttnn.close_device(device)

    print(f"# {a.repeats} repeats, identical inputs, linear({a.tokens}x{a.cin} -> {a.cout}) "
          f"dtype={a.dtype}")
    print(f"# a correct card is bit-identical across repeats; any spread is the card's noise floor")
    print(f"{'param':<6} {'bit-identical':>14} {'max_abs_spread':>15} {'rel_l2_spread':>14} "
          f"{'worst_pair':>11}")
    worst_overall = 0.0
    for k in ("x", "w", "b"):
        base = runs[0][k]
        nrm = np.linalg.norm(base)
        ident, max_abs, rel = True, 0.0, 0.0
        for r in runs[1:]:
            d = r[k] - base
            if np.any(d != 0.0):
                ident = False
            max_abs = max(max_abs, float(np.abs(d).max()))
            rel = max(rel, float(np.linalg.norm(d) / nrm) if nrm > 0 else 0.0)
        worst_overall = max(worst_overall, rel)
        print(f"{k:<6} {str(ident):>14} {max_abs:>15.3e} {rel:>14.3e} "
              f"{'0 vs i':>11}")
    print()
    verdict = ("CLEAN (bit-stable)" if worst_overall == 0.0
               else "NONZERO -- no gradient difference below this is evidence")
    print(f"CARD-NOISE-FLOOR rel_l2 = {worst_overall:.3e}  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
