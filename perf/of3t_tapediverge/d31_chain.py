#!/usr/bin/env python3
"""D31's gradient half: does the forward's 3.3e-02 deficit buy a gradient difference?

`of3t-rebase` measured the FORWARD gap -- fused 3.25e-02 from float64, the recompute 6.58e-03,
3.33e-02 apart -- and stopped there. The gradient question is different and the code answers
half of it before any run: `autograd.triangle_attention`'s backward reads q, k, v and bias and
recomputes the scores, so AT A SINGLE ISOLATED SITE the gradient does not depend on the forward
value at all and the two arms are identical by construction. The mismatch can only enter through
the activations one site hands the next.

So this is a CHAIN. `--depth` triangle-attention blocks in series, each with its own projection
weights, the shipped taped path throughout:

    arm FUSED  -- `_v_sdpa` as it ships: forward value from the stock fused SDPA
    arm SELF   -- the same tape with `value=None`, so the forward value is the function the
                  backward differentiates

Both arms are ours. No external reference is needed, which is what D31 says about its own
discriminator. What is compared is the parameter and input gradients the two arms produce from
an identical cotangent, plus the forward outputs, so the gradient difference can be read against
the forward difference that caused it.

A float64 CPU arm is computed as well, because "which of the two is closer to right" is a
different question from "how far apart are they" and the pair is cheap here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.getcwd())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8, help="attention sites in series")
    ap.add_argument("--b", type=int, default=32, help="leading axis (S for triangle attention)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="perf/of3t_tapediverge/D31_CHAIN.json")
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    g = torch.Generator().manual_seed(a.seed)
    B, H, N, D = a.b, a.heads, a.n, a.dim
    scale = D ** -0.5

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5

    # One shared set of draws for every arm, so nothing differs but the lever.
    x0 = rnd(B, H, N, D)
    SCALES = [(1.0 + 0.05 * i, 1.0 - 0.03 * i) for i in range(a.depth)]
    bias = rnd(1, H, N, N)
    cot = rnd(B, H, N, D)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def run(self_value: bool):
        """One taped chain, built the way `taped_ttnn._v_sdpa` builds one site.

        `_v_sdpa` calls the stock fused kernel on the RAW arguments, scales the bias on the tape
        with `ag.scale` because `triangle_attention` adds the bias after the scale while the
        kernel adds it before, and hands the fused output in as `value`. Both lines are
        reproduced here rather than approximated: getting the bias convention backwards would
        manufacture the finding this is looking for.

        The taped leaves are the chain INPUT and the BIAS. There are no projection weights,
        because the quantity D31 is about is how far a site's output perturbs what the next
        site differentiates, and that shows up in d/dx and d/dbias without any parameters in
        the way. Each site scales q and k by its own constant so the sites are not identical.
        """
        from tt_bio.taped_ttnn import _sdpa_chunking
        xs = ag.Tensor(to_dev(x0), requires_grad=True)
        bi = ag.Tensor(to_dev(bias), requires_grad=True)
        n_sites = 0
        h = xs
        for i, (aq, ak) in enumerate(SCALES):
            q, k, v = ag.scale(h, aq), ag.scale(h, ak), h
            out_v = None
            if not self_value:
                out_v = ttnn.transformer.scaled_dot_product_attention(
                    q.value, k.value, v.value, attn_mask=bi.value, is_causal=False,
                    scale=scale)
            cB, cQ = _sdpa_chunking(B, H, N, N, 2)
            h = ag.triangle_attention(q, k, v, ag.scale(bi, scale), scale=scale,
                                      chunk=cB, q_chunk=cQ, value=out_v)
            n_sites += 1
        h.backward(seed=to_dev(cot))
        return h.value, {"x": xs.grad, "bias": bi.grad}, n_sites

    out_f, gr_f, n_f = run(False)
    out_s, gr_s, n_s = run(True)

    def t(x):
        return ttnn.to_torch(x).to(torch.float64).reshape(-1)

    def rel(x, y):
        d = float(torch.linalg.vector_norm(x - y))
        n = float(torch.linalg.vector_norm(y))
        return d / n if n else None

    fwd_gap = rel(t(out_f), t(out_s))
    rows = []
    num = den = 0.0
    worst = (0.0, None)
    for k in sorted(gr_f):
        if gr_f[k] is None or gr_s[k] is None:
            rows.append({"name": k, "rel_l2": None, "note": "no gradient on this arm"})
            continue
        u, v = t(gr_f[k]), t(gr_s[k])
        d = float(torch.linalg.vector_norm(u - v))
        n = float(torch.linalg.vector_norm(v))
        num += d * d
        den += n * n
        r = d / n if n else None
        rows.append({"name": k, "rel_l2": r, "norm_self": n})
        if r is not None and r > worst[0]:
            worst = (r, k)

    rep = {
        "instrument": "D31 gradient half: the taped chain's gradient with the fused forward "
                      "against the same chain's gradient with the forward its backward "
                      "differentiates. One arm, both sides ours, no external reference.",
        "shape": {"depth": a.depth, "B": B, "heads": H, "n": N, "dim": D, "scale": scale},
        "sdpa_sites_taped_per_arm": {"fused": n_f, "selfvalue": n_s},
        "forward_rel_l2_fused_vs_selfvalue": fwd_gap,
        "gradient_mass_weighted_rel_l2": math.sqrt(num / den) if den else None,
        "gradient_worst_rel_l2": worst[0],
        "gradient_worst_tensor": worst[1],
        "per_tensor_bar": 0.05,
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
