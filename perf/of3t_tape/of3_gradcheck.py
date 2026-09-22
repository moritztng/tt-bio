#!/usr/bin/env python3
"""OpenFold3's own modules, by finite differences through their own forward.

`perf/of3t_tape/fusedscale.py` verifies the one new op against a float64 reference that
central differences validated first. What that cannot see is the composition: whether the
tape wires OF3's modules together correctly through the fp32-softmax tail, the sub-tile
head_dim the template stack runs at, the per-template loop and the slot aggregation.

A hand-written float64 reference for a 300-line tuned module is itself the most likely
thing to be wrong, so this does not write one. It asks production's own forward, which is
`perf/ptx_fastpath/modulecheck.py`'s method and this file reuses its sweep and its clock
sampler rather than restating them:

    analytic = <dL/dx, d>                                        from the tape
    numeric  = (L(x + eps d) - L(x - eps d)) / 2 eps             both forwards UNTAPED

`d` is the gradient's own direction at max-norm one, because a random direction in half a
million dimensions is nearly orthogonal to the gradient and an L2-unit direction puts a
per-element step below bf16's resolution, after which the cast deletes the perturbation and
the sweep reads rounding artifacts that look exactly like a systematic gradient error. Both
readings are printed; the random one is the weaker.

The modules, all three OF3-specific:

  triangle_attention_fp32   `fp32_softmax=True`, which is the path every OF3 pair stack
                            takes and no Protenix module does. It is the module that
                            carries the fused MUL_UNARY_SFPU score scale.
  template_block            `PairformerLayer` exactly as `TemplatePairStack` builds it, from
                            the shipped checkpoint: head_dim 16 (sub-tile), 4 heads, c_t 64,
                            `scale_pair_bias=False`, `fp32_softmax=True`.
  template_embedder         the whole shipped `TemplateEmbedder`, real weights, real
                            features: eight feature linears, the z bias, the two-block pair
                            stack per template, the slot aggregation, relu and `linear_t`.

A negative control runs the same comparison against a gradient scaled by 1.5, so a pass
means the check could have failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "ptx_fastpath"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
EPS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="template_block",
                    choices=("triangle_attention_fp32", "template_block",
                             "template_embedder"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bar", type=float, default=2.0e-1,
                    help="LEDGER K6's bar for a finite difference through a bf16 forward")
    a = ap.parse_args()

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import tenstorrent as tt
    from modulecheck import clocks

    rng = np.random.default_rng(a.seed)
    dev = tt.get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    D = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)

    if a.module == "triangle_attention_fp32":
        c_t, heads, head_dim, S = 64, 4, 16, a.tokens
        t = lambda *sh: torch.tensor(rng.standard_normal(sh) / np.sqrt(sh[-1]),
                                     dtype=torch.float32)
        sd = {"layer_norm.weight": torch.tensor(rng.standard_normal(c_t) * 0.2 + 1.0,
                                                dtype=torch.float32),
              "layer_norm.bias": torch.tensor(rng.standard_normal(c_t) * 0.05,
                                              dtype=torch.float32),
              "linear_q.weight": t(heads * head_dim, c_t),
              "linear_k.weight": t(heads * head_dim, c_t),
              "linear_v.weight": t(heads * head_dim, c_t),
              "linear_g.weight": t(heads * head_dim, c_t),
              "linear_o.weight": t(c_t, heads * head_dim),
              "linear.weight": t(heads, c_t)}
        module = tt.TriangleAttention(head_dim, heads, False, sd, ckc,
                                      scale_pair_bias=False, fp32_softmax=True)
        in_shape = (1, S, S, c_t)
        x0 = torch.tensor(rng.standard_normal(in_shape), dtype=torch.float32)
    else:
        from tt_bio.openfold3_host_prep import derive_template_feat, dedup_template_slots
        from tt_bio.openfold3_weights import _sub, is_openbind, remap_template_pair_stack
        import of3_golden
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        inter = of3_golden.intermediates(GOLD)
        feats = inter["input_embedder_real"]["in"]
        _, _, z_init = inter["input_embedder_real"]["out"]
        tmpl, slots = dedup_template_slots(derive_template_feat(feats))
        tb = not is_openbind(sd)

        if a.module == "template_block":
            remap = remap_template_pair_stack(_sub(sd, "template_embedder"),
                                              prefix="template_pair_stack")
            blk = tt.PairformerLayer(
                16, 4, None, None, False, remap["blocks"][0], ckc,
                scale_pair_bias=False, fp32_softmax=True, transpose_bias=tb,
                accurate_softmax=tt.accurate_softmax_site("openfold3.template"))

            class ZOnly:
                """`PairformerLayer` returns (s, z); the template stack reads the z track."""
                def __call__(self, z):
                    return blk(None, z, None, None, None)[1]

            module = ZOnly()
            # The block's input is the feature embedder's `t_embed`, and it has to BE that
            # rather than a random tensor of the same shape. Measured with a unit-normal
            # input at this scale: |dL/dx| reads 127085 and the finite difference falls off
            # like 1/eps -- 7.3e6 at 0.02 against an analytic 4.55e7, then 1.7e4 at 1.0 --
            # which is a forward driven into saturation, not a gradient that disagrees. The
            # real `t_embed` is two orders smaller and the same check reads cleanly.
            from tt_bio.openfold3_template import TemplatePairFeatureEmbedder
            fe = TemplatePairFeatureEmbedder(
                _sub(sd, "template_embedder.template_pair_embedder"), ckc)
            t_embed, _ = fe({k: D(v.float()) for k, v in tmpl.items()},
                            D(z_init.unsqueeze(0).float()))
            x0 = ttnn.to_torch(t_embed)[:1].float()
            in_shape = tuple(int(v) for v in x0.shape)
        else:
            from tt_bio.openfold3_template import TemplateEmbedder
            emb = TemplateEmbedder(_sub(sd, "template_embedder"), ckc, transpose_bias=tb)
            feat_d = {k: D(v.float()) for k, v in tmpl.items()}

            class VaryZ:
                """The embedder's own input is the cycle's pair track, so that is what is
                varied; the features are the batch's and are held fixed."""
                def __call__(self, z):
                    return emb(feat_d, z, slots)

            module = VaryZ()
            x0 = z_init.unsqueeze(0).float()
            in_shape = tuple(int(v) for v in x0.shape)

    d_rand = torch.tensor(rng.standard_normal(in_shape), dtype=torch.float32)
    stop, clk = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, clk), daemon=True)
    th.start()
    out = {"module": a.module, "shape": list(in_shape), "bar": a.bar}
    try:
        xa = ag.Tensor(D(x0), requires_grad=True)
        with ag.tape():
            res = module(xa)
        lw = torch.tensor(rng.standard_normal([int(v) for v in res.value.shape]),
                          dtype=torch.float32)
        res.backward(seed=D(lw))
        g = ttnn.to_torch(xa.grad).to(torch.float64)
        print(f"{a.module}  input {in_shape}")
        print(f"  |dL/dx| {float(g.norm()):.4f}")
        out["grad_norm"] = float(g.norm())

        def sweep(d, label):
            analytic = float((g * d.to(torch.float64)).sum())
            moved = ((x0 + EPS[0] * d).to(torch.bfloat16)
                     != x0.to(torch.bfloat16)).float().mean().item()

            def loss_at(eps):
                o = module(D(x0 + eps * d))
                return float((ttnn.to_torch(o).to(torch.float64)
                              * lw.to(torch.float64)).sum())

            best, rows = None, []
            for eps in EPS:
                num = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps)
                rel = abs(num - analytic) / max(abs(analytic), 1e-12)
                rows.append((eps, num, rel))
                if best is None or rel < best[2]:
                    best = (eps, num, rel)
            print(f"  direction: {label}")
            print(f"    analytic <dL/dx, d> = {analytic:.6f}   surviving the bf16 cast at "
                  f"eps {EPS[0]}: {moved*100:.1f}%")
            for eps, num, rel in rows:
                print(f"    eps {eps:5.3f}  numeric {num:14.6f}  rel {rel:.3e}"
                      f"{'   <- best' if best[0] == eps else ''}")
            return analytic, best, rows

        # The probe direction is the sign of the gradient, not the gradient itself. Both are
        # max-norm one, but a gradient over a [1,76,76,128] pair track is peaked: at eps 0.02
        # only 3.5 % of its elements move enough to survive the cast to bf16, and the sweep
        # then reads a difference built from a twentieth of the tensor and scatters by 2x
        # across eps -- wide enough that a negative control scaled by 1.5 lands inside it and
        # the check goes blind. Measured here on the template embedder before this line
        # existed. `sign(g)` moves EVERY element by eps, so the perturbation survives
        # everywhere, and its directional derivative is |g| in L1, which is the largest
        # reading available. The gradient's own direction is kept underneath as the second
        # reading and random as the third.
        analytic, best, rows = sweep(torch.sign(g).to(torch.float32),
                                     "the sign of the gradient (every element moves)")
        sweep((g / g.abs().max()).to(torch.float32),
              "the gradient itself (peaked, the weaker reading)")
        sweep(d_rand / d_rand.abs().max(),
              "random (nearly orthogonal, the weakest reading)")
        ok = best[2] <= a.bar
        out["best_rel"], out["best_eps"] = best[2], best[0]
        print(f"  best-of-sweep rel {best[2]:.3e} at eps {best[0]} against a {a.bar:.1e} "
              f"bar  {'PASS' if ok else 'FAIL'}")

        bad = analytic * 1.5
        badrel = min(abs(n - bad) / max(abs(bad), 1e-12) for _, n, _ in rows)
        out["control_rel"] = badrel
        fails = badrel > a.bar
        print(f"  negative control (analytic scaled 1.5x): best rel {badrel:.3e} "
              f"{'-- rejected, the check can fail' if fails else '-- CHECK IS BLIND'}")
        ok = ok and fails
    finally:
        stop.set()
        th.join(timeout=3)
    if clk:
        out["aiclk"] = [min(clk), max(clk), len(clk)]
        print(f"  aiclk DURING: min {min(clk)} max {max(clk)} MHz ({len(clk)} samples)")
    out["verdict"] = "PASS" if ok else "FAIL"
    json.dump(out, open(f"perf/of3t_tape/gradcheck_{a.module}.json", "w"), indent=1)
    print("VERDICT:", out["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
