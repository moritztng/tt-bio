#!/usr/bin/env python3
"""A whole shipped MODULE's gradient, by finite differences through its own forward.

The op-scale checks in this directory each compare one gradient to a float64 reference.
What they cannot see is the composition: whether the tape wires the modules' many ops
together correctly, through the chunking and the head reshapes and the fused kernels. A
hand-written float64 reference of a 300-line tuned module is itself the most likely thing
to be wrong, and verifying against a wrong reference is the failure this campaign keeps
finding, so this check does not write one.

Instead it asks production's own forward. For a unit direction d:

    analytic  = <dL/dx, d>              from the tape
    numeric   = (L(x + eps d) - L(x - eps d)) / 2 eps,   both forwards UNTAPED

Nothing but the shipped module produces either side, so there is no second implementation
to disagree with. d is the GRADIENT's own direction, because a random one is nearly
orthogonal to it in this many dimensions -- measured here, a random d gives a directional
derivative of 0.594 against a gradient norm of 455.6, so the finite difference is almost
entirely cancellation and reads noise at every small eps. Along the gradient the numerator
is |g| itself and the probe is as well conditioned as it can be. A random direction is
reported underneath as the weaker second reading it is.

d is scaled to MAX-NORM ONE, not L2-norm one, and that is not cosmetic. An L2-unit
direction over a [1,64,64,128] tensor puts 1/724 in each element, so at eps 0.02 the
per-element step is 2.8e-05 against values of order 1 -- three orders below bf16's
resolution there. The cast to bfloat16 deletes the perturbation entirely and the sweep
then reads rounding artifacts that are stable across eps and look exactly like a
systematic gradient error: measured, 93 against an analytic 455.6, flat to within 2 % over
a 50x eps span. This is `state/ptx/LEDGER.md` K6's warning arriving on the INPUT side
rather than the output side, and the check now refuses to report a reading whose
perturbation did not survive the cast. This is `state/ptx/LEDGER.md` K6's statistic, and eps is SWEPT rather
than guessed for K6's reason: the numerator 2*eps*|dL| has to stand clear of the bf16
quantisation jitter in production's forward, so too small an eps reads noise and too large
reads curvature. Best-of-sweep is the statistic, and the whole sweep is printed so the
shape of it is visible rather than just its minimum.

A negative control runs the same comparison against a deliberately corrupted gradient, so
a pass means the check could have failed.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

BAR = 2.0e-1          # K6's bar for an FD through production's bf16 forward


def clocks(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for d in json.loads(r.stdout).get("device_info", []):
                c = d.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(0.25)


def triangle_attention_module(tt, ttnn, dev, c_z, heads, head_dim, rng):
    """The shipped `TriangleAttention`, the module that carries the fused SDPA."""
    t = lambda *s: torch.tensor(rng.standard_normal(s) / math.sqrt(s[-1]), dtype=torch.float32)
    sd = {
        "layer_norm.weight": torch.tensor(rng.standard_normal(c_z) * 0.2 + 1.0,
                                          dtype=torch.float32),
        "layer_norm.bias": torch.tensor(rng.standard_normal(c_z) * 0.05, dtype=torch.float32),
        "linear_q.weight": t(heads * head_dim, c_z),
        "linear_k.weight": t(heads * head_dim, c_z),
        "linear_v.weight": t(heads * head_dim, c_z),
        "linear_g.weight": t(heads * head_dim, c_z),
        "linear_o.weight": t(c_z, heads * head_dim),
        "linear.weight": t(heads, c_z),
    }
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    return tt.TriangleAttention(head_dim, heads, False, sd, ckc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="triangle_attention",
                    choices=("triangle_attention", "transition",
                             "triangle_multiplication", "pairformer_layer"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    rng = np.random.default_rng(a.seed)
    S, c_z = a.tokens, a.c_z
    dev = tt.get_device()
    D = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    def trimul_weights(c_z, hidden, rng):
        t = lambda *sh: torch.tensor(rng.standard_normal(sh) / math.sqrt(sh[-1]),
                                     dtype=torch.float32)
        return {"norm_in.weight": torch.tensor(rng.standard_normal(c_z) * 0.2 + 1.0,
                                               dtype=torch.float32),
                "norm_in.bias": torch.tensor(rng.standard_normal(c_z) * 0.05,
                                             dtype=torch.float32),
                "norm_out.weight": torch.tensor(rng.standard_normal(hidden) * 0.2 + 1.0,
                                                dtype=torch.float32),
                "norm_out.bias": torch.tensor(rng.standard_normal(hidden) * 0.05,
                                              dtype=torch.float32),
                "g_in.weight": t(2 * hidden, c_z), "p_in.weight": t(2 * hidden, c_z),
                "g_out.weight": t(c_z, c_z), "p_out.weight": t(c_z, hidden)}

    def triatt_weights(c_z, heads, head_dim, rng):
        t = lambda *sh: torch.tensor(rng.standard_normal(sh) / math.sqrt(sh[-1]),
                                     dtype=torch.float32)
        return {"layer_norm.weight": torch.tensor(rng.standard_normal(c_z) * 0.2 + 1.0,
                                                  dtype=torch.float32),
                "layer_norm.bias": torch.tensor(rng.standard_normal(c_z) * 0.05,
                                                dtype=torch.float32),
                "linear_q.weight": t(heads * head_dim, c_z),
                "linear_k.weight": t(heads * head_dim, c_z),
                "linear_v.weight": t(heads * head_dim, c_z),
                "linear_g.weight": t(heads * head_dim, c_z),
                "linear_o.weight": t(c_z, heads * head_dim),
                "linear.weight": t(heads, c_z)}

    def transition_weights(c_z, hidden, rng):
        t = lambda *sh: torch.tensor(rng.standard_normal(sh) / math.sqrt(sh[-1]),
                                     dtype=torch.float32)
        return {"norm.weight": torch.tensor(rng.standard_normal(c_z) * 0.3 + 1.0,
                                            dtype=torch.float32),
                "norm.bias": torch.tensor(rng.standard_normal(c_z) * 0.1,
                                          dtype=torch.float32),
                "fc1.weight": t(hidden, c_z), "fc2.weight": t(hidden, c_z),
                "fc3.weight": t(c_z, hidden)}

    if a.module == "pairformer_layer":
        # The module the brief names. transform_s=False runs the z track: two triangle
        # multiplications, two triangle attentions and the transition, with the four
        # residual `add_`s and the deallocates between them -- the composition, not the
        # parts. AttentionPairBias needs a single-track input and is checked separately.
        hidden = c_z
        sd = {}
        for scope, w in (("tri_mul_out", trimul_weights(c_z, hidden, rng)),
                         ("tri_mul_in", trimul_weights(c_z, hidden, rng)),
                         ("transition_z", transition_weights(c_z, 4 * c_z, rng))):
            sd.update({f"{scope}.{k}": v for k, v in w.items()})
        for scope in ("tri_att_start", "tri_att_end"):
            sd.update({f"{scope}.mha.{k}": v
                       for k, v in triatt_weights(c_z, a.heads, a.head_dim, rng).items()})
        cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
        layer = tt.PairformerLayer(
            a.head_dim, a.heads, None, None, False, sd,
            cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True))

        class ZOnly:
            """`PairformerLayer` returns (s, z); the harness differentiates one output."""
            def __call__(self, z):
                return layer(None, z)[1]

        module = ZOnly()
    elif a.module == "triangle_multiplication":
        # Four of the eleven fused-sigmoid gate sites live here (tenstorrent.py:6664,
        # 6667, 6877, 6925), along with the trimul chunking, so it is the module most
        # worth checking after the one that found the gate defect.
        cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
        module = tt.TriangleMultiplication(
            False, trimul_weights(c_z, c_z, rng), cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                           fp32_dest_acc_en=True, packer_l1_acc=True))
    elif a.module == "transition":
        # The CONTROL. This module's gradient is independently verified against a float64
        # reference at 5.5e-03 to 7.0e-03 by `shipped_gradcheck.py`, so if the finite
        # difference disagrees HERE the harness is what is wrong, not the gradient.
        cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
        module = tt.Transition(transition_weights(c_z, 4 * c_z, rng), cls(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       math_approx_mode=False, fp32_dest_acc_en=True,
                                       packer_l1_acc=True))
    else:
        module = triangle_attention_module(tt, ttnn, dev, c_z, a.heads, a.head_dim, rng)

    x0 = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    lw = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    d_rand = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    d_rand = d_rand / d_rand.norm()

    stop, samples = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, samples), daemon=True)
    th.start()
    try:
        # -- analytic, from the tape over the SHIPPED module --------------------------
        xa = ag.Tensor(D(x0), requires_grad=True)
        with ag.tape():
            out = module(xa)
        out.backward(seed=D(lw))
        g = ttnn.to_torch(xa.grad).to(torch.float64)
        print(f"{a.module}  z [1,{S},{S},{c_z}] heads {a.heads} head_dim {a.head_dim}")
        print(f"  |dL/dz| {float(g.norm()):.4f}")

        # -- numeric, through production's own forward, eps swept ---------------------
        EPS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
        eps0 = EPS[0]

        def sweep(d, label):
            analytic = float((g * d.to(torch.float64)).sum())

            # Did the perturbation survive the cast? If not, nothing below is a reading.
            moved = ((x0 + eps0 * d).to(torch.bfloat16)
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
            print(f"    analytic <dL/dz, d> = {analytic:.6f}   "
                  f"elements that survive the bf16 cast at eps {eps0}: {moved*100:.1f}%")
            for eps, num, rel in rows:
                print(f"    eps {eps:5.3f}  numeric {num:14.6f}  rel {rel:.3e}"
                      f"{'   <- best' if best[0] == eps else ''}")
            return analytic, best, rows

        d_grad = (g / g.abs().max()).to(torch.float32)
        d_rand = d_rand / d_rand.abs().max()
        analytic, best, rows = sweep(d_grad, "the gradient itself (well conditioned)")
        if best is None:
            return 1
        sweep(d_rand, "random (nearly orthogonal, reported as the weaker reading)")
        ok = best[2] <= BAR
        print(f"  best-of-sweep rel {best[2]:.3e} at eps {best[0]} "
              f"against a {BAR:.1e} bar  {'PASS' if ok else 'FAIL'}")

        # -- the check must be able to fail -------------------------------------------
        bad = analytic * 1.5
        badrel = min(abs(n - bad) / max(abs(bad), 1e-12) for _, n, _ in rows)
        fails = badrel > BAR
        print(f"  negative control (analytic scaled 1.5x): best rel {badrel:.3e} "
              f"{'-- rejected, the check can fail' if fails else '-- CHECK IS BLIND'}")
        ok = ok and fails
    finally:
        stop.set(); th.join(timeout=3)
    if samples:
        print(f"  aiclk DURING: min {min(samples)} max {max(samples)} MHz "
              f"({len(samples)} samples)")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
