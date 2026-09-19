#!/usr/bin/env python3
"""Sub-module gradients on the same warmed probe the stack instrument uses.

At stack scope the forward agrees (s 2.3e-03, z 2.8e-02) and the SINGLE-track weight gradients
agree (6.1e-03 to 6.4e-02) while every PAIR-track weight reads 4.3e-01 to 1.4e+00. Either each
pair-track sub-module is individually wrong, or the composition is. This asks each one on its
own, on the identical probe, against the identical float64 reference, so the answer is a
measurement rather than a hypothesis.

The mapping from their parameters to our device tensors is `bijection_device`, the same matcher
the stack instrument uses, so a mapping error would show up in both rather than in one.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "perf/of3t_gradients")

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_gradients/bisect_grad.json"
BLOCK, BAR = 2, 5.0e-02


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device, device_weights
    from tt_bio.train.lora import walked_weights
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention as TheirTriAtt, TriangleAttentionEndingNode)
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from bijection_device import device_bijection
    from instrument_a_stack import load_ckpt, their_stack

    sd = load_ckpt()
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    P = f"pairformer_stack.blocks.{BLOCK}."

    N = 64
    g = torch.Generator().manual_seed(11)
    s64 = (torch.randn(1, N, 384, generator=g) * 0.05).double()
    z64 = (torch.randn(1, N, N, 128, generator=g) * 0.05).double()
    sm, pm = torch.ones(1, N, dtype=torch.float64), torch.ones(1, N, N, dtype=torch.float64)
    warm, _, _ = their_stack(sd, BLOCK, first=0)
    with torch.no_grad():
        for m in warm:
            s64, z64 = m(s64, z64, sm, pm)
    z64 = z64.detach()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)

    def ours_flat(prefix, strip=""):
        n = len(prefix)
        return {(k[n:][len(strip):] if k[n:].startswith(strip) else k[n:]): v
                for k, v in flat_all.items()
                if k.startswith(prefix) and k.startswith(f"layers.{BLOCK}.")}

    n_heads = flat_all[f"layers.{BLOCK}.tri_att_start.linear.weight"].shape[0]
    head_dim = flat_all[f"layers.{BLOCK}.tri_att_start.mha.linear_q.weight"].shape[0] // n_heads
    L = f"layers.{BLOCK}."
    CASES = [
        ("tri_mul_out", lambda: TriangleMultiplicationOutgoing(c_z=128, c_hidden=128),
         "pair_stack.tri_mul_out.", lambda: T.TriangleMultiplication(
             False, ours_flat(L + "tri_mul_out."), ckc)),
        ("tri_mul_in", lambda: TriangleMultiplicationIncoming(c_z=128, c_hidden=128),
         "pair_stack.tri_mul_in.", lambda: T.TriangleMultiplication(
             True, ours_flat(L + "tri_mul_in."), ckc)),
        # R11 from `of3t-memory`: OF3's `fp32_softmax=True` path has no backward. Both
        # settings are run rather than argued about, because the forward agrees on both and
        # only the gradient can tell them apart.
        ("tri_att_start", lambda: TheirTriAtt(c_in=128, c_hidden=head_dim,
                                              no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_start.", lambda: T.TriangleAttention(
             head_dim, n_heads, False, ours_flat(L + "tri_att_start.", "mha."), ckc,
             scale_pair_bias=False, fp32_softmax=True)),
        ("tri_att_start_nofp32", lambda: TheirTriAtt(c_in=128, c_hidden=head_dim,
                                                     no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_start.", lambda: T.TriangleAttention(
             head_dim, n_heads, False, ours_flat(L + "tri_att_start.", "mha."), ckc,
             scale_pair_bias=False, fp32_softmax=False)),
        ("tri_att_end", lambda: TriangleAttentionEndingNode(c_in=128, c_hidden=head_dim,
                                                            no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_end.", lambda: T.TriangleAttention(
             head_dim, n_heads, True, ours_flat(L + "tri_att_end.", "mha."), ckc,
             scale_pair_bias=False, fp32_softmax=True)),
        ("tri_att_end_nofp32", lambda: TriangleAttentionEndingNode(c_in=128, c_hidden=head_dim,
                                                                   no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_end.", lambda: T.TriangleAttention(
             head_dim, n_heads, True, ours_flat(L + "tri_att_end.", "mha."), ckc,
             scale_pair_bias=False, fp32_softmax=False)),
        # `scale_pair_bias=True` at the triangle attentions is the WRONG setting per R27/K34 and
        # is run only as this instrument's own control: it must read WORSE, and if it does not,
        # the instrument is not reading the bias path at all.
        ("tri_att_end_scaledbias", lambda: TriangleAttentionEndingNode(
            c_in=128, c_hidden=head_dim, no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_end.", lambda: T.TriangleAttention(
             head_dim, n_heads, True, ours_flat(L + "tri_att_end.", "mha."), ckc,
             scale_pair_bias=True, fp32_softmax=True)),
        ("pair_transition", lambda: SwiGLUTransition(c_in=128, n=4),
         "pair_stack.pair_transition.", lambda: T.Transition(
             ours_flat(L + "transition_z."), ckc)),
    ]

    report = {"block": BLOCK, "tokens": N, "bar": BAR,
              "probe": {"z_norm": float(z64.norm()), "warmup_blocks": BLOCK}, "cases": []}
    for label, their_ctor, their_pre, our_ctor in CASES:
        them = their_ctor().to(torch.float64)
        them.load_state_dict({k[len(P + their_pre):]: v.to(torch.float64)
                              for k, v in sd.items() if k.startswith(P + their_pre)}, strict=True)
        them.eval()
        for p in them.parameters():
            p.grad = None
        u_ref = them(z64, mask=pm)
        cg = torch.Generator().manual_seed(23)
        cot = torch.randn(*u_ref.shape, generator=cg).double()
        cot = cot / cot.norm() * float(u_ref.detach().norm())
        (u_ref * cot).sum().backward()
        g_ref = {n: p.grad.detach().clone() for n, p in them.named_parameters()}

        mod = our_ctor()
        params = walked_weights(lambda: mod(ft(z64)), None, mod)
        za = ag.Tensor(ft(z64), requires_grad=True)
        with ag.tape():
            u = mod(za)
        fwd = rel(ttnn.to_torch(u.value).to(torch.float64).numpy(),
                  u_ref.detach().numpy())
        ag.backward([u], [ft(cot)])

        dv = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod).items()}
        atoms = {n: p.detach().to(torch.float32) for n, p in them.named_parameters()}
        bij = device_bijection(dv, atoms)
        got, miss = {}, []
        for their, pls in bij["placements"].items():
            pick = next((p for p in pls if params.get(p["device_path"]) is not None
                         and params[p["device_path"]].grad is not None), None)
            if pick is None:
                miss.append(their)
                continue
            gd = ttnn.to_torch(params[pick["device_path"]].grad).to(torch.float64)
            band = gd.narrow(pick["axis"], pick["start"], pick["length"])
            if pick["layout"].endswith("transposed"):
                band = band.T
            got[their] = rel(band.contiguous().numpy(), g_ref[their].numpy())
        worst = max(got.items(), key=lambda kv: kv[1]) if got else (None, None)
        report["cases"].append({"module": label, "forward_rel": fwd, "per_tensor": got,
                                "absent": miss + bij["their_unplaced"],
                                "worst": worst[0], "worst_rel": worst[1],
                                "n_compared": len(got)})
        print(f"{label:16s} fwd {fwd:.3e} | {len(got)} compared, "
              f"worst {worst[1]:.3e} on {worst[0]} | absent {len(miss + bij['their_unplaced'])}",
              flush=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
