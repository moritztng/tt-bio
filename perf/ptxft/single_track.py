#!/usr/bin/env python3
"""The pairformer block's SINGLE track, differentiated and checked against float64.

The pair track has been differentiable since `tape_block.py` landed, and a distogram
objective cannot reach past it: z is a closed autoregression, so `attention_pair_bias`
and `single_transition` -- 2,513,280 of every block's 4,749,184 parameters, 52.9 % -- take
exactly zero gradient from it (`block_parity.py --s-independence` measures that). They are
not optional for a COMPLETE backward pass, though: the confidence head's pLDDT reads
s_single, so every confidence objective reaches them.

The reference is torch in float64 with autograd, not a hand-written backward: upstream
never writes one either, so torch's is the independent check, and it is the same thing the
tape will be seeded with. Forward and every weight gradient are compared.

The bar is the bf16 quantisation floor, pre-registered rather than fitted: u = 2^-9 =
1.95e-3 for one rounded operand and sqrt(2) u = 2.8e-3 for two, so 1.0e-2 relative L2 with
cosine >= 0.9999 is the same bar every earlier gradcheck in this row used.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float(np.linalg.norm(a - b))


def cos(a, b):
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 1.0


def reference(sd, s, z, seed_grad, *, heads, head_dim, eps=1e-5):
    """The same maths in torch float64, with autograd for the gradients."""
    import torch
    import torch.nn.functional as F
    T = lambda k: torch.tensor(sd[k].numpy(), dtype=torch.float64, requires_grad=True)
    W = {k: T(k) for k in sd}
    st = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    S, H, d = st.shape[0], heads, head_dim

    def ln(x, wk, bk):
        return (F.layer_norm(x, (x.shape[-1],), eps=eps) * W[wk]) + W[bk]

    sn = ln(st, "pre_norm_s.weight", "pre_norm_s.bias")
    q = (sn @ W["attention.proj_q.weight"].T + W["attention.proj_q.bias"]).view(
        S, H, d).permute(1, 0, 2)
    k = (sn @ W["attention.proj_k.weight"].T).view(S, H, d).permute(1, 0, 2)
    v = (sn @ W["attention.proj_v.weight"].T).view(S, H, d).permute(1, 0, 2)
    zn = ln(zt, "attention.proj_z.0.weight", "attention.proj_z.0.bias")
    bias = (zn @ W["attention.proj_z.1.weight"].T).permute(2, 0, 1)        # [H, S, S]
    att = torch.softmax(q @ k.transpose(-1, -2) * (d ** -0.5) + bias, dim=-1) @ v
    o = att.permute(1, 0, 2).reshape(S, H * d)
    o = o * torch.sigmoid(sn @ W["attention.proj_g.weight"].T)
    s1 = st + o @ W["attention.proj_o.weight"].T

    tn = ln(s1, "transition_s.norm.weight", "transition_s.norm.bias")
    x1 = tn @ W["transition_s.fc1.weight"].T
    x1 = x1 * torch.sigmoid(x1)
    x2 = tn @ W["transition_s.fc2.weight"].T
    out = s1 + (x1 * x2) @ W["transition_s.fc3.weight"].T

    (out * torch.tensor(seed_grad, dtype=torch.float64)).sum().backward()
    grads = {k: (W[k].grad.numpy() if W[k].grad is not None else None) for k in sd}
    return out.detach().numpy(), grads, st.grad.numpy(), zt.grad.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bar", type=float, default=1.0e-2)
    ap.add_argument("--cos-bar", type=float, default=0.9999)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--parts", action="store_true",
                    help="compare the attention's intermediates one at a time, which is "
                         "what localises a whole-block disagreement")
    a = ap.parse_args()

    import ttnn
    import torch
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from perf.ptxft.block_parity import load_blocks
    from perf.ptxft.tape_block import PairTrackBlock, SINGLE_TRACK_TARGETS
    from perf.ptxft.overfit import random_block_sd
    from perf.clocksample import during

    dt = ttnn.bfloat16 if a.dtype == "bf16" else ttnn.float32
    rng = np.random.default_rng(a.seed)
    with during() as clk:
        device = get_device()
        real, _head = load_blocks(a.ckpt, [0])
        shapes = {k: tuple(v.shape) for k, v in real[0].items()}
        sd = random_block_sd(shapes, rng)
        blk = PairTrackBlock(sd, device, dtype=dt, adapter_dtype=ttnn.float32,
                             train_base=True)
        S = a.n
        c_s = shapes["pre_norm_s.weight"][0]
        c_z = shapes["attention.proj_z.0.weight"][0]
        # [1, S, c_s]: the production single representation carries the batch axis.
        s_np = (rng.standard_normal((1, S, c_s)) * 0.5).astype(np.float32)
        z_np = (rng.standard_normal((S, S, c_z)) * 0.5).astype(np.float32)
        seed_np = (rng.standard_normal((1, S, c_s)) * 0.1).astype(np.float32)

        st = ag.Tensor(ft.to_device(s_np, device, dtype=dt), requires_grad=True)
        zt = ag.Tensor(ft.to_device(z_np, device, dtype=dt), requires_grad=True)
        out = blk.single(st, zt)
        got = ft.to_host(out.value).reshape(S, c_s)
        out.backward(seed=ft.to_device(seed_np, device, dtype=dt))

        want, gref, gs, gz = reference(sd, s_np[0], z_np, seed_np[0],
                                       heads=blk.s_heads, head_dim=blk.s_head_dim)

        if a.parts:
            # Walk the same chain on both sides and print where they first separate. A
            # single end-to-end number cannot say whether the twin or the transcription
            # is wrong, and it cannot say WHICH op.
            import torch
            import torch.nn.functional as F
            H, hd, pd = blk.s_heads, blk.s_head_dim, blk.s_pad_dim
            Wt = {k: torch.tensor(v.numpy(), dtype=torch.float64) for k, v in sd.items()}
            st64 = torch.tensor(s_np[0], dtype=torch.float64)
            zt64 = torch.tensor(z_np, dtype=torch.float64)
            rsn = (F.layer_norm(st64, (c_s,), eps=1e-5) * Wt["pre_norm_s.weight"]
                   + Wt["pre_norm_s.bias"])
            rq = (rsn @ Wt["attention.proj_q.weight"].T
                  + Wt["attention.proj_q.bias"]).view(S, H, hd).permute(1, 0, 2)
            rk = (rsn @ Wt["attention.proj_k.weight"].T).view(S, H, hd).permute(1, 0, 2)
            rv = (rsn @ Wt["attention.proj_v.weight"].T).view(S, H, hd).permute(1, 0, 2)
            rzn = (F.layer_norm(zt64, (c_z,), eps=1e-5) * Wt["attention.proj_z.0.weight"]
                   + Wt["attention.proj_z.0.bias"])
            rb = (rzn @ Wt["attention.proj_z.1.weight"].T).permute(2, 0, 1)
            ratt = (torch.softmax(rq @ rk.transpose(-1, -2) * (hd ** -0.5) + rb, -1) @ rv)
            tsn = blk._ln(st, "pre_norm_s")

            def th(name, bias=None):
                x = blk._lin(tsn, f"attention.proj_{name}", bias=bias)
                return ag.permute(ag.reshape(x, [1, S, H, pd]), (0, 2, 1, 3))

            tq = th("q", bias="attention.proj_q.bias")
            tk, tv = th("k"), th("v")
            tzn = blk._ln(zt, "attention.proj_z.0")
            tb = ag.reshape(ag.permute(blk._lin(tzn, "attention.proj_z.1"), (2, 0, 1)),
                            [1, H, S, S])
            tatt = ag.triangle_attention(tq, tk, tv, tb, scale=hd ** -0.5)
            un = lambda t: ft.to_host(t.value).reshape(1, H, S, pd)[0, :, :, :hd]
            print()
            print(f"{'intermediate':<28} {'rel L2':>10} {'cos':>10}")
            for nm, g, r in (("s LN", ft.to_host(tsn.value).reshape(S, c_s), rsn.numpy()),
                             ("q heads", un(tq), rq.numpy()),
                             ("k heads", un(tk), rk.numpy()),
                             ("v heads", un(tv), rv.numpy()),
                             ("z LN", ft.to_host(tzn.value).reshape(S, S, c_z),
                              rzn.numpy()),
                             ("pair bias", ft.to_host(tb.value).reshape(H, S, S),
                              rb.numpy()),
                             ("attention out", un(tatt), ratt.numpy())):
                print(f"{nm:<28} {rel_l2(g, r):>10.3e} {cos(g, r):>10.6f}")
            print()

        print(f"# --single-track: {S} tokens, c_s {c_s}, c_z {c_z}, "
              f"{blk.s_heads} heads x {blk.s_head_dim}, {a.dtype}, seed {a.seed}")
        print(f"# reference: torch float64 autograd. bar {a.bar:.1e} rel L2, "
              f"cos >= {a.cos_bar}")
        print(f"{'quantity':<28} {'rel L2':>10} {'cos':>10}")
        rows = [("forward", got, want),
                ("d/ds (input)", ft.to_host(st.grad).reshape(S, c_s), gs),
                ("d/dz (input)", ft.to_host(zt.grad).reshape(S, S, c_z), gz)]
        for name in SINGLE_TRACK_TARGETS + ("pre_norm_s.weight", "pre_norm_s.bias",
                                            "attention.proj_q.bias",
                                            "attention.proj_z.0.weight",
                                            "attention.proj_z.0.bias",
                                            "transition_s.norm.weight",
                                            "transition_s.norm.bias"):
            key = name if name.endswith(("weight", "bias")) else f"{name}.weight"
            t = blk.params.get(name)
            if t is None or t.grad is None:
                rows.append((f"d/d {name}", None, None))
                continue
            g = ft.to_host(t.grad)
            ref = gref[key]
            # The tape holds (in, out) where the checkpoint holds (out, in), and the
            # attention weights additionally carry the head-lane padding the twin needs
            # to keep [1, H, S, d] on a tile. Strip the pad before comparing: the pad
            # lanes have no counterpart upstream, and their gradient must be zero, which
            # is checked rather than assumed.
            H, hd, pd = blk.s_heads, blk.s_head_dim, blk.s_pad_dim
            if pd != hd and name in ("attention.proj_q", "attention.proj_k",
                                     "attention.proj_v", "attention.proj_g",
                                     "attention.proj_o", "attention.proj_q.bias"):
                if name == "attention.proj_q.bias":
                    gg = g.reshape(H, pd)
                    pad_g, g = gg[:, hd:], gg[:, :hd].reshape(H * hd)
                elif name == "attention.proj_o":
                    gg = g.reshape(H, pd, -1)
                    pad_g, g = gg[:, hd:], gg[:, :hd].reshape(H * hd, -1)
                else:
                    gg = g.reshape(-1, H, pd)
                    pad_g, g = gg[:, :, hd:], gg[:, :, :hd].reshape(gg.shape[0], H * hd)
                if np.abs(pad_g).max() > 0:
                    fails_pad.append(f"{name}: pad lanes carry gradient "
                                     f"{np.abs(pad_g).max():.3e}, so they were not inert")
            rows.append((f"d/d {name}", g.reshape(ref.T.shape).T if ref.ndim == 2
                         else g.reshape(ref.shape), ref))
        fails, fails_pad = [], []
        for name, g, ref in rows:
            if g is None:
                print(f"{name:<28} {'ABSENT':>10} {'':>10}   <-- FAIL")
                fails.append(f"{name}: no gradient at all")
                continue
            r, c = rel_l2(g, ref), cos(g, ref)
            ok = r <= a.bar and c >= a.cos_bar
            print(f"{name:<28} {r:>10.3e} {c:>10.6f}{'' if ok else '   <-- FAIL'}")
            if not ok:
                fails.append(f"{name}: rel L2 {r:.3e}, cos {c:.6f}")
    print()
    print(clk.line(0))
    print()
    fails += fails_pad
    if fails:
        print(f"SINGLE-TRACK FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"SINGLE-TRACK PASS: {len(rows)} quantities, all within {a.bar:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
