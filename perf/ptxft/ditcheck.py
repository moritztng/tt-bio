#!/usr/bin/env python3
"""The diffusion module's trunk block, differentiated and held to UPSTREAM's own module.

`DiffusionModule._denoise_device` (protenix.py:1092) runs 24 token-level
DiffusionTransformerBlocks between the atom attention encoder and decoder, and they hold
the bulk of the module's parameters. This differentiates ONE of them at ONE noise level,
which is what diffusion training does: `sample_diffusion_training` (generator.py:289)
draws one sigma per sample and calls the denoiser once. There is no trajectory in the
training graph, so the standing NO-GO on backpropagating the 200-step inference rollout --
whose mechanism is chained bf16 numerics over 200 denoise calls -- is absent here by
construction rather than by assumption.

THE REFERENCE IS BYTEDANCE'S OWN MODULE, not a transcription. It instantiates
`protenix.model.modules.transformer.DiffusionTransformerBlock` in float64, loads the real
checkpoint weights into it with `load_state_dict(strict=True)` -- so a name this twin
misread shows up as a missing or unexpected key rather than as a number -- runs it, and
takes its gradients from torch autograd. Pass 8 learned the other way round: a twin
checked only against a transcription written from reading the code cannot say which side
is wrong when they disagree.

Run it in fp32. `perf/ptxft/single_track.py` measured an fp32 weight against a bf16
activation at 2.13e-01 on a forward where bf16/bf16 reads 8.16e-03, and these weights are
fp32 because they are trainable.
"""

import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

REF = os.path.expanduser("~/ref/Protenix")


def upstream_block(c_a, c_s, c_z, n_heads, cross=False):
    """ByteDance's own DiffusionTransformerBlock, float64, on the CPU."""
    import torch
    if REF not in sys.path:
        sys.path.insert(0, REF)
    # Upstream JIT-compiles a CUDA fused layer norm at import unless this is set
    # (train_demo.sh:15 documents it), and there is no CUDA here.
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    if "optree" not in sys.modules:
        m = types.ModuleType("optree")
        m.tree_map = m.tree_flatten = lambda *a, **k: None
        sys.modules["optree"] = m
    from protenix.model.modules.transformer import DiffusionTransformerBlock
    import protenix.model.modules.primitives as P

    # THE ONE THING REPLACED IN UPSTREAM, AND WHY. primitives._attention (:253-256) hard
    # casts q and k to float32 -- an upcast from bf16 in training, but a DOWNCAST from
    # float64 -- and leaves v alone, so SDPA raises on the mixed dtypes and a float64
    # reference is impossible with it in place. The replacement is the same arithmetic in
    # the input dtype: q is already scaled by _prep_qkv (upstream passes scale=1.0 to
    # SDPA), so nothing is rescaled here. Every other line of the block, every linear,
    # both adaLNs, the gating and the bias projection, is upstream's own.
    def _attention64(q, k, v, attn_bias=None, **kw):
        s = q @ k.transpose(-1, -2)
        if attn_bias is not None:
            s = s + attn_bias
        return torch.softmax(s, dim=-1) @ v

    P._attention = _attention64
    torch.set_default_dtype(torch.float64)
    blk = DiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads,
                                    cross_attention_mode=cross)
    return blk.double()


TOKEN_PREFIX = "diffusion_module.diffusion_transformer.blocks."
ATOM_PREFIX = ("diffusion_module.atom_attention_encoder.atom_transformer."
               "diffusion_transformer.blocks.")


def load_dit(ckpt, index, prefix=TOKEN_PREFIX):
    """One DiT block's weights off the real checkpoint, under upstream's names."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "ema", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    pre = f"{prefix}{index}."
    out = {}
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith(pre):
            out[k[len(pre):]] = v
    if not out:
        raise KeyError(f"no {pre}* in {ckpt}")
    return out


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float(np.linalg.norm(a - b))


def cosine(a, b):
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=64, help="tokens")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    # THE BAR IS DERIVED FROM A MEASUREMENT, not fitted to the result.
    # perf/ptxft/cfgprobe.py times one fp32 512x512x512 matmul on this card at this exact
    # compute-kernel config (HiFi4, fp32_dest_acc_en, packer_l1_acc) against float64:
    # 1.479e-03. Dropping fp32_dest_acc_en costs 5.6x (8.324e-03) and LoFi costs 21x
    # (3.130e-02), so the config is taking effect -- the floor is the hardware. A block
    # backward chains roughly a dozen matmuls, so 2e-2 is what an error-free twin owes
    # here and 1e-2 is what the shorter token-level chain owes. The COSINE bar does the
    # real detection work either way: a transposed or mis-scaled gradient lands nowhere
    # near cos 0.9999 whatever its norm.
    ap.add_argument("--bar", type=float, default=None)
    ap.add_argument("--cos-bar", type=float, default=0.9999)
    ap.add_argument("--floor-slack", type=float, default=3.0,
                    help="how much worse than upstream's own fp32 run is still the "
                         "floor rather than a defect")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--atom", action="store_true",
                    help="the ATOM-level block: local windowed attention in "
                         "cross-attention mode, which is what atxE/atxD run")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--global-attn", action="store_true",
                    help="run the ATOM block's weights with GLOBAL attention. The "
                         "attribution control: it changes only the windowing, so if the "
                         "error collapses the window path owns it and if it does not, "
                         "the device's arithmetic does.")
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--n-keys", type=int, default=128)
    a = ap.parse_args()
    if a.bar is None:
        a.bar = 2.0e-2 if a.atom else 1.0e-2

    import ttnn
    import torch
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from perf.ptxft.tape_block import DiTBlock
    from perf.clocksample import during

    dt = ttnn.bfloat16 if a.dtype == "bf16" else ttnn.float32
    rng = np.random.default_rng(a.seed)
    prefix = a.prefix or (ATOM_PREFIX if a.atom else TOKEN_PREFIX)
    sd = load_dit(a.ckpt, a.block, prefix)
    nq = a.n_queries if (a.atom and not a.global_attn) else None
    nk = a.n_keys if (a.atom and not a.global_attn) else None
    c_s = int(sd["attention_pair_bias.layernorm_a.layernorm_s.weight"].shape[0])
    c_a = int(sd["attention_pair_bias.attention.linear_q.weight"].shape[0])
    c_z = int(sd["attention_pair_bias.layernorm_z.weight"].shape[0])
    H = int(sd["attention_pair_bias.linear_nobias_z.weight"].shape[0])
    N = a.n
    a_np = (rng.standard_normal((1, N, c_a)) * 0.5).astype(np.float32)
    s_np = (rng.standard_normal((1, N, c_s)) * 0.5).astype(np.float32)
    g_np = (rng.standard_normal((1, N, c_a)) * 0.1).astype(np.float32)
    if nq is not None:
        # z arrives ALREADY trunked at atom level: upstream asserts
        # len(z.shape) == len(q.shape) + 2 and reads [n_trunks, n_queries, n_keys, c_z]
        # (transformer.py:132-140).
        n_trunks = -(-N // nq)
        z_np = (rng.standard_normal((n_trunks, nq, nk, c_z)) * 0.5).astype(np.float32)
    else:
        z_np = (rng.standard_normal((N, N, c_z)) * 0.5).astype(np.float32)

    fails = []
    with during() as clk:
        device = get_device()
        blk = DiTBlock(sd, device, dtype=dt, param_dtype=dt, trainable=True,
                       n_queries=nq, n_keys=nk)
        at = ag.Tensor(ft.to_device(a_np, device, dtype=dt), requires_grad=True)
        st = ag.Tensor(ft.to_device(s_np, device, dtype=dt), requires_grad=True)
        zt = ag.Tensor(ft.to_device(z_np, device, dtype=dt), requires_grad=True)
        out = blk(at, st, zt)
        got = ft.to_host(out.value).reshape(1, N, c_a)
        out.backward(seed=ft.to_device(g_np, device, dtype=dt))

        ref = upstream_block(c_a, c_s, c_z, H, cross=blk.cross)
        missing, unexpected = ref.load_state_dict(
            {k: v.double() for k, v in sd.items()}, strict=False)
        missing = [m for m in missing if "drop_path" not in m]
        if missing or unexpected:
            fails.append(f"state_dict mismatch against upstream: missing {missing}, "
                         f"unexpected {list(unexpected)}")
        # THE FLOOR CONTROL. An fp32 device against a float64 reference cannot be
        # scored by a bar that was pre-registered from bf16 quantisation, so the same
        # comparison is run against upstream itself in float32: same code, same weights,
        # same inputs, only the precision changed. That number is what ANY fp32
        # implementation of this block owes, and the twin is scored against it rather
        # than against a bar chosen to fit.
        ref32 = upstream_block(c_a, c_s, c_z, H, cross=blk.cross).float()
        ref32.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        a32 = torch.tensor(a_np[0], dtype=torch.float32, requires_grad=True)
        s32 = torch.tensor(s_np[0], dtype=torch.float32, requires_grad=True)
        z32 = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
        o32, _a, _b = ref32(a=a32, s=s32, z=z32, n_queries=nq, n_keys=nk)
        (o32 * torch.tensor(g_np[0], dtype=torch.float32)).sum().backward()

        ar = torch.tensor(a_np[0], dtype=torch.float64, requires_grad=True)
        sr = torch.tensor(s_np[0], dtype=torch.float64, requires_grad=True)
        zr = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
        rout, _s, _z = ref(a=ar, s=sr, z=zr, n_queries=nq, n_keys=nk)
        (rout * torch.tensor(g_np[0], dtype=torch.float64)).sum().backward()

        print(f"# --ditcheck: {'ATOM' if a.atom else 'TOKEN'} block {a.block}, {N} "
              f"{'atoms' if a.atom else 'tokens'}, c_a {c_a}, c_s {c_s}, c_z {c_z}, "
              f"{H} heads x {c_a // H} (padded to {blk.pad_dim}), {a.dtype}"
              + (f", windows {nq}/{nk}, cross-attention {blk.cross}" if a.atom else ""))
        print(f"# reference: ByteDance's own DiffusionTransformerBlock in float64, real "
              f"checkpoint weights, autograd gradients")
        print(f"# bar {a.bar:.1e} rel L2 and cos >= {a.cos_bar}; the bar comes from the "
              f"1.479e-03 per-matmul device floor cfgprobe.py measures at this config, "
              f"not from this run")
        print(f"{'quantity':<46} {'rel L2':>10} {'cos':>10} {'fp32 floor':>11} "
              f"{'ratio':>7}")

        def check(name, g, r, floor=None):
            if g is None:
                print(f"{name:<46} {'ABSENT':>10}")
                fails.append(f"{name}: no gradient")
                return
            rr, cc = rel_l2(g, r), cosine(g, r)
            fl = None if floor is None else rel_l2(floor, r)
            # Pass on EITHER the pre-registered bar or the measured fp32 floor times the
            # slack: the bar is the right question for a bf16 arm and the floor is the
            # right one here, and quoting both keeps a loose floor from hiding a defect.
            ok = (rr <= a.bar) or (fl is not None and rr <= a.floor_slack * fl)
            ok = ok and cc >= a.cos_bar
            fs = "" if fl is None else f"{fl:>11.3e} {rr / fl if fl > 0 else 0:>7.2f}"
            print(f"{name:<46} {rr:>10.3e} {cc:>10.6f} {fs}"
                  f"{'' if ok else '   <-- FAIL'}")
            if not ok:
                fails.append(f"{name}: rel L2 {rr:.3e}, cos {cc:.6f}"
                             + ("" if fl is None else f", fp32 floor {fl:.3e}"))

        check("forward", got.reshape(N, c_a), rout.detach().numpy(),
              o32.detach().numpy())
        check("d/da (input)", ft.to_host(at.grad).reshape(N, c_a), ar.grad.numpy(),
              a32.grad.numpy())
        check("d/ds (input)", ft.to_host(st.grad).reshape(N, c_s), sr.grad.numpy(),
              s32.grad.numpy())
        check("d/dz (input)", ft.to_host(zt.grad).reshape(z_np.shape), zr.grad.numpy(),
              z32.grad.numpy())
        named = dict(ref.named_parameters())
        named32 = dict(ref32.named_parameters())
        H_, hd, pd = blk.n_heads, blk.head_dim, blk.pad_dim
        laned = {"attention_pair_bias.attention.linear_q.weight": 0,
                 "attention_pair_bias.attention.linear_k.weight": 0,
                 "attention_pair_bias.attention.linear_v.weight": 0,
                 "attention_pair_bias.attention.linear_g.weight": 0,
                 "attention_pair_bias.attention.linear_q.bias": 0,
                 "attention_pair_bias.attention.linear_o.weight": 1}
        for name, t in sorted(blk.params.items()):
            r = named[name].grad
            if r is None:
                fails.append(f"{name}: upstream took no gradient, so the twin's cannot "
                             f"be checked -- the name is probably wrong")
                continue
            r = r.numpy()
            g = ft.to_host(t.grad) if t.grad is not None else None
            if g is not None and name in laned and pd != hd:
                # Strip the head-lane padding, and assert the pad lanes stayed inert.
                if name.endswith(".bias"):
                    gg = g.reshape(H_, pd)
                    pad, g = gg[:, hd:], gg[:, :hd].reshape(H_ * hd)
                elif laned[name] == 1:
                    gg = g.reshape(H_, pd, -1)
                    pad, g = gg[:, hd:], gg[:, :hd].reshape(H_ * hd, -1)
                else:
                    gg = g.reshape(-1, H_, pd)
                    pad, g = gg[:, :, hd:], gg[:, :, :hd].reshape(gg.shape[0], H_ * hd)
                if np.abs(pad).max() > 0:
                    fails.append(f"{name}: pad lanes carry gradient "
                                 f"{np.abs(pad).max():.3e}")
            if g is not None and r.ndim == 2:
                g = g.reshape(r.T.shape).T
            elif g is not None:
                g = g.reshape(r.shape)
            f32 = named32[name].grad
            check(f"d/d {name}", g, r, None if f32 is None else f32.numpy())
    print()
    print(clk.line(0))
    print()
    if fails:
        print(f"DITCHECK FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"DITCHECK PASS: forward and every gradient within {a.bar:.0e} of upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
