#!/usr/bin/env python3
"""Protenix v2's confidence heads, differentiated and checked against float64.

Four heads, all on the REAL checkpoint weights rather than random ones, because they are
small enough that there is no reason not to: PAE and PDE are a layer norm and one linear
over the pair tensor, pLDDT and the experimentally-resolved head are a layer norm over the
single track followed by a gathered matmul, one (c_s, n_bins) matrix per atom chosen by
its token-atom type out of 24.

WHERE THE HOST BOUNDARY IS AND WHY IT IS EXACT. Two gathers sit in the per-atom heads:
`s_single[atom_to_token_idx]` and `plddt_weight[atom_to_tokatom_idx]`. A gather is the one
op this tape does not have, and both are avoidable rather than worth an op:

* layer norm is per row, so LN(s[a2t]) == (LN(s))[a2t]. The norm is taped over TOKENS and
  the atom gather happens after it, which is the same arithmetic in a different order.
* the gathered matmul runs on host in float64 and hands the tape the exact analytic
  gradient of its input, scatter-added from atoms back to tokens. That is the arrangement
  the distogram loss already uses: the loss VALUE and this head's 460,800 parameters are
  cheap on host, and what crosses back to the tape is the analytic gradient, not an
  approximation of it.

The reference is torch float64 autograd over the whole composite, gathers included, so the
host/device split is inside what is being checked rather than assumed away.

Run it in fp32. `perf/ptxft/single_track.py` measured an fp32 weight against a bf16
activation at 2.13e-01 on a forward where bf16/bf16 reads 8.16e-03, and these heads keep
their trainable weights in fp32, so a bf16 activation arm here would be measuring that
defect and not the heads.
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


def cosine(a, b):
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 1.0


def load_conf(ckpt):
    """The confidence head's own weights off the real checkpoint."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "ema", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    pre = "confidence_head."
    out = {}
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith(pre) and "pairformer_stack" not in k:
            out[k[len(pre):]] = v
    if not out:
        raise KeyError(f"no {pre}* weights in {ckpt}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=48, help="tokens")
    ap.add_argument("--atoms-per-token", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bar", type=float, default=1.0e-2)
    ap.add_argument("--cos-bar", type=float, default=0.9999)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    a = ap.parse_args()

    import ttnn
    import torch
    import torch.nn.functional as F
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from perf.ptxft.tape_block import ConfidenceHeads, gathered_head
    from perf.clocksample import during

    dt = ttnn.bfloat16 if a.dtype == "bf16" else ttnn.float32
    rng = np.random.default_rng(a.seed)
    sd = load_conf(a.ckpt)
    NT = a.n
    NA = NT * a.atoms_per_token
    c_z = sd["pae_ln.weight"].shape[0]
    c_s = sd["plddt_ln.weight"].shape[0]
    n_types = sd["plddt_weight"].shape[0]
    a2t = np.repeat(np.arange(NT), a.atoms_per_token)
    a2ta = rng.integers(0, n_types, size=NA)
    z_np = (rng.standard_normal((NT, NT, c_z)) * 0.5).astype(np.float32)
    s_np = (rng.standard_normal((1, NT, c_s)) * 0.5).astype(np.float32)
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=True)

    rows, fails = [], []
    with during() as clk:
        device = get_device()
        print(f"# --confhead: {NT} tokens, {NA} atoms, c_z {c_z}, c_s {c_s}, "
              f"{n_types} token-atom types, {a.dtype}, seed {a.seed}")
        print(f"# reference: torch float64 autograd over the whole composite, both "
              f"gathers included. bar {a.bar:.1e} rel L2, cos >= {a.cos_bar}")
        print(f"{'quantity':<34} {'rel L2':>10} {'cos':>10}")

        def check(name, got, ref):
            r, c = rel_l2(got, ref), cosine(got, ref)
            ok = r <= a.bar and c >= a.cos_bar
            print(f"{name:<34} {r:>10.3e} {c:>10.6f}{'' if ok else '   <-- FAIL'}")
            if not ok:
                fails.append(f"{name}: rel L2 {r:.3e}, cos {c:.6f}")

        # ---------------------------------------------------------------- pair heads
        for head, ln, lin in (("pae", "pae_ln", "linear_no_bias_pae"),
                              ("pde", "pde_ln", "linear_no_bias_pde")):
            H = ConfidenceHeads(sd, device, dtype=dt, param_dtype=dt, trainable=True)
            zt = ag.Tensor(ft.to_device(z_np, device, dtype=dt), requires_grad=True)
            out = getattr(H, head)(zt)
            n_bins = int(out.value.shape[-1])
            g = (rng.standard_normal((NT, NT, n_bins)) * 0.1).astype(np.float32)
            got = ft.to_host(out.value).reshape(NT, NT, n_bins)
            out.backward(seed=ft.to_device(g, device, dtype=dt))

            zr = T(z_np)
            w, b = T(sd[f"{ln}.weight"]), T(sd[f"{ln}.bias"])
            lw = T(sd[f"{lin}.weight"])
            x = zr + zr.transpose(0, 1) if head == "pde" else zr
            ref = (F.layer_norm(x, (c_z,), eps=1e-5) * w + b) @ lw.T
            (ref * torch.tensor(g, dtype=torch.float64)).sum().backward()

            check(f"{head} forward", got, ref.detach().numpy())
            check(f"{head} d/dz", ft.to_host(zt.grad).reshape(NT, NT, c_z),
                  zr.grad.numpy())
            check(f"{head} d/d {ln}.weight",
                  ft.to_host(H.params[f"{ln}.weight"].grad).reshape(-1), w.grad.numpy())
            check(f"{head} d/d {ln}.bias",
                  ft.to_host(H.params[f"{ln}.bias"].grad).reshape(-1), b.grad.numpy())
            check(f"{head} d/d {lin}",
                  ft.to_host(H.params[lin].grad).T, lw.grad.numpy())

        # ---------------------------------------------------------------- atom heads
        for head, wname in (("plddt", "plddt_weight"), ("resolved", "resolved_weight")):
            H = ConfidenceHeads(sd, device, dtype=dt, param_dtype=dt, trainable=True)
            st = ag.Tensor(ft.to_device(s_np, device, dtype=dt), requires_grad=True)
            aln = H.atom_norm(st, head)
            tok = ft.to_host(aln.value).reshape(NT, c_s).astype(np.float64)
            atom = tok[a2t]
            logits, back = gathered_head(atom, H.host[wname], a2ta)
            n_bins = logits.shape[-1]
            g = (rng.standard_normal((NA, n_bins)) * 0.1).astype(np.float64)
            dx_atom, dW = back(g)
            # atoms -> tokens: every atom of a token feeds the same normed row, so the
            # scatter must ACCUMULATE. A fancy-index assignment would keep one atom's
            # contribution per token and silently drop the other three.
            dtok = np.zeros((NT, c_s))
            np.add.at(dtok, a2t, dx_atom)
            aln.backward(seed=ft.to_device(dtok.reshape(1, NT, c_s).astype(np.float32),
                                           device, dtype=dt))

            sr = T(s_np[0])
            w, b = T(sd[f"{head}_ln.weight"]), T(sd[f"{head}_ln.bias"])
            Wr = T(sd[wname])
            alnr = F.layer_norm(sr, (c_s,), eps=1e-5) * w + b
            ref = torch.einsum("nc,ncb->nb", alnr[torch.tensor(a2t)],
                               Wr[torch.tensor(a2ta)])
            (ref * torch.tensor(g)).sum().backward()

            check(f"{head} forward", logits, ref.detach().numpy())
            check(f"{head} d/ds", ft.to_host(st.grad).reshape(NT, c_s), sr.grad.numpy())
            check(f"{head} d/d {head}_ln.weight",
                  ft.to_host(H.params[f"{head}_ln.weight"].grad).reshape(-1),
                  w.grad.numpy())
            check(f"{head} d/d {head}_ln.bias",
                  ft.to_host(H.params[f"{head}_ln.bias"].grad).reshape(-1),
                  b.grad.numpy())
            check(f"{head} d/d {wname} (host)", dW, Wr.grad.numpy())
    print()
    print(clk.line(0))
    print()
    if fails:
        print(f"CONFHEAD FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"CONFHEAD PASS: all four heads, forward and every gradient within {a.bar:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
