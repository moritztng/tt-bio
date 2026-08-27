#!/usr/bin/env python3
"""Score the tt-bio Protenix port against an upstream v0.5.0 capture, module by module.

Both arms read the same feature dict, so every number here is the port's arithmetic.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$WT \
      env/bin/python3 scripts/protenix_v1_port/tt_parity.py \
        --feats /tmp/pv1/feats_multimer.pt --ref /tmp/pv1/ref_multimer.pt --ckpt <v1.pt>

The port's inner stages are tapped by wrapping instance attributes here, not by adding debug
hooks to tt_bio: the production path stays exactly the path that is being scored.
"""
from __future__ import annotations

import argparse
import os

import torch


def align(got, want):
    """Crop the port's tensor to the reference's shape, per AXIS.

    The port's trunk tensors come back BUCKETED -- z is (1, 256, 256, 128) where the reference
    is (228, 228, 128). Flattening both and truncating to the shorter one compares row 0 of the
    padded tensor against rows 0..1 of the reference and reports nonsense: it read PCC 0.29 on
    a z track whose end-to-end PCC is 0.992. Crop each axis instead.
    """
    # float64 for the reduction: at 3.3 M elements the float32 dot product accumulates enough
    # error to report PCC 1.000995, and a correlation above 1 is not a number anyone can read.
    got = torch.as_tensor(got).double()
    want = torch.as_tensor(want).double()
    got = got.reshape([d for d in got.shape if d != 1] or [1])
    want = want.reshape([d for d in want.shape if d != 1] or [1])
    if got.dim() != want.dim():
        return got.flatten(), want.flatten()
    for ax, n in enumerate(want.shape):
        if got.shape[ax] < n:
            return got.flatten(), want.flatten()
        got = got.narrow(ax, 0, n)
    return got.flatten(), want.flatten()


def pcc(a, b):
    a, b = align(a, b)
    if a.numel() == 0:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm())
    return float("nan") if d == 0 else float((a @ b) / d)


def rel(a, b):
    a, b = align(a, b)
    return float((a - b).norm() / (b.norm() + 1e-12))


class Report:
    def __init__(self, threshold):
        self.rows = []
        self.threshold = threshold

    #: absolute agreement a [0, 1] confidence scalar must reach
    SCALAR_TOL = 0.01

    def scalar(self, name, got, want):
        d = abs(got - want)
        self.rows.append((name, float("nan"), d, "port %.4f vs ref %.4f  |d|<%.2f" %
                          (got, want, self.SCALAR_TOL), d <= self.SCALAR_TOL))

    def add(self, name, got, want, note=""):
        if got is None or want is None:
            self.rows.append((name, float("nan"), float("nan"), "MISSING " + note, False))
            return
        p, r = pcc(got, want), rel(got, want)
        self.rows.append((name, p, r, note, p == p and p >= self.threshold))

    def show(self):
        print("\n%-34s %9s %9s  %-6s %s" % ("module", "PCC", "relerr", "", "note"))
        npass = 0
        for name, p, r, note, ok in self.rows:
            npass += bool(ok)
            print("%-34s %9s %9.2e  %-6s %s"
                  % (name, "--" if p != p else "%.6f" % p, r, "PASS" if ok else "FAIL", note))
        print("\nPARITY modules=%d/%d threshold=%.4f" % (npass, len(self.rows), self.threshold))
        return npass, len(self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio import weights
    from tt_bio.protenix import Protenix

    ckpt = args.ckpt or str(weights.fetch("protenix-v1"))
    feats = torch.load(args.feats, map_location="cpu", weights_only=False)["feats"]
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    cap = ref["cap"]
    N, ncyc = ref["N"], ref["n_cycle"]
    print("ref N=%d n_cycle=%d" % (N, ncyc))

    m = Protenix.load_from_checkpoint(ckpt)
    print("trunk c_z=%d n_cycles=%d tpl_blocks=%d msa=%d pf=%d"
          % (m.trunk.C_Z, m.trunk.N_CYCLES, len(m.trunk.TPL),
             len(getattr(m.trunk, "MSA", []) or []), len(m.trunk.PF.blocks)))

    rep = Report(args.threshold)

    # ---- tap the trunk's per-cycle stages -------------------------------------------------
    import ttnn
    taps = {}

    def dev(t, shape=None):
        return Protenix._to_host(t, shape)

    orig_msa = m.trunk._msa
    def msa_tap(z3, *a, **k):
        out = orig_msa(z3, *a, **k)
        taps.setdefault("msa_out", []).append(dev(out))
        return out
    m.trunk._msa = msa_tap

    orig_pf = m.trunk.PF.__call__
    def pf_tap(s, z, *a, **k):
        so, zo = orig_pf(s, z, *a, **k)
        taps.setdefault("pf_s", []).append(dev(so))
        taps.setdefault("pf_z", []).append(dev(zo))
        return so, zo
    m.trunk.PF = type("PFTap", (), {"__call__": staticmethod(pf_tap),
                                    "blocks": m.trunk.PF.blocks})()

    # ---- trunk ---------------------------------------------------------------------------
    cond, aux = m._trunk_cond(feats, n_cycles=ncyc)
    rep.add("input_embedder -> s_inputs", aux["s_inputs"], ref["s_inputs"])
    for i in range(ncyc):
        if "msa_module" in cap and i < len(cap["msa_module"]) and i < len(taps.get("msa_out", [])):
            rep.add("msa_module cyc%d" % i, taps["msa_out"][i], cap["msa_module"][i])
    for i in range(ncyc):
        if "pairformer_stack" in cap and i < len(cap["pairformer_stack"]) and i < len(taps.get("pf_s", [])):
            rs, rz = cap["pairformer_stack"][i][:2]
            rep.add("pairformer48 s cyc%d" % i, taps["pf_s"][i], rs)
            rep.add("pairformer48 z cyc%d" % i, taps["pf_z"][i], rz)
    rep.add("trunk s_trunk", aux["s_trunk"], ref["s_trunk"])
    rep.add("trunk z_trunk", aux["z_trunk"], ref["z_trunk"])

    # ---- diffusion: one denoise at the reference's own fixed x_noisy / t_hat --------------
    d = m.diffusion
    dtaps = {}
    class _Tap:
        """Forward every attribute to the wrapped module; record only what it returns.

        atxE/atxD are AtomTransformer INSTANCES, not functions -- denoise also calls
        `self.atxE.precompute_biases(...)`, so a bare lambda replacement breaks the fold.
        """
        def __init__(self, inner, key):
            self._inner, self._key = inner, key

        def __call__(self, *a, **k):
            out = self._inner(*a, **k)
            dtaps.setdefault(self._key, out)
            return out

        def __getattr__(self, name):
            return getattr(self._inner, name)

    d.atxE = _Tap(d.atxE, "q")
    d.atxD = _Tap(d.atxD, "qd")
    if getattr(d, "device_dit", False):
        o_dit = d._token_dit_device
        d._token_dit_device = lambda *a, **k: dtaps.setdefault("a_t", o_dit(*a, **k))
    else:
        o_dit = d._token_dit
        d._token_dit = lambda *a, **k: dtaps.setdefault("a_t", o_dit(*a, **k))

    denoised = d.denoise(ref["x_noisy"], ref["t_hat"], cond)
    if "q" in dtaps and "diff_atom_encoder" in cap:
        enc = cap["diff_atom_encoder"][0]
        q_skip = enc[1] if isinstance(enc, tuple) and len(enc) > 1 else None
        rep.add("diffusion atom encoder q", dev(dtaps["q"]), q_skip)
    if "a_t" in dtaps and "dit" in cap:
        got = dtaps["a_t"]
        rep.add("DiT 24 blocks", dev(got) if not torch.is_tensor(got) else got, cap["dit"][0])
    # The port's atom decoder output is r_update, which denoise folds straight into the EDM
    # preconditioning. Invert that algebra rather than re-implementing the last two layers here:
    #   denoised = x/(1+sr^2) + t_hat/sqrt(1+sr^2) * r_update,  sr = t_hat/sigma_data
    if "diff_atom_decoder" in cap:
        sd_ = float(ref.get("sigma_data", 16.0))
        th = ref["t_hat"].reshape(-1, 1, 1)
        sr = th / sd_
        r_update = (denoised - ref["x_noisy"] / (1.0 + sr ** 2)) / (th / torch.sqrt(1.0 + sr ** 2))
        rep.add("diffusion atom decoder r_update", r_update, cap["diff_atom_decoder"][0])
    rep.add("denoise (x0 at sigma=%g)" % float(ref["t_hat"][0]), denoised, ref["denoised"])

    # ---- confidence heads ------------------------------------------------------------------
    conf = m.confidence_head.confidence(aux["s_inputs"], aux["s_trunk"], aux["z_trunk"],
                             denoised.reshape(-1, 3), feats)
    # ConfidenceHead.forward returns (plddt_pred, pae_pred, pde_pred, resolved_pred) --
    # confidence.py:347. The capture stores the raw tuple under "out".
    rc = ref["confidence"].get("out")
    if isinstance(rc, tuple) and len(rc) >= 3:
        ref_plddt, ref_pae, ref_pde = (rc[0].squeeze(0), rc[1].squeeze(0), rc[2].squeeze(0))
        # The port's confidence() returns POST-processed quantities (expected distances, pTM,
        # ipTM), upstream returns bin LOGITS. Run the reference logits through the port's own
        # _postprocess so the two sides are the same quantity -- one function, two inputs --
        # instead of scoring an expected distance against a 64-bin logit tensor.
        rp = m.confidence_head._postprocess(ref_pae, ref_pde, ref_plddt, feats)
        for k, label in (("pae", "confidence PAE"), ("pde", "confidence PDE"),
                         ("plddt_atom", "confidence pLDDT (per atom)")):
            rep.add(label, conf.get(k), rp.get(k))
        print("\nheads   port      ref")
        for k in ("plddt", "ptm", "iptm"):
            print("%-7s %-9.4f %-9.4f" % (k, float(conf.get(k, float("nan"))),
                                          float(rp.get(k, float("nan")))))
        # pTM/ipTM are single scalars in [0, 1], so they are judged on ABSOLUTE delta, not on a
        # correlation of one number and not on relative error: ipTM is near zero on a poorly
        # packed interface, where a 0.002 absolute agreement reads as a 2% relative "failure".
        for k in ("ptm", "iptm"):
            got, want = float(conf.get(k, float("nan"))), float(rp.get(k, float("nan")))
            rep.scalar("confidence %s" % k, got, want)
    print("\nport heads: plddt=%.4f ptm=%.4f iptm=%.4f"
          % (float(conf.get("plddt", float("nan"))), float(conf.get("ptm", float("nan"))),
             float(conf.get("iptm", float("nan")))))

    npass, ntot = rep.show()
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    raise SystemExit(main())
