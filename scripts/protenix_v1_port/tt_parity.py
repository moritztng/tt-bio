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


def pcc(a, b):
    a = torch.as_tensor(a).float().flatten()
    b = torch.as_tensor(b).float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    if a.numel() == 0:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm())
    return float("nan") if d == 0 else float((a @ b) / d)


def rel(a, b):
    a = torch.as_tensor(a).float().flatten(); b = torch.as_tensor(b).float().flatten()
    n = min(a.numel(), b.numel())
    return float((a[:n] - b[:n]).norm() / (b[:n].norm() + 1e-12))


class Report:
    def __init__(self, threshold):
        self.rows = []
        self.threshold = threshold

    def add(self, name, got, want, note=""):
        if got is None or want is None:
            self.rows.append((name, float("nan"), float("nan"), "MISSING " + note))
            return
        p, r = pcc(got, want), rel(got, want)
        self.rows.append((name, p, r, note))

    def show(self):
        print("\n%-34s %9s %9s  %-6s %s" % ("module", "PCC", "relerr", "", "note"))
        npass = 0
        for name, p, r, note in self.rows:
            ok = p == p and p >= self.threshold
            npass += ok
            print("%-34s %9.6f %9.2e  %-6s %s" % (name, p, r, "PASS" if ok else "FAIL", note))
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
    o_atxE, o_atxD = d.atxE, d.atxD
    d.atxE = lambda *a, **k: dtaps.setdefault("q", o_atxE(*a, **k))
    d.atxD = lambda *a, **k: dtaps.setdefault("qd", o_atxD(*a, **k))
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
    if "qd" in dtaps and "diff_atom_decoder" in cap:
        rep.add("diffusion atom decoder", dev(dtaps["qd"]), cap["diff_atom_decoder"][0],
                "shape-aligned prefix")
    rep.add("denoise (x0 at sigma=%g)" % float(ref["t_hat"][0]), denoised, ref["denoised"])

    # ---- confidence heads ------------------------------------------------------------------
    conf = m.confidence_head.confidence(aux["s_inputs"], aux["s_trunk"], aux["z_trunk"],
                             denoised.reshape(-1, 3), feats)
    rc = ref["confidence"]
    print("\nref confidence keys:", sorted(rc))
    print("port confidence keys:", sorted(conf))
    for pk, rk in (("pae", "pae"), ("pde", "pde"), ("plddt_atom", "plddt")):
        if pk in conf and rk in rc:
            rep.add("confidence %s" % pk, conf[pk], rc[rk])
    print("\nport heads: plddt=%.4f ptm=%.4f iptm=%.4f"
          % (float(conf.get("plddt", float("nan"))), float(conf.get("ptm", float("nan"))),
             float(conf.get("iptm", float("nan")))))

    npass, ntot = rep.show()
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    raise SystemExit(main())
