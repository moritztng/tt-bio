#!/usr/bin/env python3
"""The reference's OWN bf16 ceiling for the atom encoder.

The parity harness was scoring against the pair track's device error, which is the
floor for C_L/P_LL but not for a 3-block attention stack that accumulates on top of
them. This measures the real roof the same way the rest of this port does: run the
vendored reference encoder on identical inputs in fp32 and under cpu-bf16 autocast,
and take the disagreement between torch and itself.

A device number at or below this is at the ceiling. Above it is the port.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SEL = {"template": dict(template_selection=["9dfn_A"]),
       "cyclic": dict(cyclic_chains=["A"])}


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture", default="rna")
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    from tt_bio.rf3.weights import load_reference

    d = REPO / "scripts/rf3_port/parity_artifacts" / args.fixture
    prev = os.getcwd(); os.chdir(d)
    try:
        out = featurize("input.json", n_recycles=2, diffusion_batch_size=1, seed=42,
                        **SEL.get(args.fixture, {}))[0]
    finally:
        os.chdir(prev)

    net, _ = load_reference(args.ckpt, num_steps=2)
    enc = net.feature_initializer.input_feature_embedder.atom_attention_encoder

    def run(bf16):
        f = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
             for k, v in out["feats"].items()}
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            a, q, c, p = enc(f, None, None, None)
        return [t.float() for t in (a, q, c, p)]

    hi = run(False)
    lo = run(True)
    names = ["A_I", "Q_L", "C_L", "P_LL"]
    rep = {"fixture": args.fixture, "L": int(hi[2].shape[0]),
           "ceiling": {n: {"rel_rms": round(rel_rms(l, h), 6),
                           "pcc": round(pcc(l, h), 7)}
                       for n, h, l in zip(names, hi, lo)}}
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
