#!/usr/bin/env python3
"""Score `tt_bio.rf3.confidence` against upstream's own reduction code.

The reductions are pure functions of the confidence logits, so this needs no device and
no fold: seeded synthetic logits over a REAL fixture's features exercise every path
(chain grouping, symmetric interface masks, the non-uniform bin midpoints, pTM's d0).

Two arms, because upstream's `rf3.utils.predicted_error` is not part of what tt-bio
vendors -- it lives only in the reference env:

    <tt-bio env>   parity_confidence_metrics.py --arm mine     --work /tmp/rf3_conf
    <ref env>      parity_confidence_metrics.py --arm upstream --work /tmp/rf3_conf

The second arm prints the comparison and exits non-zero on any disagreement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = "two_protein_chains"     # two chains, so the interface blocks are non-empty


def synth_logits(n_token: int, seed: int = 0):
    """Seeded logits of the head's real shapes. pde is symmetrised, as the head does."""
    g = torch.Generator().manual_seed(seed)
    pde = torch.randn(n_token, n_token, 64, generator=g)
    return {
        "plddt_logits": torch.randn(n_token, 23 * 50, generator=g) * 2,
        "pae_logits": torch.randn(n_token, n_token, 64, generator=g),
        "pde_logits": pde + pde.transpose(0, 1),
    }


def arm_mine(work: Path) -> int:
    sys.path.insert(0, str(REPO))
    from tt_bio.rf3 import confidence
    from tt_bio.rf3.featurize import featurize

    d = REPO / "scripts/rf3_port/parity_artifacts" / FIXTURE
    prev = os.getcwd()
    os.chdir(d)
    try:
        out = featurize("input.json", n_recycles=1, diffusion_batch_size=1, seed=42)[0]
    finally:
        os.chdir(prev)
    f, gt = out["feats"], out["ground_truth"]
    is_real_atom = out["confidence_feats"]["is_real_atom"]
    n_token = int(f["asym_id"].shape[-1])
    logits = synth_logits(n_token)
    aa = out["atom_array"]
    coord = aa.coord.copy()

    got = confidence.summary(logits, f, is_real_atom,
                             gt["chain_iid_token_lvl"], aa, coord)
    # a second coord arm that definitely clashes: both chains on top of each other
    clashed = coord.copy()
    import numpy as np
    pn = np.asarray(aa.pn_unit_id)
    clashed[pn == sorted(set(pn.tolist()))[0]] = clashed[pn == sorted(set(pn.tolist()))[1]][0]
    got["__clash_on_collapsed_coords__"] = confidence.has_clash(aa, clashed)

    work.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits, "asym_id": f["asym_id"],
                "is_real_atom": is_real_atom,
                "chain_iid_token_lvl": gt["chain_iid_token_lvl"],
                "atom_array": aa, "coord": coord, "coord_clashed": clashed},
               work / "inputs.pt")
    (work / "mine.json").write_text(json.dumps(got, indent=2, default=str) + "\n")
    print(json.dumps(got, indent=2, default=str))
    return 0


def arm_upstream(work: Path) -> int:
    from omegaconf import OmegaConf
    from rf3.metrics.clashing_chains import CountClashingChains
    from rf3.metrics.predicted_error import compute_ptm
    from rf3.utils.predicted_error import compile_af3_style_confidence_outputs

    st = torch.load(work / "inputs.pt", weights_only=False)
    lg, asym = st["logits"], st["asym_id"].reshape(-1)
    cfg = OmegaConf.create({
        "plddt": {"n_bins": 50, "max_value": 1.0},
        "pae": {"n_bins": 64, "max_value": 32},
        "pde": {"n_bins": 64, "max_value": 32}})
    ref = compile_af3_style_confidence_outputs(
        plddt_logits=lg["plddt_logits"].unsqueeze(0),
        pae_logits=lg["pae_logits"].unsqueeze(0),
        pde_logits=lg["pde_logits"].unsqueeze(0),
        chain_iid_token_lvl=st["chain_iid_token_lvl"],
        is_real_atom=st["is_real_atom"],
        atom_array=st["atom_array"],
        confidence_loss_cfg=cfg)
    want = dict(ref["summary_confidences"])
    want["ptm"] = float(compute_ptm(lg["pae_logits"].unsqueeze(0), None)[0])
    cross = (asym[None, :] != asym[:, None])
    want["iptm"] = (float(compute_ptm(lg["pae_logits"].unsqueeze(0), cross)[0])
                    if bool(cross.any()) else None)

    from biotite.structure import AtomArrayStack, stack
    clash_metric = CountClashingChains()

    def up_clash(coord):
        aa = st["atom_array"].copy()
        aa.coord = coord
        s: AtomArrayStack = stack([aa])
        return bool(clash_metric.compute(
            X_L=torch.zeros(1, aa.array_length(), 3),
            predicted_atom_array_stack=s)["has_clash_0"])

    want["has_clash"] = up_clash(st["coord"])
    want["__clash_on_collapsed_coords__"] = up_clash(st["coord_clashed"])

    got = json.loads((work / "mine.json").read_text())
    bad = []
    for k, w in want.items():
        g = got.get(k, "<missing>")
        same = (json.dumps(g, default=str, sort_keys=True)
                == json.dumps(w, default=str, sort_keys=True))
        if not same and isinstance(w, float) and isinstance(g, (int, float)):
            same = abs(float(g) - float(w)) <= 1e-6 * max(1.0, abs(float(w)))
        print(f"{'ok  ' if same else 'DIFF'} {k}\n     mine {g}\n     ref  {w}")
        if not same:
            bad.append(k)
    extra = sorted(set(got) - set(want))
    if extra:
        print(f"keys mine has that the reference does not: {extra}")
    print(f"\n{len(want) - len(bad)}/{len(want)} keys agree")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("mine", "upstream"), required=True)
    ap.add_argument("--work", default="/tmp/rf3_conf")
    a = ap.parse_args()
    return arm_mine(Path(a.work)) if a.arm == "mine" else arm_upstream(Path(a.work))


if __name__ == "__main__":
    sys.exit(main())
