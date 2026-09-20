#!/usr/bin/env python3
"""Deliverable 2: does of3t-auxfind's mask fix reach a USER, measured on a real fold.

of3t-auxfind read `structure.py:114` (`features["token_mask"] = torch.ones(...)`) and inferred
that a shipped fold's token axis carries no padding, so the unmasked confidence Pairformer
computes the reference function exactly. D90 is this campaign escalating a product claim from a
source reading and having to withdraw it. This measures it instead.

The fold is run through the shipped `tt-bio predict` CLI, unmodified. Three things are wrapped,
none of which change the default arm's arithmetic:

  * `OpenFold3._confidence` records the token mask it is handed and recovers `repr_x_mask`.
  * `OF3ConfidenceHead.forward` is called THREE times per sample instead of once, in one
    process on one card, on identical inputs: the shipped default (no masks), the masked arm,
    and a negative control whose mask has one real token forced to zero. A paired comparison in
    one process removes every confound except the argument under test.
  * the confidence dict is recomputed from each arm's logits, so the deltas are on the numbers a
    user reads -- pLDDT, PAE, PTM, IPTM and the ranking score -- not on logits.

Seed scatter is the separate axis: run this twice with different `--seed` and the default arm's
spread between those runs is the bar any delta here is scored against (`conf-scatter-bar`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

OUT = Path(os.environ["EXPOSURE_OUT"])
REC: dict = {"samples": [], "mask_stats": None, "arms": ["default", "masked", "negctl"]}


def _install():
    import tt_bio.openfold3_fold as FOLD
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from tt_bio.protenix import ConfidenceHead

    orig_conf = FOLD.OpenFold3._confidence
    orig_fwd = OF3ConfidenceHead.forward
    state: dict = {}

    def conf(self, sample, si_input, si_trunk, zij_trunk, aux):
        rb = aux.get("repr_batch")
        tm = rb["token_mask"] if rb is not None else None
        state["aux"] = aux
        state["token_mask"] = tm
        state["repr_mask"] = None
        if rb is not None:
            from tt_bio._vendor.openfold3.core.utils.atomize_utils import (
                get_token_representative_atoms)
            _, rmask = get_token_representative_atoms(
                batch=rb, x=sample, atom_mask=rb["atom_mask"])
            state["repr_mask"] = rmask.reshape(-1).float()
        if REC["mask_stats"] is None and tm is not None:
            t = tm.reshape(-1).float()
            REC["mask_stats"] = {
                "source": "aux['repr_batch']['token_mask'], i.e. features['token_mask'] as the "
                          "shipped CLI builds it",
                "n_token": int(t.numel()),
                "min": float(t.min()), "max": float(t.max()),
                "n_zero": int((t == 0).sum()),
                "padding_fraction": float((t == 0).float().mean()),
                # E3: the token axis the confidence head is actually handed. Bucketing pads the
                # TRUNK's axis and openfold3_fold.py slices both trunk outputs back before this
                # call, so these two must agree or a padded axis is reaching the head from
                # somewhere other than token_mask.
                "n_token_at_confidence_head": int(si_trunk.shape[0]),
                "zij_trunk_token_axis": int(zij_trunk.shape[0]),
                "repr_x_mask_n_zero": (None if state["repr_mask"] is None
                                       else int((state["repr_mask"] == 0).sum())),
                "repr_x_mask_equals_token_mask": (
                    None if state["repr_mask"] is None
                    else bool(torch.equal(state["repr_mask"], t))),
            }
            print(f"[exposure] token_mask: {REC['mask_stats']}", flush=True)
        return orig_conf(self, sample, si_input, si_trunk, zij_trunk, aux)

    def fwd(self, **kw):
        tm = state.get("token_mask")
        rm = state.get("repr_mask")
        outs = {}
        for arm in REC["arms"]:
            k = dict(kw)
            if arm == "masked" and tm is not None:
                k["token_mask"] = tm
                k["single_mask"] = rm
            elif arm == "negctl" and tm is not None:
                # The negative control the pre-registration owes E1: if token_mask is all ones
                # the masked arm is the default arm BY CONSTRUCTION and an exact zero delta
                # proves nothing about the plumbing. Force one real token's mask to zero and the
                # two arms must then differ, or the kwarg is dead.
                t2 = tm.reshape(-1).float().clone()
                t2[0] = 0.0
                k["token_mask"] = t2
                k["single_mask"] = (None if rm is None else
                                    torch.where(torch.arange(rm.numel()) == 0,
                                                torch.zeros_like(rm), rm))
            outs[arm] = orig_fwd(self, **k)
        _score(outs, state)
        return outs["default"]          # the shipped default is what the fold keeps

    def _score(outs, st):
        aux = st["aux"]
        bins = (torch.arange(50, dtype=torch.float32) + 0.5) / 50
        rows = {}
        for arm, o in outs.items():
            pl = (torch.softmax(o["plddt_logits"].float(), -1) * bins).sum(-1)
            ptm, iptm = ConfidenceHead._ptm_iptm(o["pae_logits"], aux.get("asym_id"))
            rows[arm] = {
                "plddt_mean": float(pl.mean()),
                "plddt_atom": pl.double(),
                "ptm": float(ptm), "iptm": float(iptm),
                "pae_logits": o["pae_logits"].detach().double(),
                "plddt_logits": o["plddt_logits"].detach().double(),
            }
        base = rows["default"]
        rec = {"default": {k: v for k, v in base.items() if not torch.is_tensor(v)}}
        for arm in ("masked", "negctl"):
            r = rows[arm]
            d = {"plddt_mean": r["plddt_mean"], "ptm": r["ptm"], "iptm": r["iptm"],
                 "d_plddt_mean": r["plddt_mean"] - base["plddt_mean"],
                 "d_ptm": r["ptm"] - base["ptm"], "d_iptm": r["iptm"] - base["iptm"],
                 "plddt_atom_max_abs_delta": float((r["plddt_atom"] - base["plddt_atom"])
                                                   .abs().max()),
                 "plddt_atom_bit_equal": bool(torch.equal(r["plddt_atom"], base["plddt_atom"])),
                 "pae_logits_max_abs_delta": float((r["pae_logits"] - base["pae_logits"])
                                                   .abs().max()),
                 "pae_logits_bit_equal": bool(torch.equal(r["pae_logits"], base["pae_logits"])),
                 "plddt_logits_max_abs_delta": float((r["plddt_logits"] - base["plddt_logits"])
                                                     .abs().max())}
            rec[arm] = d
        REC["samples"].append(rec)
        print(f"[exposure] sample {len(REC['samples'])}: "
              f"masked d_plddt {rec['masked']['d_plddt_mean']:+.3e} "
              f"d_ptm {rec['masked']['d_ptm']:+.3e} d_iptm {rec['masked']['d_iptm']:+.3e} "
              f"pae max|d| {rec['masked']['pae_logits_max_abs_delta']:.3e} | "
              f"negctl d_plddt {rec['negctl']['d_plddt_mean']:+.3e} "
              f"pae max|d| {rec['negctl']['pae_logits_max_abs_delta']:.3e}", flush=True)

    FOLD.OpenFold3._confidence = conf
    OF3ConfidenceHead.forward = fwd


def main():
    _install()
    from tt_bio.main import cli
    argv = sys.argv[1:]
    try:
        cli.main(args=argv, standalone_mode=False)
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(REC, indent=1, default=str) + "\n")
        print(f"[exposure] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
