#!/usr/bin/env python3
"""Deliverable 2: does of3t-auxfind's mask fix reach a USER, measured on a real fold.

of3t-auxfind read `structure.py:114` (`features["token_mask"] = torch.ones(...)`) and inferred
that a shipped fold's token axis carries no padding, so the unmasked confidence Pairformer
computes the reference function exactly. D90 is this campaign escalating a product claim from a
source reading and having to withdraw it. This measures it instead.

The fold runs through the shipped `tt-bio predict` CLI, unmodified. Three things are wrapped,
none of which change the default arm's arithmetic:

  * `OpenFold3._confidence` records the token mask it is handed and recovers `repr_x_mask`.
  * `OF3ConfidenceHead.forward` is called THREE times per sample instead of once, in one
    process on one card, on identical inputs: the shipped default (no masks), the masked arm,
    and a negative control whose mask has one real token forced to zero. A paired comparison in
    one process removes every confound except the argument under test. The default arm's return
    value is what the fold keeps, so the structures and the results.json are the shipped ones.
  * the confidence dict is recomputed from each arm's logits, so the deltas land on the numbers
    a user reads -- pLDDT, PAE, PTM, IPTM and the ranking score -- not on logits.

`tt-bio predict` runs the fold in an `mp.get_context("spawn")` CHILD, so patching in the parent
reaches nothing (fleet memory `in-process-patch-never-reaches-a-spawn-child`, measured here on
the first attempt: a completed 89.8 s fold with an empty record). `exposure_hook/sitecustomize.py`
is what arms this, because `site` imports it in every interpreter including a spawned one.

Seed scatter is the separate axis: run the whole thing twice with different `--seed`, and the
default arm's spread between those runs is the bar any delta here is scored against
(`conf-scatter-bar`).
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import torch

ARMS = ("default", "masked", "negctl")
REC: dict = {"samples": [], "mask_stats": None, "arms": list(ARMS), "pid": os.getpid()}
_STATE: dict = {}


def dump():
    out = os.environ.get("EXPOSURE_OUT")
    if not out or REC["mask_stats"] is None:
        return                      # not the fold child, nothing measured here
    p = Path(out)
    p = p.with_name(f"{p.stem}{p.suffix}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(REC, indent=1, default=str) + "\n")
    print(f"[exposure] wrote {p} from pid {os.getpid()}", flush=True)


def _bucket_state(n_token):
    """What the token-axis bucket did on THIS fold, so E3 is not a vacuous check."""
    try:
        from tt_bio.token_axis import bucket_enabled, bucket_multiple, pad_amount
        on = bool(bucket_enabled())
        mult = int(bucket_multiple("openfold3"))
        pad = int(pad_amount(n_token, mult)) if on else 0
        return {"bucket_enabled": on, "bucket_multiple": mult,
                "trunk_token_axis_padded_to": n_token + pad, "trunk_tok_pad": pad}
    except Exception as e:                                   # never break a fold to log
        return {"bucket_state_error": repr(e)}


def install():
    """Idempotent. Called by the sitecustomize hook once both modules are importable."""
    if _STATE.get("installed"):
        return
    _STATE["installed"] = True
    import tt_bio.openfold3_fold as FOLD
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    # tt_bio.protenix is imported lazily inside _score, exactly as the shipped _confidence does
    # it (openfold3_fold.py:300). Importing it here arms during its own initialisation.

    orig_conf = FOLD.OpenFold3._confidence
    orig_fwd = OF3ConfidenceHead.forward

    def conf(self, sample, si_input, si_trunk, zij_trunk, aux):
        rb = aux.get("repr_batch")
        tm = rb["token_mask"] if rb is not None else None
        _STATE["aux"] = aux
        _STATE["token_mask"] = tm
        _STATE["repr_mask"] = None
        if rb is not None:
            from tt_bio._vendor.openfold3.core.utils.atomize_utils import (
                get_token_representative_atoms)
            _, rmask = get_token_representative_atoms(
                batch=rb, x=sample, atom_mask=rb["atom_mask"])
            _STATE["repr_mask"] = rmask.reshape(-1).float()
        if REC["mask_stats"] is None and tm is not None:
            t = tm.reshape(-1).float()
            rm = _STATE["repr_mask"]
            REC["mask_stats"] = {
                "source": "aux['repr_batch']['token_mask'] == features['token_mask'] as the "
                          "shipped CLI builds it",
                "n_token": int(t.numel()),
                "min": float(t.min()), "max": float(t.max()),
                "n_zero": int((t == 0).sum()),
                "padding_fraction": float((t == 0).float().mean()),
                # E3: the token axis the confidence head is actually handed. Bucketing pads the
                # TRUNK's axis and openfold3_fold.py slices both trunk outputs back before this
                # call, so these must agree with n_token or a padded axis is reaching the head
                # from somewhere other than token_mask.
                "n_token_at_confidence_head": int(si_trunk.shape[0]),
                "zij_trunk_token_axis": int(zij_trunk.shape[0]),
                "repr_x_mask_n_zero": None if rm is None else int((rm == 0).sum()),
                "repr_x_mask_equals_token_mask": (None if rm is None
                                                  else bool(torch.equal(rm, t))),
                # E3 only means something if bucketing was actually ON: with it off there is no
                # trunk-side pad to slice back and the check is vacuous.
                **_bucket_state(int(t.numel())),
            }
            print(f"[exposure] token_mask: {REC['mask_stats']}", flush=True)
        return orig_conf(self, sample, si_input, si_trunk, zij_trunk, aux)

    def fwd(self, **kw):
        tm = _STATE.get("token_mask")
        rm = _STATE.get("repr_mask")
        outs = {}
        for arm in ARMS:
            k = dict(kw)
            if arm == "masked" and tm is not None:
                k["token_mask"] = tm
                k["single_mask"] = rm
            elif arm == "negctl" and tm is not None:
                # The negative control E1 owes: if token_mask is all ones the masked arm IS the
                # default arm by construction, and an exact zero delta then says nothing about
                # the plumbing. Force one real token's mask to zero; the arms must diverge, or
                # the kwarg is dead and the zero above was never evidence.
                t2 = tm.reshape(-1).float().clone()
                t2[0] = 0.0
                k["token_mask"] = t2
                k["single_mask"] = (None if rm is None else
                                    torch.where(torch.arange(rm.numel()) == 0,
                                                torch.zeros_like(rm), rm))
            outs[arm] = orig_fwd(self, **k)
        _score(outs)
        return outs["default"]          # the shipped default is what the fold keeps

    def _score(outs):
        from tt_bio.protenix import ConfidenceHead
        aux = _STATE["aux"]
        bins = (torch.arange(50, dtype=torch.float32) + 0.5) / 50
        rows = {}
        for arm, o in outs.items():
            pl = (torch.softmax(o["plddt_logits"].float(), -1) * bins).sum(-1)
            ptm, iptm = ConfidenceHead._ptm_iptm(o["pae_logits"], aux.get("asym_id"))
            rows[arm] = {"plddt_mean": float(pl.mean()), "plddt_atom": pl.double(),
                         "ptm": float(ptm), "iptm": float(iptm),
                         "pae_logits": o["pae_logits"].detach().double(),
                         "plddt_logits": o["plddt_logits"].detach().double()}
        base = rows["default"]
        rec = {"default": {"plddt_mean": base["plddt_mean"], "ptm": base["ptm"],
                           "iptm": base["iptm"]}}
        for arm in ("masked", "negctl"):
            r = rows[arm]
            rec[arm] = {
                "plddt_mean": r["plddt_mean"], "ptm": r["ptm"], "iptm": r["iptm"],
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
        REC["samples"].append(rec)
        m, n = rec["masked"], rec["negctl"]
        print(f"[exposure] sample {len(REC['samples'])}: "
              f"masked d_plddt {m['d_plddt_mean']:+.3e} d_ptm {m['d_ptm']:+.3e} "
              f"d_iptm {m['d_iptm']:+.3e} pae max|d| {m['pae_logits_max_abs_delta']:.3e} | "
              f"negctl d_plddt {n['d_plddt_mean']:+.3e} "
              f"pae max|d| {n['pae_logits_max_abs_delta']:.3e}", flush=True)

    FOLD.OpenFold3._confidence = conf
    OF3ConfidenceHead.forward = fwd
    atexit.register(dump)
    print(f"[exposure] armed in pid {os.getpid()}", flush=True)
