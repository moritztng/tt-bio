#!/usr/bin/env python3
"""PXDesign design-featurizer value-parity scorer (card-free, CPU-only, no upstream install).

Scores `tt_bio.pxdesign.featurize` against a COMMITTED capture of the upstream PXDesign
featurizer on the PD-L1 quick-start target (`parity_artifacts/pdl1/`, 196 tokens, 1250
atoms). Every comparison is bit-exact -- these are integer bins, boolean masks and one-hots,
so there is no tolerance to argue about.

Three arms:

  1. `conditional_templ` / `conditional_templ_mask` from the captured inputs to upstream's
     own `get_condition_template_feature`. This is the port's sharpest edge.
  2. The `xpb` arm. The binder placeholder must be excluded from the conditioning, and
     nothing downstream complains if it is not -- the model just gets told the answer and
     returns a plausible structure. So the gate asserts it directly: the conditioned
     sub-block must be exactly the resolved non-`xpb` tokens, and re-running with the
     placeholder NOT excluded must produce a DIFFERENT feature. An arm that cannot fail on
     the bug it is named after is not an arm.
  3. `restype`, 36-way, recovered from the captured one-hot's own column order.

What this gate does NOT cover, stated so it is not mistaken for coverage: the atom-array
construction upstream of these functions (CIF parse, tokenization, crop, hotspot annotation).
It scores the design-specific arithmetic on captured inputs. `hotspot` and the token-level
identity keys are captured and shape-checked but not yet recomputed by a tt-bio path.

    python3 scripts/pxdesign_port/parity_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ART = REPO / "scripts" / "pxdesign_port" / "parity_artifacts" / "pdl1"
REF_F = ART / "ref_design_f.pt"
REF_IN = ART / "ref_condition_inputs.pt"
META = ART / "ref_design_f.meta.json"


def featurizer_parity() -> dict:
    import torch
    sys.path.insert(0, str(REPO))
    from tt_bio.pxdesign.featurize import (RESTYPE_VOCAB, condition_template,
                                           condition_template_index, restype_onehot)

    for p in (REF_F, REF_IN, META):
        if not p.exists():
            return {"mode": "pxdesign_featurizer", "verdict": "ERROR",
                    "error": f"missing committed reference {p}"}

    ref = torch.load(REF_F, weights_only=False)
    cin = torch.load(REF_IN, weights_only=False)
    meta = json.loads(META.read_text())

    checks, mismatches = [], []

    def bitexact(name, got, want):
        ok = (got.dtype == want.dtype and got.shape == want.shape
              and bool(torch.equal(got, want)))
        checks.append(name)
        if not ok:
            mismatches.append({
                "key": name,
                "got": [list(got.shape), str(got.dtype)],
                "want": [list(want.shape), str(want.dtype)],
                "n_differ": (int((got != want).sum()) if got.shape == want.shape else None),
            })
        return ok

    # 1. the conditioning distogram, from upstream's own inputs
    out = condition_template(cin["coord"], cin["res_name"], cin["mol_type"],
                             cin["is_resolved"])
    bitexact("conditional_templ", out["conditional_templ"], ref["conditional_templ"])
    bitexact("conditional_templ_mask", out["conditional_templ_mask"],
             ref["conditional_templ_mask"])

    # the 65-row lookup index, the form the model actually consumes
    idx = condition_template_index(ref["conditional_templ"], ref["conditional_templ_mask"])
    checks.append("condition_template_index")
    if int(idx.min()) != 0 or int(idx.max()) > 64:
        mismatches.append({"key": "condition_template_index",
                           "error": f"index range [{int(idx.min())}, {int(idx.max())}] "
                                    f"outside the embedding's 65 rows"})

    # 2. the xpb arm
    is_xpb = torch.tensor([r == "xpb" for r in cin["res_name"]])
    expect = (~is_xpb) & torch.as_tensor(cin["is_resolved"]).bool()
    got_rows = out["conditional_templ_mask"].any(dim=1)
    checks.append("xpb_excluded_from_conditioning")
    if not bool(torch.equal(got_rows, expect)):
        mismatches.append({"key": "xpb_excluded_from_conditioning",
                           "error": f"conditioned rows {int(got_rows.sum())} != resolved "
                                    f"non-xpb tokens {int(expect.sum())}"})
    # the arm must be able to fail: not excluding the placeholder has to change the feature
    leaked = condition_template(cin["coord"], ["ALA" if r == "xpb" else r
                                               for r in cin["res_name"]],
                                cin["mol_type"], cin["is_resolved"])
    checks.append("xpb_arm_is_sensitive")
    if bool(torch.equal(leaked["conditional_templ"], out["conditional_templ"])):
        mismatches.append({"key": "xpb_arm_is_sensitive",
                           "error": "leaking the binder placeholder into the conditioning "
                                    "produced an identical feature, so this arm proves "
                                    "nothing on this fixture"})

    # 3. restype, 36-way
    want_rt = ref["restype"]
    checks.append("restype_vocab_width")
    if want_rt.shape[-1] != len(RESTYPE_VOCAB):
        mismatches.append({"key": "restype_vocab_width",
                           "got": want_rt.shape[-1], "want": len(RESTYPE_VOCAB)})
    else:
        names = [RESTYPE_VOCAB[i] for i in want_rt.argmax(dim=-1).tolist()]
        bitexact("restype", restype_onehot(names), want_rt)

    n_xpb = int(is_xpb.sum())
    return {
        "mode": "pxdesign_featurizer",
        "verdict": "PASS" if not mismatches else "FAIL",
        "fixture": meta["yaml"],
        "n_token": meta["n_token"], "n_atom": meta["n_atom"],
        "n_binder_placeholder_tokens": n_xpb,
        "n_conditioned_tokens": int(expect.sum()),
        "upstream_keys_captured": len(meta["all_keys"]),
        "checks_total": len(checks), "checks_passed": len(checks) - len(mismatches),
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    r = featurizer_parity()
    print(json.dumps(r, indent=1))
    sys.exit(0 if r.get("verdict") == "PASS" else 1)
