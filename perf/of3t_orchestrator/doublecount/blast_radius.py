#!/usr/bin/env python3
"""of3t-orchestrator pass 395: D242 is a double count, it is in ref_grad.py:201, and it reaches EVERY arm the campaign has injected -- but not equally.

`of3t-frameself` root-caused D242 and the derivation is correct; I re-derived it independently
before writing this. The last block's `attn_pair_bias` reads the `z` its own `pair_stack` just
produced, so **`z_out` is an ANCESTOR of `s_out` and `(s_out, z_out)` is not a graph cut.**
With `L = f(a, b)`, `a = s_out(theta, b)`, `b = z_out(theta)`, a hook gives
`ga = df/da` and `gb = df/db + ga.da/db` -- the total derivative, which already contains the
route through `s_out`. Differentiating the surrogate `ga.a + gb.b` then yields

    ga.da/dtheta  +  df/db.db/dtheta  +  2.ga.(da/db).(db/dtheta)

against the truth's single `ga.(da/db).(db/dtheta)`: **one extra copy of the s_out <- z_out
route.** It predicts every reading the row has taken, including the ones that looked like
separate puzzles -- the 768 s-branch tensors bit-identical, the 1,968 pair-branch ones
inflated, `ds_in` exact while `dz_in` takes 1.756x, near-constant over 48 blocks because the
duplication is structural rather than arithmetic, and nearly parallel to the truth, which is
the 0.948-0.998 cos and the ten degrees.

**The line is `perf/of3t_trunkg043/ref_grad.py:201`**:

    loss = (s_out.to(torch.float64) * cot_s).sum() + (z_out.to(torch.float64) * cot_z).sum()

so this is not confined to the model frame. **Every arm the campaign has ever driven through
`ref_grad.py` carries it**, including the whole frame384 softmax ladder. What differs is
whether the arm's REFERENCE carries it too, and that is the entire blast-radius question.

**It does, for the comparisons the campaign relies on.** `FRAME_*.json`'s `refs` block records a
`policy` for `REF_LOCAL_f64_n384` ("f64") and `REF_LOCAL_bf16_n384` ("bf16auto") -- both are
`ref_grad.py` arms on the same boundary and the same captured cotangent -- while `REF_MODEL_f64`
carries `policy: null` and is `grads_f64_043.pt`, a real full-model backward with no injection.
**So the campaign's own `MATCHED/` versus `CROSSFRAME_` naming has been separating valid from
invalid comparisons all along, for a reason nobody had named.**

**And a caveat on the surviving class that is sharper than the old one.** A common-mode defect
does not cancel in a rel_l2: both sides read `truth + extra`, so the quotient is a like-for-like
comparison of two arms computing THE SAME FUNCTIONAL -- but that functional is not the model's
gradient, it is the gradient plus one duplicated `s <- z` route. The duplicated route is the
PAIR-BIAS coupling, so these readings are taken on a functional that over-weights exactly the
pair path. That is R151's operating-point caveat with a mechanism attached, and it now applies
to the softmax ladder as well as to R143.

**The path from here is short and needs no new instrument.** The repair is one line: inject the
PARTIAL `df/db` rather than the total `dL/db`, i.e. `cot_z - autograd.grad(outputs=s_out,
grad_outputs=cot_s, inputs=z_out)`, which is exactly the break control the row is already
running. Then re-score the existing arms with the existing scorer and read pass 394's
pre-registered ladder. No new bar, no new capture, no new device time.

Zero card. Reads committed artifacts and source.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SURVIVES = {
    "what": "both legs injected through ref_grad.py on the same boundary and cotangent, so the "
            "duplicated route is common-mode and the quotient compares two arms computing the "
            "same functional",
    "readings": [
        "every MATCHED/ reading in FRAME_*.json -- ours_vs_REF_LOCAL_f64_n384, "
        "ours_vs_REF_LOCAL_bf16_n384 and the floor between the two references",
        "R149's whole softmax table: device 0.702981502944001, verb 0.4179981990834974, "
        "verb+module-wide 0.41752141981218177, route 0.5605347900452246",
        "the 1.0525x ceiling and D245's 34.25 %",
        "the A/A floors and the two cross-board controls, which are identity checks",
        "of3t-twoside's two-sided 1.7997765325758555 -- both legs injected, same cotangent",
    ],
    "but": "the functional is the gradient PLUS one duplicated s <- z route, which is the "
           "pair-bias coupling, so every one of these is read on a functional that "
           "over-weights the pair path. Valid as a comparison; not yet a statement about the "
           "true gradient.",
}

INVALID = {
    "what": "an injected arm against grads_f64_043.pt, which is a real full-model backward and "
            "does not double count",
    "readings": [
        "every CROSSFRAME_ours_vs_grads_f64_043 reading",
        "the clause's 1.7814428090278143x -- already withdrawn at pass 389, now explained",
        "the trunk's 2.9702x and block 47's 4.1613x -- already withdrawn",
        "D242's own 0.7945281613194305, which is no longer an unexplained defect but the "
        "measured size of the duplication",
    ],
    "and": "none of these needs a new retraction. frameself's pass-389 sort had already put "
           "them here; what pass 395 adds is the reason.",
}


def main() -> int:
    with open(os.path.join(ROOT, "perf/of3t_f64route/FRAME_ROUTE_HF.json")) as fh:
        refs = json.load(fh)["refs"]

    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D242",
        "root_caused_by": "of3t-frameself",
        "algebra_independently_rederived_here": True,
        "device_involved": False,
        "why_no_aiclk": "CPU only; reads committed artifacts and source",
        "THE_LINE": {
            "file": "perf/of3t_trunkg043/ref_grad.py",
            "line": 201,
            "code": "loss = (s_out.to(torch.float64) * cot_s).sum() + "
                    "(z_out.to(torch.float64) * cot_z).sum()",
            "why_it_is_wrong": "a multi-output cotangent injection is only valid when the "
                               "outputs form a GRAPH CUT. z_out is an ancestor of s_out, so the "
                               "hooked cot_z is a TOTAL derivative that already contains the "
                               "route through s_out, and the surrogate adds that route a second "
                               "time.",
            "so_the_reach_is": "every arm the campaign has driven through ref_grad.py, not only "
                               "the model frame -- the frame384 softmax ladder included.",
        },
        "THE_EVIDENCE_THAT_SORTS_IT": {
            "refs_block_from_FRAME_ROUTE_HF": refs,
            "read": "REF_LOCAL_f64_n384 and REF_LOCAL_bf16_n384 carry a `policy`, so both are "
                    "ref_grad.py arms and both double count. REF_MODEL_f64 carries policy null "
                    "and is grads_f64_043.pt, a real full-model backward that does not.",
        },
        "SURVIVES": SURVIVES,
        "INVALID": INVALID,
        "THE_NAMING_WAS_ALREADY_RIGHT":
            "the artifacts' MATCHED/ versus CROSSFRAME_ prefixes separate exactly these two "
            "classes, and have since before the mechanism was known. The campaign has been "
            "protected by a convention it could not justify; it can justify it now.",
        "THE_PATH_FROM_HERE": {
            "1": "frameself's break control confirms the repair -- subtract "
                 "autograd.grad(outputs=s_out, grad_outputs=cot_s, inputs=z_out) from cot_z and "
                 "re-inject; block 47 should go from 0.7849281738435908 to the 1e-12 bar",
            "2": "fix ref_grad.py:201 to inject the PARTIAL df/db rather than the total dL/db",
            "3": "re-score the existing arms with the existing scorer -- no new capture, no new "
                 "device time",
            "4": "read pass 394's pre-registered ladder. NO NEW BAR.",
            "cost": "the instrument work is one line and the re-scores reuse banked artifacts.",
        },
        "DOESNOT": "this measures nothing new. It re-derives the row's algebra independently, "
                   "locates the line, and sorts which published readings the defect invalidates "
                   "and which it leaves as valid comparisons on a modified functional.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
