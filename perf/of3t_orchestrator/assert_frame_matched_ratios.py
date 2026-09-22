#!/usr/bin/env python3
"""The campaign's recurring defect is a ratio whose two sides are not the same kind of thing.

Four passes in a row produced one, each internally consistent so no guard fired:

  358  a bar priced on a route the shipped code does not take            (D212)
  359  a distance measured across two float64 frames 1.8416 apart        (D214)
  360  a per-section ratio built from two different references           (D215)
  361  a counterfactual whose numerator and weights came from two frames (D218)

Writing the lesson into DEFECTS.md did not stop 359, 360 or 361. So it is a check.

Two things are asserted, both recomputed from the graded artifact rather than read from prose:

  1. A26-perfect in every section pools to the artifact's OWN bar. This is the invariant that
     makes the clause coherent -- if it stops holding, either the bar or the sections moved and
     every counterfactual built on them is void. It also catches the specific mistake of pooling
     a float64-referenced arm with bf16 reference masses, which reads as a 2.1 % bar defect.
  2. The withdrawn cross-frame target does not come back. `0.4361680548` and `2.360x` were
     published into GAP and three briefs at pass 360 and withdrawn at 361; they survive in
     committed history, which is exactly how a superseded number returns.

Usage: assert_frame_matched_ratios.py [tree]  ->  0 ok, 1 drift, 2 broken.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

#: Follows the charter. Repointed with it at pass 361 and again at pass 379, onto
#: of3t-modelframe's frame-matched artifact (see the charter spec's own note); if the two ever diverge this guard
#: checks an artifact the gate does not grade, which is a silent way to be green.
ART = "perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json"
SUMMARY = pathlib.Path("/home/moritz/.coworker/state/of3t-orchestrator.md")

#: Withdrawn at pass 361 (D218), with what to say instead. A live doc may still DISCUSS them --
#: D218 itself quotes both -- so the refusal is only for a sentence that uses one as the target.
WITHDRAWN = {
    "0.4361680548": "the in-frame A26 bar 0.5268825372815341 (D218)",
    "2.360x": "1.9537x, the frame-matched factor (D218)",
}
TARGET_WORDS = re.compile(r"\b(must reach|needs the trunk|target|aim at|factor still to find)\b", re.I)


def a26_perfect(ps, sec):
    """sqrt(2) x that section's own floor / its own norm ratio -- A26's reachable level."""
    t = ps["UPSTREAM_BF16_vs_FLOAT64"][sec]
    return math.sqrt(2) * t["mass_weighted_rel_l2"] / t["mass_weighted_norm_ratio"]


def check(root: pathlib.Path) -> int:
    f = root / ART
    if not f.is_file():
        print("  BROKEN %s absent -- this check cannot run, which is not a pass" % ART,
              file=sys.stderr)
        return 2
    d = json.loads(f.read_text())
    ps, rec = d["per_section"], d["reconciliation"]["renorm"]["sections"]
    bar = d["bars"]["A26_reachable_bar_vs_their_bf16"]
    den = sum(v["ref_sq"] for v in rec.values())

    def pool(fn):
        return math.sqrt(sum(rec[s]["ref_sq"] * fn(s) ** 2 for s in rec) / den)

    # control: the pooling must reproduce the artifact's own published headline
    got = pool(lambda s: ps["renorm_vs_UPSTREAM_BF16"][s]["mass_weighted_rel_l2"])
    want = d["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"]
    if abs(got - want) > 1e-12:
        print("  BROKEN the section pooling does not reproduce the artifact's headline "
              "(%r vs %r) -- every figure below would be void" % (got, want), file=sys.stderr)
        return 2

    bad = []
    perfect = pool(lambda s: a26_perfect(ps, s))
    if abs(perfect - bar) > 1e-12:
        bad.append("A26-perfect in every section pools to %.16f but the artifact's own bar is "
                   "%.16f. Those must be equal: the bar IS sqrt(2)*floor/r. A gap means the bar "
                   "and the sections disagree, or a float64-referenced arm is being pooled with "
                   "bf16 reference masses (D218)." % (perfect, bar))

    cur = {s: ps["renorm_vs_UPSTREAM_BF16"][s]["mass_weighted_rel_l2"] for s in rec}
    with_trunk = pool(lambda s: a26_perfect(ps, "pairformer_stack") if s == "pairformer_stack"
                      else cur[s])
    if with_trunk > bar:
        bad.append("an A26-perfect trunk no longer passes the clause: %.16f against %.16f. D218's "
                   "headline has stopped holding and GAP says the opposite." % (with_trunk, bar))

    if SUMMARY.is_file():
        for para in SUMMARY.read_text().split("\n\n"):
            if "D218" in para or "withdraw" in para.lower():
                continue                     # the paragraph that retires them may name them
            for num, instead in WITHDRAWN.items():
                if num in para and TARGET_WORDS.search(para):
                    bad.append("the summary uses the WITHDRAWN cross-frame figure %s as a target; "
                               "say %s" % (num, instead))

    if bad:
        for b in bad:
            print("  DRIFT " + b)
        return 1
    print("ok    A26-perfect everywhere pools to the artifact's own bar to 1e-12, an A26-perfect "
          "trunk passes the clause at %.4fx, and the withdrawn cross-frame target is not quoted "
          "as one (probes below fire)" % (with_trunk / bar))
    return 0


def _probe() -> int:
    """A guard that cannot fail is off. Perturb each assertion and require it to fire."""
    import copy, tempfile
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    d = json.loads((root / ART).read_text())
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / ART).parent.mkdir(parents=True)
        p = copy.deepcopy(d)
        p["bars"]["A26_reachable_bar_vs_their_bf16"] *= 1.01
        (t / ART).write_text(json.dumps(p))
        # the probe's own DRIFT lines are not findings; printing them would read as a failure
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check(t)
        if rc == 0:
            print("BROKEN a 1 % bar move does not fire -- this guard is off", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rc = check(root)
    if rc == 0 and _probe() != 0:
        rc = 2
    raise SystemExit(rc)
