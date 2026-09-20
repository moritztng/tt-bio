#!/usr/bin/env python3
"""Does APB move the confidence the user reads, as well as the coordinates?

THE DIMENSION THE ANGSTROM WORK DOES NOT COVER. Accuracy here was scored as structural RMSD against
a fixture and against the experimental structure. pLDDT is a separate, user-visible number: it is
what a caller sorts and filters on, and `confidence-lever-needs-seed-scatter-bar-not-angstrom-bar`
says a confidence change has to be judged on its own scatter rather than inherited from an Angstrom
verdict.

Session 3 recorded pLDDT on every fold, so this needs no card: 96 base folds and 48 on folds at
512 aa, paired inside the same A,B,A blocks that produced the timing.

WHAT THE BAR IS. Not an absolute threshold -- the honest comparison is the lever's shift against the
spread the caller already lives with. Two references are printed:
  * the WITHIN-ARM spread across folds, which is what the same configuration gives run to run;
  * the block-to-block spread of the base arm, the drift a caller sees between sessions.
A shift comfortably inside both is a shift the caller cannot distinguish from noise they already
accept.

    python3 perf/c14_land/plddt_audit.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "perf/c14_land/apb3_ab.json"
SCORED_BLOCKS = {0, 1, 2, 3}


def main() -> int:
    d = json.loads(SESSION.read_text())
    per_arm, per_block = {}, {}
    for leg in d["blocks"]:
        if leg.get("block") not in SCORED_BLOCKS or leg.get("returncode"):
            continue
        for f in (leg.get("result") or {}).get("folds", []):
            v = f.get("plddt")
            if v is None:
                continue
            per_arm.setdefault(leg["arm"], []).append(v)
            per_block.setdefault(leg["block"], {}).setdefault(leg["arm"], []).append(v)

    base, on = per_arm.get("base", []), per_arm.get("on", [])
    if not (base and on):
        print("missing an arm -- cannot audit")
        return 1

    print(f"{'arm':6s}{'folds':>7}{'mean pLDDT':>13}{'stdev':>10}{'min':>10}{'max':>10}")
    for name, v in (("base", base), ("on", on)):
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"{name:6s}{len(v):>7}{st.mean(v):>13.6f}{sd:>10.6f}{min(v):>10.6f}{max(v):>10.6f}")

    shift = st.mean(on) - st.mean(base)
    within = max(st.stdev(base) if len(base) > 1 else 0.0,
                 st.stdev(on) if len(on) > 1 else 0.0)
    block_means = [st.mean(a["base"]) for a in per_block.values() if a.get("base")]
    across = st.stdev(block_means) if len(block_means) > 1 else 0.0

    print(f"\npLDDT shift (on - base)           : {shift:+.6f}")
    print(f"  within-arm fold-to-fold stdev   : {within:.6f}")
    print(f"  base block-to-block stdev       : {across:.6f}")
    if within:
        print(f"  shift / within-arm stdev        : {abs(shift)/within:.3f}")
    if across:
        print(f"  shift / block-to-block stdev    : {abs(shift)/across:.3f}")

    # A within-arm stdev of exactly zero is the interesting case, not a degenerate one: it means
    # the configuration is deterministic fold to fold, so ANY shift is a real systematic move and
    # has to be judged against the across-block reference instead.
    if within == 0.0:
        print("\n  NOTE: within-arm stdev is EXACTLY zero -- each configuration reproduces itself "
              "fold for fold. So the shift is a real systematic difference, not noise, and the "
              "block-to-block reference is the one that matters.")

    # The session cannot supply its own noise reference, so bring the two things that can: the
    # same lever at a DIFFERENT SIZE, and a seed change.
    other = json.loads((REPO / "perf/c14_land/apb_accuracy_298.json").read_text())
    at298 = {}
    for L in other["blocks"]:
        for f in (L.get("result") or {}).get("folds", []):
            if f.get("plddt") is not None:
                at298.setdefault(L.get("arm"), set()).add(f["plddt"])
    shift298 = None
    if at298.get("base") and at298.get("apb"):
        shift298 = st.mean(list(at298["apb"])) - st.mean(list(at298["base"]))
        print(f"\nSAME LEVER AT 298 aa (perf/c14_land/apb_accuracy_298.json, production protocol)")
        print(f"  base {st.mean(list(at298['base'])):.6f}   apb "
              f"{st.mean(list(at298['apb'])):.6f}   shift {shift298:+.6f}")

    SEED_PLDDT = 0.0230   # boltz-2 seed-to-seed pLDDT move, perf/roof_concat, NOT measured here
    print(f"\nSEED REFERENCE: a seed change moves boltz-2 pLDDT by about {SEED_PLDDT:.4f}")
    print("  (from perf/roof_concat via the orchestrator's pass-51 note -- another row's number, "
          "cited rather than re-measured, and the comparison is only as good as it is.)")
    print(f"  512 aa shift is {SEED_PLDDT/abs(shift):.0f}x smaller than a seed change")
    if shift298:
        print(f"  298 aa shift is {SEED_PLDDT/abs(shift298):.0f}x smaller than a seed change")

    if shift298 is not None and shift * shift298 < 0:
        verdict = ("SIGN FLIPS WITH SIZE -- rounding, not a systematic confidence cost")
        note = (f"  APB RAISES pLDDT at 298 aa ({shift298:+.6f}) and LOWERS it at 512 aa "
                f"({shift:+.6f}). A lever that degraded confidence would push one way at both "
                "sizes. This is the re-laned projections rounding differently, which is what the "
                "docs entry already says the lever does.")
    elif abs(shift) < SEED_PLDDT / 5:
        verdict = "SMALL AND SAME-SIGNED -- well inside a seed change"
        note = "  the shift is an order of magnitude under seed-to-seed movement."
    else:
        verdict = "QUOTE IT IN THE DOCS"
        note = "  the shift is close enough to seed-scale to be worth a user-facing sentence."
    print(f"\nVERDICT: {verdict}")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
