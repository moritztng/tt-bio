#!/usr/bin/env python3
"""Re-state D56's ~2,172x floor against the precision upstream 0.4.3 ACTUALLY uses.

CPU only. No card. No new run. Reads `perf/of3t_fp32islands/version_split.json`.

WHY
---
D118 put seven defects in one fp32-ceiling bucket and warned, in its own text, that *"a satisfying
mechanism is also the moment over-attribution starts"*. D120 then corrected two of them -- D9 and
half of D8 -- and left **D56, the one D118 called the strongest**, untouched. This applies the same
correction to it, and the correction is a denominator, not a measurement.

D56's surviving claim, after D62 withdrew its conditioning mechanism, is:

    "the 25.5795 % sits on a constant ~2,172x device arithmetic floor against torch fp32,
     which is a port gap and not a property of the arithmetic"

That ratio's denominator is **torch fp32**. OpenFold3 **0.4.3** does not compute this island in
fp32: `normalization.py:64-75` runs LayerNorm in bf16 with autocast explicitly disabled (D120).
0.5.0 restores the fp32 upcast. So the question D56 asks -- is this a port gap? -- is answered by
which boundary you score against, and `of3t-fp32islands` measured both.

D56's construct is `g_gamma = sum_i (dL/d ln_out)_i . s_hat_i`, the LayerNorm affine WEIGHT
gradient. That is row `layer_norm BACKWARD d(gamma)` in `version_split.json`. **Different harness**
from D56's synthetic K-ladder -- this is the real island at crop-384 shapes, not a ladder -- which
is why the claim below is about the DENOMINATOR CHOICE and is not a re-measurement of D56's ratio.
"""
import json
import sys
from pathlib import Path

SRC = Path("/tmp/of3t/of3t-orchestrator/compose/perf/of3t_fp32islands/version_split.json")
OUT = Path(__file__).with_name("D56_RESTATED.json")
ROW = "layer_norm BACKWARD d(gamma)"


def main() -> int:
    if not SRC.is_file():
        print(f"REFUSING: {SRC} not present -- run compose_verify.sh first.")
        return 2
    rows = json.loads(SRC.read_text())
    hit = [r for r in rows if r["row"].startswith(ROW)]
    if len(hit) != 1:
        print(f"REFUSING: expected exactly one '{ROW}' row, found {len(hit)} -- the artifact's "
              f"shape has changed and this script must be re-read, not re-run")
        return 1
    a = hit[0]["arms"]

    def pick(*needles):
        for k in a:
            if all(n in k for n in needles):
                return k, a[k]
        print(f"REFUSING: no arm matching {needles} in {list(a)}")
        sys.exit(1)

    k_043, v_043 = pick("0.4.3")
    k_050, v_050 = pick("0.5.0")
    k_ours, v_ours = pick("ours", "bf16")
    k_oursf, v_oursf = pick("ours", "fp32")

    res = {
        "what": "D56's ~2,172x floor re-scored against the boundary OpenFold3 0.4.3 actually uses",
        "source": str(SRC),
        "row": hit[0]["row"],
        "construct": "g_gamma = sum_i (dL/d ln_out)_i . s_hat_i -- the LayerNorm affine WEIGHT "
                     "gradient, which is D56's tensor class",
        "harness_note": "the real island at crop-384 shapes, NOT D56's synthetic K-ladder. This "
                        "re-states which DENOMINATOR is the right one; it does not re-measure "
                        "D56's 2,172x.",
        "arms": {k_043: v_043, k_050: v_050, k_ours: v_ours, k_oursf: v_oursf},
        "against_0_4_3_the_served_boundary": {
            "upstream": v_043, "ours_shipped": v_ours,
            "x_upstream": v_ours / v_043,
            "we_are_more_accurate": v_ours < v_043,
            "how_much_more_accurate": v_043 / v_ours,
        },
        "against_0_5_0": {
            "upstream": v_050, "ours_shipped": v_ours, "ours_fp32": v_oursf,
            "x_upstream_shipped": v_ours / v_050,
            "x_upstream_our_fp32": v_oursf / v_050,
        },
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"row: {hit[0]['row']}\n")
    for k, v in res["arms"].items():
        print(f"  {k:<44} {v:.6e}")
    d = res["against_0_4_3_the_served_boundary"]
    e = res["against_0_5_0"]
    print(f"\nagainst 0.4.3, the boundary the served checkpoint is bound to:")
    print(f"  ours / upstream = {d['x_upstream']:.4f}   -> we are {d['how_much_more_accurate']:.4f}x "
          f"MORE accurate" if d["we_are_more_accurate"] else
          f"  ours / upstream = {d['x_upstream']:.4f}")
    print(f"\nagainst 0.5.0, which our port does not target:")
    print(f"  ours shipped / upstream = {e['x_upstream_shipped']:.1f}x")
    print(f"  ours fp32    / upstream = {e['x_upstream_our_fp32']:.1f}x   <- the silicon ceiling, "
          f"real and only relevant here")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
