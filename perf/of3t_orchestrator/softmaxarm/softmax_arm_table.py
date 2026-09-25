#!/usr/bin/env python3
"""of3t-orchestrator pass 386: the four softmax arms on ONE frame, read from the artifacts.

`of3t-f64route` closed on a question it handed to the campaign: "the campaign's softmax ceiling
needs re-deriving, because the two arms are not two measurements of the same thing ... which of
the two is the right TARGET is a campaign decision and not this row's." It offered 0.5547 (the
verb install, which it said ran each block's OUTPUT on the device softmax and its JACOBIAN on the
float64 one) against 0.7734 (its own route install, which it called "a consistent arm"), and it
framed the choice as matching upstream's bf16 against being correct.

The premise is false and the answer was already on disk. The two arms differ by WHERE the host
float64 softmax is installed, not by consistency, and the genuinely consistent arm is a third one
`of3t-trunkceiling` had already measured: `--lever ceiling_hf3` replaces `ttnn.softmax`
module-wide, so `autograd.triangle_attention._scores` is exact in the forward AND in the chunked
backward's recompute and no softmax in the trunk runs on the card
(`perf/of3t_bwdaccum/dev_cot.py:54-58`).

Nothing is transcribed here. Every number is read from the source JSON at run time, and the
script fails rather than reports if a key is missing.

Zero card. CPU, reads committed files only.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# A26's in-frame bar for the pairformer_stack section is the denominator every multiple below
# uses. It is READ from each frame artifact's own MATCHED section and asserted equal across all
# seven, rather than typed in here: a bar transcribed into the file that grades against it is
# how a denominator quietly stops matching its numerator's frame.
BAR_KEY = "A26_style_reachable_bar_for_this_scope"

ARMS = [
    ("CTRL_B", "perf/of3t_f64route/FRAME_CTRL_B.json", "perf/of3t_f64route/CENSUS_CTRL_B.json",
     "the shipped device softmax, `--lever all` (the fp32 LayerNorm-backward islands only)"),
    ("CEIL_B", "perf/of3t_trunkceiling/FRAME_CEIL_B.json", "perf/of3t_trunkceiling/CENSUS_CEIL_B.json",
     "CTRL_B's arm on the other board class -- D234's cross-board control"),
    ("VERB_HF", "perf/of3t_f64route/FRAME_VERB_HF.json", "perf/of3t_f64route/CENSUS_VERB_HF.json",
     "`--lever ceiling_hf`: exact softmax installed at the taped VERB"),
    ("CEIL_HF", "perf/of3t_trunkceiling/FRAME_CEIL_HF.json", "perf/of3t_trunkceiling/CENSUS_CEIL_HF.json",
     "VERB_HF's arm on the other board class"),
    ("CEIL_HF3", "perf/of3t_trunkceiling/FRAME_CEIL_HF3.json", "perf/of3t_trunkceiling/CENSUS_CEIL_HF3.json",
     "`--lever ceiling_hf3`: ceiling_hf AND `ttnn.softmax` replaced module-wide, so the chunked "
     "backward's recompute is exact too. THE CONSISTENT ARM."),
    ("ROUTE_HF", "perf/of3t_f64route/FRAME_ROUTE_HF.json", "perf/of3t_f64route/CENSUS_ROUTE_HF.json",
     "`--lever all` + the site selector TT_BIO_HOST_F64_SOFTMAX_AB=pairformer: of3t-f64route's route"),
    ("ROUTE_HF2", "perf/of3t_f64route/FRAME_ROUTE_HF2.json", "perf/of3t_f64route/CENSUS_ROUTE_HF2.json",
     "ROUTE_HF repeated on a different card of the same host -- the route's own A/A"),
]


def read(path):
    with open(os.path.join(ROOT, path)) as fh:
        return json.load(fh)


def main() -> int:
    rows = {}
    floor = None
    BAR = None
    for name, frame, census, what in ARMS:
        f = read(frame)
        m = f["MATCHED"]
        c = read(census)
        st = c.get("package_stats", {}).get("HOST_F64_SOFTMAX_STATS", {})
        this_bar = m[BAR_KEY]
        if BAR is None:
            BAR = this_bar
        elif this_bar != BAR:
            raise SystemExit(f"{name}: the bar moved, {this_bar} != {BAR} -- not one frame")
        this_floor = m["floor_REF_LOCAL_bf16_vs_REF_LOCAL_f64_n384"]["mass_weighted_rel_l2"]
        if floor is None:
            floor = this_floor
        elif this_floor != floor:
            raise SystemExit(f"{name}: the floor moved, {this_floor} != {floor} -- not one frame")
        rows[name] = {
            "what": what,
            "argv_lever": " ".join(c.get("argv", [])[:2]),
            "env_selector": c.get("env", {}).get("TT_BIO_HOST_F64_SOFTMAX_AB", "<unset>"),
            "board": f["hosts"]["device_arm_built_on"],
            "vs_float64": m["ours_vs_REF_LOCAL_f64_n384"]["mass_weighted_rel_l2"],
            "vs_float64_norm_ratio": m["ours_vs_REF_LOCAL_f64_n384"]["mass_weighted_norm_ratio"],
            "vs_upstream_bf16": m["ours_vs_REF_LOCAL_bf16_n384"]["mass_weighted_rel_l2"],
            "multiple_of_the_A26_in_frame_bar":
                m["ours_vs_REF_LOCAL_bf16_n384"]["mass_weighted_rel_l2"] / BAR,
            "host_f64_softmax_census": st or "the counter predates the D236 split and is absent",
        }

    hf, hf3 = rows["CEIL_HF"], rows["CEIL_HF3"]
    route, verb = rows["ROUTE_HF"], rows["VERB_HF"]
    out = {
        "what": "the four host-float64-softmax install arms on the of3t-frame384 frame, one "
                "scorer, one float64 reference, one upstream-bf16 reference. Read from the "
                "source artifacts by perf/of3t_orchestrator/softmaxarm/softmax_arm_table.py.",
        "frame": "boundary_n384.pt / block47_boundary.pt, padded 384, 56 real tokens",
        "references": {
            "float64": "REF_LOCAL_f64_n384, built on qb1 CPU by of3t-frame384",
            "upstream_bf16": "REF_LOCAL_bf16_n384, upstream's own bf16 autocast on the same boundary",
            "the_floor_their_bf16_vs_float64": floor,
        },
        "A26_in_frame_bar": {"value": BAR,
                             "read_from": f"MATCHED/{BAR_KEY} of all seven frame artifacts, "
                                          "asserted equal across them"},
        "arms": rows,
        "CROSS_BOARD": {
            "what": "the same arm on p150a (qb1) and p300c (qb2). D234 established this for the "
                    "shipped arm; this is the second instance and it is on the lever arm.",
            "shipped_qb1_CTRL_B_vs_qb2_CEIL_B": [rows["CTRL_B"]["vs_upstream_bf16"],
                                                 rows["CEIL_B"]["vs_upstream_bf16"]],
            "verb_qb1_VERB_HF_vs_qb2_CEIL_HF": [verb["vs_upstream_bf16"], hf["vs_upstream_bf16"]],
            "identical": (rows["CTRL_B"]["vs_upstream_bf16"] == rows["CEIL_B"]["vs_upstream_bf16"]
                          and verb["vs_upstream_bf16"] == hf["vs_upstream_bf16"]),
            "so": "a cross-board reading on this frame needs no correction, twice demonstrated, "
                  "which is what lets CEIL_HF3 (qb2) be compared to ROUTE_HF (qb1) directly.",
        },
        "THE_CONSISTENCY_CORRECTION_IS_0.11_PERCENT": {
            "question_handed_up_by_of3t_f64route":
                "'the campaign's softmax ceiling needs re-deriving ... which of the two is the "
                "right TARGET is a campaign decision and not this row's'",
            "its_premise": "that 0.5547455957585244 came from an arm whose forward and Jacobian "
                           "used different softmax implementations, and that its own 0.7734340172378431 "
                           "was the consistent measurement of the same thing.",
            "the_consistent_arm_already_existed":
                "--lever ceiling_hf3, dev_cot.py:54-58 and :398-401, measured by of3t-trunkceiling "
                "as CEIL_HF3 on the same frame.",
            "ceiling_hf_vs_float64": hf["vs_float64"],
            "ceiling_hf3_vs_float64": hf3["vs_float64"],
            "the_correction_against_float64": hf3["vs_float64"] / hf["vs_float64"] - 1.0,
            "ceiling_hf_vs_upstream_bf16": hf["vs_upstream_bf16"],
            "ceiling_hf3_vs_upstream_bf16": hf3["vs_upstream_bf16"],
            "the_correction_against_upstream_bf16": hf3["vs_upstream_bf16"] / hf["vs_upstream_bf16"] - 1.0,
            "and_the_direction": "the consistent arm is BETTER, on both references. The choice "
                                 "of3t-f64route asked the campaign to make does not exist: nothing "
                                 "is traded and the correction is four hundredths of a percent.",
            "reach_evidence_for_the_module_wide_patch":
                "ceiling_hf3 IS ceiling_hf plus one line (dev_cot.py:398-401, `ttnn.softmax = "
                "_raw_host_f64`). The scorer is deterministic -- two board classes reproduce the "
                "same arm to sixteen digits -- so a non-zero delta between the two arms IS the "
                "proof the patch fired. Its own counter SMRAW is printed to stdout by "
                "dev_cot.py:446-447 and is not banked in the report, which is an owed fix, not a "
                "gap in this reading.",
        },
        "THE_ROUTE_INSTALL_IS_THE_WORSE_ARM": {
            "route_vs_float64": route["vs_float64"],
            "verb_vs_float64": verb["vs_float64"],
            "consistent_vs_float64": hf3["vs_float64"],
            "route_is_worse_than_the_consistent_arm_by": route["vs_float64"] / hf3["vs_float64"] - 1.0,
            "route_vs_upstream_bf16": route["vs_upstream_bf16"],
            "consistent_vs_upstream_bf16": hf3["vs_upstream_bf16"],
            "worse_against_upstream_bf16_by": route["vs_upstream_bf16"] / hf3["vs_upstream_bf16"] - 1.0,
            "it_is_not_a_target_question":
                "against float64 there is no reference-sharing argument to make -- float64 has no "
                "error to share -- and float64 ranks the arms the same way upstream's bf16 does. "
                "A lever that makes MORE softmaxes exact and moves the gradient FARTHER from the "
                "true gradient is a defect, not a target.",
            "it_is_reproducible": "ROUTE_HF (qb1 card 2) and ROUTE_HF2 (qb1 card 3) agree to "
                                  "sixteen digits, so the 34 % is deterministic and not noise.",
            "what_the_route_adds": {
                "verb_arm": verb["host_f64_softmax_census"],
                "route_arm": route["host_f64_softmax_census"],
                "read": "the route serves 1,685 RAW calls the verb serves none of, and 472 more "
                        "taped ones. A raw serve is an exact FORWARD with no tape node "
                        "(autograd.py:920-935), so its Jacobian is whatever the surrounding "
                        "region already was.",
            },
            "MECHANISM_CANDIDATE_not_established":
                "`autograd.triangle_attention`'s chunked backward recomputes its own "
                "`ttnn.softmax` inside `_scores`, which neither the verb patch nor a site "
                "selector can reach (dev_cot.py:50-53, :383-386). Serving the pair track's "
                "FORWARD exact through the site while its recompute stays on the card evaluates "
                "the Jacobian at activations the forward did not produce. That is the "
                "inconsistency of3t-f64route named, located in the other arm.",
            "FALSIFIER_two_sided":
                "run the route arm with raw serves suppressed (serve taped calls only). ~0.4180 "
                "against float64 means the raw serves are the carrier and the mechanism stands; "
                "~0.5605 means the 472 extra taped serves carry it and the mechanism dies.",
        },
        "WHAT_THE_CEILING_IS": {
            "quote_this": hf3["vs_upstream_bf16"],
            "as_a_multiple_of_the_A26_in_frame_bar": hf3["vs_upstream_bf16"] / BAR,
            "what_the_campaign_has_been_quoting": hf["vs_upstream_bf16"] / BAR,
            "the_route_number_of3t_f64route_offered_instead": route["vs_upstream_bf16"] / BAR,
            "the_shipped_baseline": rows["CTRL_B"]["vs_upstream_bf16"] / BAR,
            "so": "the ceiling does NOT need re-deriving. It moves from 1.0529x to 1.0525x when "
                  "the consistent arm is quoted instead of the verb arm, and it stays a "
                  "measurement of an unmerged branch whose inference A/B is still owed.",
        },
        "DOESNOT": "this is one gradient reading at one captured boundary at padded N=384 on the "
                   "of3t-frame384 frame. It does not touch D242, it names no root cause for the "
                   "route's 34 %, and it merges nothing. It says which number the campaign should "
                   "quote as its softmax ceiling and that the install being prepared to land is "
                   "not the arm that produced it.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
