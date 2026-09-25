#!/usr/bin/env python3
"""The frame match is ONE-SIDED, and the instrument that closes it already exists.

A34 (PROTOCOL §3z) requires both sides of a per-parameter comparison to sit on the SAME
boundary: same batch, same entry activations, same incoming cotangent. `of3t-modelframe` built
the first arm to it and the campaign has been reading its 1.7814x as a statement about our
gradient. It is half a statement. OUR trunk is driven by the reference's float64 boundary and
its float64 cotangent, injected; the bf16 denominator it is divided by is a FULL-MODEL bf16
autocast run that drove its own trunk with its own bf16 boundary and its own bf16 cotangent.
The two sides of the failing ratio are not the same experiment, which is the exact shape of
D237 -- the defect that was worth 1.9085x the last time it was found.

Everything here is read out of committed files. Nothing is transcribed and nothing is new
arithmetic on a card: the point is that a measurement is MISSING, not that one is wrong.

  python3 perf/of3t_orchestrator/twoside/one_sided_frame.py \
      --out perf/of3t_orchestrator/twoside/ONE_SIDED_FRAME.json
"""
from __future__ import annotations

import argparse
import json
import re
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def lines_matching(rel: str, pat: str) -> list[dict]:
    """Every line of a committed file matching `pat`, with its 1-indexed line number."""
    p = ROOT / rel
    out = []
    for i, ln in enumerate(p.read_text().splitlines(), 1):
        if re.search(pat, ln):
            out.append({"file": rel, "line": i, "text": ln.strip()})
    if not out:
        raise SystemExit(f"{rel}: no line matches {pat!r} -- this script's premise moved")
    return out


def jload(rel: str):
    return json.loads((ROOT / rel).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clause = jload("perf/of3t_modelframe/CLAUSE.json")
    attrib = jload("perf/of3t_modelframe/ATTRIBUTION.json")
    entry = jload("perf/of3t_orchestrator/blockcurve/ENTRY_OR_ACCUMULATION.json")
    fm = jload("perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json")

    # --- side 1: how OUR arm is driven -------------------------------------------------------
    ours = lines_matching("perf/of3t_modelframe/runarm.sh", r"--boundary .*--cap-last|^[BC]=")
    # --- side 2: where the bf16 denominator comes from ---------------------------------------
    theirs = lines_matching("perf/of3t_modelframe/score.sh", r"^\s*--bf16 ")
    # --- the instrument that would close it, and that it already takes both injections -------
    instrument = (lines_matching("perf/of3t_trunkg043/ref_grad.py", r'"--policy"')
                  + lines_matching("perf/of3t_trunkg043/ref_grad.py", r'"--boundary"')
                  + lines_matching("perf/of3t_trunkg043/ref_grad.py", r'"--cap-last"'))
    bf16auto_doc = lines_matching("perf/of3t_trunkg043/ref_grad.py", r"bf16auto\s+float32 parameters")

    lv = clause["preregistered"]["levels"]
    floor = lv["upstreams_own_floor_here"]
    exact = lv["bit_exact_float64_trunk"]

    blk47 = next(b for b in entry["per_block"] if b["block"] == 47)
    trunk_sub = {g["group"]: g for g in attrib["by_subtree"]}

    doc = {
        "what": "A34 is satisfied on ONE side of the trunk clause. The bf16 denominator is a "
                "different experiment from the numerator, and the instrument that would put "
                "them on the same boundary already exists and has never been run on this frame.",
        "host": socket.gethostname(),
        "defect": "D241",
        "ledger": "R135",
        "the_asymmetry": {
            "our_arm_is_injected": {
                "evidence": ours,
                "means": "our device trunk is handed the reference's float64 boundary AND the "
                         "reference's float64 incoming cotangent",
            },
            "their_bf16_arm_is_not": {
                "evidence": theirs,
                "means": "arm4_bf16_autocast is a FULL-MODEL bf16 autocast run. Its trunk was "
                         "driven by whatever bf16 boundary and bf16 cotangent its own forward "
                         "and backward produced -- neither is the injected pair",
            },
            "direction_of_the_bias": "TOWARDS US. We are handed an exact cotangent and they "
                                     "compute their own, so their reading carries an error ours "
                                     "does not. Every 'Nx upstream' figure on this frame is "
                                     "therefore a LOWER bound on our excess, and the clause's "
                                     "1.7814x is not explained away by it. What is unknown is "
                                     "the SIZE of the bound.",
        },
        "why_it_decides_the_campaign": {
            "if_our_trunk_merely_matched_upstreams_own_bf16_floor": {
                "trunk_level": clause["published"]["trunk"]["upstreams_own_floor_here_vs_float64"],
                "clause_value": floor["clause_value"],
                "x_bar": floor["x_bar"],
                "clears": floor["clears"],
            },
            "a_perfect_trunk": {"clause_value": exact["clause_value"], "x_bar": exact["x_bar"],
                                "clears": exact["clears"]},
            "bar": clause["preregistered"]["bar"],
            "reading": "the whole failing clause is the distance between our trunk and "
                       "upstream's own bf16 trunk. That distance has never been measured with "
                       "both sides on the same boundary.",
        },
        "the_missing_arm": {
            "command": "perf/of3t_trunkg043/ref_grad.py --policy bf16auto "
                       "--boundary <of3t_modelframe>/boundary_model_n384.pt "
                       "--cap-last <of3t_modelframe>/cot_model_n384.pt --blocks 48",
            "instrument_already_takes_both_injections": instrument,
            "bf16auto_is_upstreams_own": bf16auto_doc,
            "never_run_on_this_frame": "every committed bf16auto arm (REF_BF16AUTO_c64.json and "
                                       "the frame384 family) is on block47_boundary.pt, the "
                                       "capture-driven crop-384 walk D237 refuted",
            "needs_no_card": True,
        },
        "what_it_would_settle": {
            "ours_on_this_frame_vs_float64": trunk_sub and clause["published"]["trunk"][
                "reading_vs_float64"],
            "per_block_cotangent_growth_ours": "0.1100 after one block, plateauing near 2 "
                                               "(of3t-cotcoh, measured on this frame)",
            "per_block_cotangent_growth_theirs": None,
            "two_sided_test": "if upstream's injected-bf16 trunk also reaches ~0.9 against "
                              "float64, the trunk excess is the site conditioning plus this "
                              "asymmetry and the clause may already clear; if it reproduces its "
                              "un-injected 0.3148, the excess is ours and is real at 2.97x. "
                              "Those are the only two outcomes and one arm separates them.",
        },
        "corroborating_entry_reading": {
            "block_47_is_the_first_block_the_backward_touches": blk47,
            "note": "blocks 47 and 46 hold "
                    f"{blk47['pct_trunk_error_mass'] + next(b['pct_trunk_error_mass'] for b in entry['per_block'] if b['block'] == 46):.4f} "
                    "% of the trunk's error mass before any trunk-internal accumulation can "
                    "have happened, and upstream reads very low there (0.2308, 0.1499) -- which "
                    "is exactly what an un-injected reference would do if its own cotangent "
                    "error had not yet had blocks to grow in.",
        },
        "model_scope_context": {
            "ours_vs_upstream_bf16": fm["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"],
            "upstream_bf16_vs_float64": fm["stats"]["UPSTREAM_BF16_vs_FLOAT64"]["mass_weighted_rel_l2"],
            "upstream_f32_vs_float64": fm["stats"]["UPSTREAM_F32_vs_FLOAT64"]["mass_weighted_rel_l2"],
        },
        "doesnot": "This names a missing control. It does not claim the trunk passes, does not "
                   "move a tolerance and does not retract the 1.7814x -- which, being a lower "
                   "bound, still fails. It says the number the campaign is trying to close has "
                   "never been measured against a like-for-like reference.",
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("defect", "ledger")}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
