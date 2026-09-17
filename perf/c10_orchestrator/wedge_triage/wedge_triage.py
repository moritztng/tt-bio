#!/usr/bin/env python3
"""Triage the 768 aa Blackhole wedge against the known sub-tile-last-axis hang class.

`c10-size-scaling`'s sixth 768 aa Boltz-2 fold hung a p300c: 101 % CPU in state R, board power at
22 W against the 46 W a fold draws, SIGTERM ignored, ARC alive. Not an error, not an OOM.

The knowledge base carries a standing instruction for exactly this: "Action for the next worker who
hits a Blackhole hang (not error, not OOM) on a slice/chunk/reshape op: check this shape against the
sub-tile-last-axis pattern first before treating it as a new mechanism."
(`blackhole-ttnn-reshape-hang-second-sighting`, which itself points at
`blackhole-sub-tile-last-axis-slice-of-large-dram-tensor-wedges`.)

This is that check, done on CPU against the committed 512 aa op census. It produces a short
candidate list for the next worker with a chip. **It does not root-cause anything** — no fold was
run and no hang was reproduced here.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parents[1] / "roof_launch" / "op_census_512.json"

TILE = 32
FAMILY = ("chunk", "slice", "reshape", "permute", "pad", "concat", "to_layout", "__getitem__")
# The shape named in the second sighting, which hung a Blackhole card twice in roof_launch's
# shape ladder and had to be killed by PID both times.
SECOND_SIGHTING_SHAPE = "1x768x512"
# Folds the campaign has completed at 512 aa with no hang, per concluded row. Lower bounds: each
# row's own state doc reports these as accepted warm folds.
CLEAN_512_FOLDS = {"c10-bare-baseline": 16, "c10-fixed-cost": 28, "c10-trace-lever": 20,
                   "c10-size-scaling": 10}
WEDGE = {"size_aa": 768, "steps": 200, "folds_before": 5, "signature": "101 % CPU state R, "
         "22 W board power against 46 W, SIGTERM ignored, ARC alive"}


def _dims(sig):
    """Every dim of every tensor in a census signature, per tensor."""
    return [[int(x) for x in t.split("x")] for t in re.findall(r"\b\d+(?:x\d+)+\b", sig)]


def analyse():
    census = json.loads(CENSUS.read_text())
    ts = census["top_shapes"]

    family, subtile, flagged = [], [], []
    for k, v in ts.items():
        op = k.split("|")[0]
        if not any(f in op for f in FAMILY):
            continue
        tensors = _dims(k)
        row = {"shape": k, "op": op, "calls": v["calls"], "GB": v["B"] / 1e9,
               "tiles_per_call": v["in_tiles"] + v["out_tiles"]}
        family.append(row)
        # The pattern: a last-axis extent that is not a whole number of tiles.
        bad = sorted({t[-1] for t in tensors if t and t[-1] % TILE != 0})
        if bad:
            subtile.append(dict(row, sub_tile_last_axis_extents=bad))
        # Exact operand match. "1x1x768x512" CONTAINS "1x768x512" as a substring, so a naive
        # `in` test matches tensors that are not the flagged shape at all.
        if any("x".join(str(d) for d in t) == SECOND_SIGHTING_SHAPE for t in tensors):
            flagged.append(row)

    # Axes that are not a whole number of tiles anywhere in the fold, last-axis or not: a permute
    # or reshape can move any of these INTO the last position.
    odd = {}
    for k in ts:
        for t in _dims(k):
            for d in t:
                if d % TILE != 0 and d > 1:
                    odd.setdefault(d, []).append(k.split("|")[0])

    clean = sum(CLEAN_512_FOLDS.values())
    flagged_calls = sum(r["calls"] for r in flagged)
    out = {
        "scope": "CPU cross-reference of a committed op census against two knowledge-base memories. "
                 "No device, no fold, no hang reproduced, no root cause claimed.",
        "census": str(CENSUS),
        "wedge": WEDGE,
        "known_class": {
            "memories": ["blackhole-sub-tile-last-axis-slice-of-large-dram-tensor-wedges",
                         "blackhole-ttnn-reshape-hang-second-sighting"],
            "mechanism": "a sub-tile last-axis slice/chunk of a large TILE-layout DRAM tensor hangs "
                         "on Blackhole and returns fine on Wormhole; piece width decides, not row "
                         "width. Unfixed upstream.",
            "prior_sightings": 2,
        },
        "family_ops_in_the_fold": sorted(family, key=lambda r: -r["calls"]),
        "sub_tile_last_axis_candidates": sorted(subtile, key=lambda r: -r["GB"]),
        "non_tile_multiple_axes_anywhere": {str(k): sorted(set(v)) for k, v in
                                            sorted(odd.items())},
    }

    # --- the finding that cuts against a naive match ------------------------------------------
    out["second_sighting_shape_is_in_this_fold"] = {
        "shape": SECOND_SIGHTING_SHAPE,
        "calls_per_512aa_fold": flagged_calls,
        "rows": flagged,
        "clean_512aa_folds_in_this_campaign": CLEAN_512_FOLDS,
        "clean_total": clean,
        "executions_without_a_hang": flagged_calls * clean,
        "reading": "The exact reshape the second sighting names, [1,768,512], runs %d times in "
                   "every 512 aa fold, and this campaign has completed at least %d such folds with "
                   "zero hangs -- on the order of %.1f million executions. So the SHAPE ALONE is "
                   "not the trigger. That is new information for that memory's open item, which "
                   "left 'same root cause?' unconfirmed: it weakens shape-alone and strengthens "
                   "shape-plus-context, meaning layout, DRAM residency or allocator state."
                   % (flagged_calls, clean, flagged_calls * clean / 1e6),
    }
    cands = ", ".join(f"{r['op']} with last-axis {r['sub_tile_last_axis_extents']} "
                      f"({r['calls']:.0f} calls)" for r in subtile) or "none"
    out["verdict"] = (
        "NOT MATCHED to the known class on this evidence, and not excluded either. The pattern's "
        "defining feature is a sub-tile last-axis extent on a large TILE-layout DRAM tensor, and "
        "the recorded 512 aa shapes contain %d such shape: %s. That is the per-head channel width, "
        "768/16 = 48, which is not a tile multiple though it is above one tile -- the known class "
        "is about widths BELOW a tile, so this is adjacent rather than matching. The stronger "
        "point is the campaign's own counter-evidence: the shape the second sighting names runs "
        "millions of times here without hanging."
        % (len(subtile), cands)
    )
    out["for_the_next_worker_with_a_chip"] = [
        "The census is 512 aa. The wedge was at 768 aa, where every token-scaled extent changes, so "
        "a 768 aa census is the first thing to capture -- the trigger shape may simply not exist at "
        "512 aa. That is the single most useful next step and it needs a chip.",
        "Rank by the candidate list here rather than by the whole fold: 18 family shapes, of which "
        "the largest by traffic are the ones worth a per-op repro.",
        "Do not reproduce this on a card another row is using, and remember SIGTERM was ignored: "
        "budget for a SIGKILL and a reset, and read the standing rule about killing a chip holder.",
        "If it does turn out to be the same class, the two prior sightings and this one should be "
        "filed upstream together, which is what the second-sighting memory already asked for.",
    ]
    out["limits"] = [
        "Nothing here was run. This is a shape cross-reference against a census taken at a "
        "different size from the failure.",
        "The census records 60 shapes covering 87 % of calls but only 32 % of bytes, so a trigger "
        "shape outside that table would be invisible to this check.",
        "One hang in six folds is not a rate, and a shape that appears in a fold that hung is not "
        "thereby the cause.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "wedge_triage.json").write_text(json.dumps(r, indent=2) + "\n")
    s = r["second_sighting_shape_is_in_this_fold"]
    print(f"family ops in the fold: {len(r['family_ops_in_the_fold'])}")
    print(f"sub-tile last-axis candidates: {len(r['sub_tile_last_axis_candidates'])}")
    print(f"the second sighting's {s['shape']} reshape runs {s['calls_per_512aa_fold']:.0f}x per "
          f"512 aa fold, {s['clean_total']} clean folds, "
          f"~{s['executions_without_a_hang']/1e6:.1f}M executions with no hang")
    print(f"non-tile-multiple axes anywhere: {sorted(int(k) for k in r['non_tile_multiple_axes_anywhere'])}")
    print(f"VERDICT: {r['verdict'][:80]}...")
