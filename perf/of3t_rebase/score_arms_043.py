#!/usr/bin/env python3
"""Score the corrected trunk arms with the campaign's own A14 rule and bars, beside the ones
they replace.

The scoring is `of3t-orchestrator`'s `revision/d8_vs_endnode.py` unchanged -- same bars
(5.0e-02 per tensor, 2.0e-02 median), same A14 floor (reference norm 1e-12), same definition of
over-bar norm share (the over-bar tensors' squared reference norm over the compared set's) --
pointed at both sets of arms so the two columns are one table rather than two write-ups.

A14 is doing real work here and not a formality: the worst tensor in every arm on both sides is
`attn_pair_bias.layer_norm_z.bias`, whose reference norm is around 1e-19, and it reads 1e+13
without the rule. That tensor is noise divided by nothing, not a finding.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BAR, MBAR, ZERO_REF = 5.0e-2, 2.0e-2, 1e-12

OLD = Path("perf/of3t_gradients")
NEW = Path("perf/of3t_rebase")
ARMS = [
    ("block 0  shipped", OLD / "instrument_a_bundle_block0_r0_crop64.json",
     NEW / "instrument_a_bundle_043spb_block0_crop64_tbshipped.json"),
    ("block 0  tb-off ", OLD / "instrument_a_bundle_block0_r0_crop64_tboff.json",
     NEW / "instrument_a_bundle_043spb_block0_crop64_tboff.json"),
    ("block 23 shipped", OLD / "instrument_a_bundle_block23_r0_crop64_tbshipped.json",
     NEW / "instrument_a_bundle_043spb_block23_crop64_tbshipped.json"),
    ("block 23 tb-off ", OLD / "instrument_a_bundle_block23_r0_crop64_tboff.json",
     NEW / "instrument_a_bundle_043spb_block23_crop64_tboff.json"),
    ("block 47 shipped", OLD / "instrument_a_bundle_block47_r0_crop64_tbshipped.json",
     NEW / "instrument_a_bundle_043spb_block47_crop64_tbshipped.json"),
    ("block 47 tb-off ", OLD / "instrument_a_bundle_block47_r0_crop64_tboff.json",
     NEW / "instrument_a_bundle_043spb_block47_crop64_tboff.json"),
]
STACKS = [
    ("stack 0..47 shipped", OLD / "instrument_a_bundle_stack0_47_n384.json",
     NEW / "instrument_a_bundle_043_stack0_47_n384_tbshipped.json"),
    ("stack 0..47 tb-off ", OLD / "instrument_a_bundle_stack0_47_n384_tboff.json",
     NEW / "instrument_a_bundle_043_stack0_47_n384_tboff.json"),
]


def score(path: Path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    kept = [r for r in d["per_parameter"] if r["ref_norm"] >= ZERO_REF]
    if not kept:
        return None
    v = [r["rel_l2"] for r in kept]
    med = statistics.median(v)
    over = [r for r in kept if r["rel_l2"] > BAR]
    sq = sum(r["ref_norm"] ** 2 for r in kept) or 1.0
    w = max(kept, key=lambda r: r["rel_l2"])
    return {
        "file": path.name, "n": len(kept), "n_dropped_by_a14": len(d["per_parameter"]) - len(kept),
        "median": med, "median_inside_bar": med <= MBAR,
        "over_bar": len(over), "over_bar_norm_share": sum(r["ref_norm"] ** 2 for r in over) / sq,
        "worst": w["rel_l2"], "worst_tensor": w["their_tensor"],
        "forward_z_masked": d.get("forward_rel", {}).get("z_masked"),
        "upstream_revision": d.get("bundle", {}).get("upstream_revision", "0.5.0"),
        "scale_pair_bias": d.get("shipped_config", {}).get("scale_pair_bias"),
        "transpose_bias": d.get("shipped_config", {}).get("transpose_bias"),
    }


def main() -> int:
    rep = {"bars": {"per_tensor": BAR, "median": MBAR}, "a14_zero_ref": ZERO_REF,
           "scoring": "of3t-orchestrator revision/d8_vs_endnode.py, unchanged", "arms": {},
           "stacks": {}}
    hdr = (f"{'arm':<17} {'n':>3}  {'median 0.5.0':>12} {'median 0.4.3':>12}  "
           f"{'over-bar':>9} {'norm share':>10}  move")
    print(hdr); print("-" * len(hdr))
    for tag, old, new in ARMS + STACKS:
        o, n = score(old), score(new)
        into = rep["stacks" if "stack" in tag else "arms"]
        into[tag.strip()] = {"against_0.5.0": o, "against_0.4.3": n}
        if o is None or n is None:
            om = f"{o['median']:.4e}" if o else "ABSENT"
            nm = f"{n['median']:.4e}" if n else "ABSENT"
            print(f"{tag:<17} {'--':>3}  {om:>12} {nm:>12}")
            continue
        move = ("better" if n["median"] < o["median"] else "WORSE")
        ratio = o["median"] / n["median"] if n["median"] else float("inf")
        into[tag.strip()]["direction"] = move
        into[tag.strip()]["median_ratio_old_over_new"] = ratio
        print(f"{tag:<17} {n['n']:>3}  {o['median']:>12.4e} {n['median']:>12.4e}  "
              f"{n['over_bar']:>4}/{n['n']:<4} {100*n['over_bar_norm_share']:>9.1f}%  "
              f"{move} {ratio:.2f}x  "
              f"[{'PASS' if n['median_inside_bar'] else 'FAIL'} vs {MBAR:.0e}]")
    Path("perf/of3t_rebase/score_arms_043.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("\n-> perf/of3t_rebase/score_arms_043.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
