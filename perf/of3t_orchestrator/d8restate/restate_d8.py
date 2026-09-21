#!/usr/bin/env python3
"""Re-state D8 against upstream's OWN bf16 recipe instead of against float64.

CPU only. No card. No new run. Reads `perf/of3t_apbgrad/SCOPE_c64.json`, which already carries
every number this needs and has never been read this way.

WHY
---
The campaign's own record says this is owed:

    "D8 should be re-stated against upstream's own bf16 before more engineering is spent on it,
     exactly the correction D120 applied to D9."

D8 says *"the assembled pairformer block's pair-track gradients are an order of magnitude outside
the bar"*. The bar is the 5.0e-02 per-tensor relative-L2 against a **float64** reference. D120
already showed what that framing costs on D9: our fp32 softmax is 3.2x MORE accurate than
upstream's bf16 one, and D9 dissolved into a policy mismatch once the reference was the right one.
The question here is whether D8 dissolves the same way.

A26 fixes the comparison: an independent implementation exactly as good as upstream's own bf16
recipe reads sqrt(2) x upstream's distance from float64. A27 fixes the bookkeeping: a ratio must
name how its denominator arm was built. The denominator here is
`SCOPE_c64.json::floor_their_bf16_vs_float64` and its per-block companion `floor_per_block` --
upstream 0.4.3's own bf16-autocast training step scored against `grads_f64_043.pt`, on the same
2,736 tensors, by the same scorer, in the same run.

SCOPE HONESTY
-------------
This re-states D8 on the **pairformer stack** -- 2,736 tensors, 48 blocks, BOTH tracks -- because
that is the scope on which a bf16 floor exists. D8's own words are narrower: one assembled block's
**pair track**. So this is a re-statement on a SUPERSET, reported per block so the block-level claim
is visible, and it does not separately re-score the pair track alone. Said here rather than left
for a reader to notice.
"""
import json
import statistics
import sys
from pathlib import Path

SRC = Path("/tmp/of3t/of3t-orchestrator/compose/perf/of3t_apbgrad/SCOPE_c64.json")
OUT = Path(__file__).with_name("D8_RESTATED.json")
A26 = 2 ** 0.5


def main() -> int:
    if not SRC.is_file():
        print(f"REFUSING: {SRC} not present -- run compose_verify.sh first so the composed tree "
              f"exists. This script reads published artifacts and computes nothing on a card.")
        return 2
    d = json.loads(SRC.read_text())
    bar = d["bars"]["per_tensor"]
    floor = d["floor_their_bf16_vs_float64"]
    fpb = d["floor_per_block"]
    arms = d["arms"]

    rows = []
    for name in ("CTRL", "RENORM"):
        a = arms[name]
        o64 = a["ours_vs_float64"]
        rows.append({
            "arm": name,
            "mass_weighted_rel_l2_vs_float64": o64["mass_weighted_rel_l2"],
            "x_upstream_bf16": o64["mass_weighted_rel_l2"] / floor["mass_weighted_rel_l2"],
            "x_A26_reachable": o64["mass_weighted_rel_l2"] / (A26 * floor["mass_weighted_rel_l2"]),
            "over_per_tensor_bar": o64["over_per_tensor_bar"],
            "over_per_tensor_bar_mass": o64["over_per_tensor_bar_mass"],
        })

    # the same two counts for upstream's OWN bf16 step, which is the whole point
    floor_row = {
        "arm": "upstream 0.4.3 bf16 autocast (the DENOMINATOR, A27)",
        "mass_weighted_rel_l2_vs_float64": floor["mass_weighted_rel_l2"],
        "x_upstream_bf16": 1.0,
        "x_A26_reachable": 1.0 / A26,
        "over_per_tensor_bar": floor["over_per_tensor_bar"],
        "over_per_tensor_bar_mass": floor["over_per_tensor_bar_mass"],
    }

    # per block: our renorm arm against upstream's own bf16 floor for THAT block
    per_block, ratios = {}, []
    for b in sorted(fpb, key=int):
        f = fpb[b]["mass_weighted_rel_l2"]
        o = arms["RENORM"]["per_block"][b]["mass_weighted_rel_l2"]
        r = o / f if f else float("inf")
        ratios.append(r)
        per_block[b] = {"n": fpb[b]["n"],
                        "mass_share_of_the_stack": fpb[b]["mass_share_of_the_stack"],
                        "upstream_bf16_vs_float64": f,
                        "ours_renorm_vs_float64": o,
                        "x_upstream_bf16": r,
                        "inside_A26": r <= A26}
    inside = sum(1 for r in ratios if r <= A26)
    at_or_better = sum(1 for r in ratios if r <= 1.0)
    worst_b = max(per_block, key=lambda b: per_block[b]["x_upstream_bf16"])
    best_b = min(per_block, key=lambda b: per_block[b]["x_upstream_bf16"])

    # --- the per-block decomposition must recompose to the published headline -----------------
    # Otherwise the per-block table is a different measurement wearing the headline's name, which
    # is exactly what D84 was (two columns from differently-scoped sets, cross-compared). The
    # identity: mass_weighted_rel = sqrt(sum_b mass_b * rel_b^2) when the mass shares sum to 1.
    import math
    def _recompose(key):
        return math.sqrt(sum(per_block[b]["mass_share_of_the_stack"] * per_block[b][key] ** 2
                             for b in per_block))
    _checks = {}
    for _key, _pub, _lbl in ((["ours_renorm_vs_float64"][0],
                              arms["RENORM"]["ours_vs_float64"]["mass_weighted_rel_l2"], "ours"),
                             ("upstream_bf16_vs_float64",
                              floor["mass_weighted_rel_l2"], "upstream bf16")):
        _got = _recompose(_key)
        _rel = abs(_got - _pub) / _pub if _pub else float("inf")
        _checks[_lbl] = {"recomposed": _got, "published": _pub, "relative_difference": _rel}
        if _rel > 1e-12:
            print(f"REFUSING: the per-block table does not recompose to the published {_lbl} "
                  f"headline ({_got!r} vs {_pub!r}, rel {_rel:.3e}) -- the blocks are a different "
                  f"measurement and must not be reported beside it")
            return 1
    _mass_sum = sum(per_block[b]["mass_share_of_the_stack"] for b in per_block)
    if abs(_mass_sum - 1.0) > 1e-9:
        print(f"REFUSING: the block mass shares sum to {_mass_sum!r}, not 1.0")
        return 1

    outside = [b for b in per_block if not per_block[b]["inside_A26"]]
    mass_outside = sum(per_block[b]["mass_share_of_the_stack"] for b in outside)

    res = {
        "what": "D8 re-stated against upstream 0.4.3's own bf16 training step instead of float64",
        "scope_honesty": "pairformer stack, 2736 tensors, 48 blocks, BOTH tracks -- a SUPERSET of "
                         "D8's 'one assembled block, pair track'. Reported per block so the "
                         "block-level claim is visible; the pair track alone is NOT re-scored.",
        "source": str(SRC),
        "per_tensor_bar": bar,
        "denominator_A27": "SCOPE_c64.json::floor_their_bf16_vs_float64 -- upstream 0.4.3's own "
                           "bf16-autocast training step against grads_f64_043.pt, same 2736 "
                           "tensors, same scorer, same run",
        "headline": rows + [floor_row],
        "per_block_renorm_vs_upstream_bf16": per_block,
        "recomposition_check": _checks,
        "per_block_summary": {
            "blocks": len(ratios),
            "outside_A26": len(outside),
            "mass_share_held_by_blocks_outside_A26": mass_outside,
            "at_or_better_than_upstream_bf16": at_or_better,
            "inside_A26_reachable_sqrt2": inside,
            "median_x_upstream_bf16": statistics.median(ratios),
            "worst_block": {"block": worst_b, **per_block[worst_b]},
            "best_block": {"block": best_b, **per_block[best_b]},
        },
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"per-tensor bar {bar}\n")
    print(f"{'arm':<48} {'rel vs f64':>12} {'x upstream':>11} {'over bar':>9} {'mass over':>10}")
    for r in rows + [floor_row]:
        print(f"{r['arm']:<48} {r['mass_weighted_rel_l2_vs_float64']:>12.6e} "
              f"{r['x_upstream_bf16']:>11.4f} {r['over_per_tensor_bar']:>9d} "
              f"{r['over_per_tensor_bar_mass']:>10.4f}")
    s = res["per_block_summary"]
    print(f"\nper block, renorm arm against upstream's own bf16 for the SAME block:")
    print(f"  {s['at_or_better_than_upstream_bf16']} of {s['blocks']} at or better than upstream's "
          f"own bf16; {s['inside_A26_reachable_sqrt2']} of {s['blocks']} inside A26's sqrt(2)")
    print(f"  {s['outside_A26']} of {s['blocks']} OUTSIDE A26, holding "
          f"{s['mass_share_held_by_blocks_outside_A26']*100:.2f} % of the stack's gradient mass")
    print(f"  median {s['median_x_upstream_bf16']:.4f}x   "
          f"worst block {s['worst_block']['block']} at {s['worst_block']['x_upstream_bf16']:.4f}x   "
          f"best block {s['best_block']['block']} at {s['best_block']['x_upstream_bf16']:.4f}x")
    print("\nper-block table recomposes to the published headline: "
          + ", ".join(f"{k} rel {v['relative_difference']:.3e}" for k, v in _checks.items()))
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
