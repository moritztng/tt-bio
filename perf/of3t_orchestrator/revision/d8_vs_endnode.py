"""How much of D8 is the ending-node function mismatch?

D8: every pair-track sub-module passes alone while the assembled pairformer block reads 4.3e-01
to 1.4e+00. D23 says our model and the 0.5.0 reference compute different ending-node functions,
which would make the assembled comparison invalid at every block.

The `tb-off` arms already on the branch equalise the two sides (they move OUR model onto 0.5.0's
convention -- the wrong side, but the mismatch is what is being removed, so it is a valid proxy
for the question). Recomputed here with the campaign's own A14 rule and bars rather than read
from a summary. Run against perf/of3t_gradients/ in the composition.
"""
import json, sys, numpy as np
BAR, MBAR, ZERO_REF = 5.0e-2, 2.0e-2, 1e-12
D = sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_gradients"
ARMS = [("block0  crop64 SHIPPED", "instrument_a_bundle_block0_r0_crop64.json"),
        ("block0  crop64 tb-off ", "instrument_a_bundle_block0_r0_crop64_tboff.json"),
        ("block23 crop64 SHIPPED", "instrument_a_bundle_block23_r0_crop64_tbshipped.json"),
        ("block23 crop64 tb-off ", "instrument_a_bundle_block23_r0_crop64_tboff.json"),
        ("block47 crop64 SHIPPED", "instrument_a_bundle_block47_r0_crop64_tbshipped.json"),
        ("block47 crop64 tb-off ", "instrument_a_bundle_block47_r0_crop64_tboff.json")]
out = []
for tag, f in ARMS:
    try:
        rows = json.load(open(f"{D}/{f}"))["per_parameter"]
    except FileNotFoundError:
        print(f"{tag}: ABSENT"); continue
    kept = [r for r in rows if r["ref_norm"] >= ZERO_REF]
    v = np.array([r["rel_l2"] for r in kept]); med = float(np.median(v))
    over = [r for r in kept if r["rel_l2"] > BAR]
    w = max(kept, key=lambda r: r["rel_l2"])
    sq = sum(r["ref_norm"] ** 2 for r in kept) or 1.0
    rec = {"arm": tag.strip(), "n": len(kept), "median": med,
           "median_inside_bar": med <= MBAR, "over_bar": len(over),
           "over_bar_norm_share": sum(r["ref_norm"] ** 2 for r in over) / sq,
           "worst": w["rel_l2"], "worst_tensor": w["their_tensor"]}
    out.append(rec)
    print(f"{tag}: n={rec['n']}  median={med:.4f} [{'PASS' if rec['median_inside_bar'] else 'FAIL'}]  "
          f"over-bar {len(over)}/{len(kept)}  norm share {100*rec['over_bar_norm_share']:5.1f}%  "
          f"worst {w['rel_l2']:.4f} @ {w['their_tensor']}")
json.dump({"bars": {"per_tensor": BAR, "median": MBAR}, "a14_zero_ref": ZERO_REF, "arms": out},
          open("d8_vs_endnode.json", "w"), indent=1)
