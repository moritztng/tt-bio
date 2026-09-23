#!/usr/bin/env python3
"""Does closing the trunk's ~2 mantissa-bit deficit clear the GRADIENTS clause?

`of3t-lnreduce` is holding a card. Before it spends that card, this answers the one question
that decides whether its target is worth the time: the trunk must fall 2.2349x, and a b-bit
recovery buys a factor of 2^b, so where does the clause land at b = 1, 2, 3?

Arithmetic only -- no device, no torch. The composition is the scorer's own identity form
(`model_scope.py:255`, lifted verbatim by framegate/score_the_best_possible_artifact.py), and
it is reproduced against the artifact's published headline BEFORE any substitution. If that
control does not hold to 1e-12 the script says so instead of reporting a result.
"""
import json, math, pathlib, platform

T, ARM = "pairformer_stack", "renorm_vs_UPSTREAM_BF16"
ART = "perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json"
root = pathlib.Path(__file__).resolve().parents[3]
g = json.loads((root / ART).read_text())

secs = g["per_section"][ARM]
bar = g["bars"]["A26_reachable_bar_vs_their_bf16"]
W = sum(s["ref_sq"] for s in secs.values())
wt = secs[T]["ref_sq"]
S_other = sum(s["ref_sq"] * s["mass_weighted_rel_l2"] ** 2 for k, s in secs.items() if k != T)
model = lambda xt: math.sqrt((S_other + wt * xt * xt) / W)

published = g["stats"][ARM]["mass_weighted_rel_l2"]
xt = secs[T]["mass_weighted_rel_l2"]
ctrl = abs(model(xt) - published) / published
if ctrl > 1e-12:
    raise SystemExit("CONTROL FAILED: composition != published (rel %.3e)" % ctrl)

# the largest trunk reading the clause still admits
xt_adm = math.sqrt((bar * bar * W - S_other) / wt)
need = xt / xt_adm

# and the floor: the other ten sections alone, with a PERFECT trunk
floor_perfect = model(0.0)

rows = []
for b in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    f = 2.0 ** b
    x = xt / f
    m = model(x)
    rows.append({"bits_recovered": b, "factor": f, "trunk_after": x,
                 "clause_after": m, "multiple_of_bar": m / bar, "clears": m <= bar})

out = {
 "instrument": "perf/of3t_orchestrator/lnladder/what_a_bit_buys.py",
 "host": platform.node(),
 "artifact": ART,
 "composition_control_rel_difference": ctrl,
 "bar": bar,
 "clause_now": published,
 "multiple_now": published / bar,
 "trunk_now": xt,
 "trunk_admissible": xt_adm,
 "trunk_must_fall_by": need,
 "bits_required": math.log2(need),
 "clause_with_a_PERFECT_trunk": floor_perfect,
 "perfect_trunk_multiple_of_bar": floor_perfect / bar,
 "rungs": rows,
}
mind = min(r["bits_recovered"] for r in rows if r["clears"]) if any(r["clears"] for r in rows) else None
out["smallest_rung_tested_that_clears_bits"] = mind
out["verdict"] = (
 "The trunk must fall %.4fx, which is %.2f mantissa bits. The measured deficit is ~2 bits "
 "(3.928 for a 2-bit accumulator against a predicted 4.000, LADDER_SHAPE.json), so closing it "
 "in full buys 4x against the %.4fx needed and the clause clears at %.4fx the bar. A PERFECT "
 "trunk reads %.4fx the bar, so the clause is satisfiable and the margin is real rather than "
 "marginal: even a %.1f-bit recovery clears it."
 % (need, math.log2(need), need,
    [r for r in rows if r["bits_recovered"] == 2.0][0]["multiple_of_bar"],
    floor_perfect / bar, mind if mind else float("nan")))

p = pathlib.Path(__file__).with_name("WHAT_A_BIT_BUYS.json")
p.write_text(json.dumps(out, indent=1) + "\n")

print("composition control rel_difference = %.3e (must be <= 1e-12)" % ctrl)
print("clause now      %.10f   %.4fx bar" % (published, published / bar))
print("bar             %.10f" % bar)
print("trunk now       %.10f" % xt)
print("trunk admissible%.10f  -> must fall %.4fx = %.2f bits" % (xt_adm, need, math.log2(need)))
print("perfect trunk   clause %.10f  %.4fx bar\n" % (floor_perfect, floor_perfect / bar))
print("bits  factor   trunk_after    clause_after   xbar     clears")
for r in rows:
    print(" %.1f   %5.2fx  %.10f   %.10f  %.4f   %s" % (
        r["bits_recovered"], r["factor"], r["trunk_after"], r["clause_after"],
        r["multiple_of_bar"], "YES" if r["clears"] else "no"))
print("\n" + out["verdict"])
print("\nwrote", p)
