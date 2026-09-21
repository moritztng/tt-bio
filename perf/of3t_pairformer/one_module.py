#!/usr/bin/env python3
"""Does the trunk leave "the failure is one module" standing?

The claim at pass 175 is that `diffusion_module.diffusion_transformer` carries 99.6036 % of the
diffusion arm's error mass against upstream's own bf16 training step, and it is scoped to the
92.1651 % of the model that had a direct reading. `pairformer_stack` -- 5.8282 %, the largest
unread piece -- now has one.

The arithmetic is the campaign's own: error mass = (share of the model's squared gradient norm)
x rel^2, the same formula THE_ERROR_MASS_IS_ONE_MODULE.json uses on of3t-trajectory's table.

WHAT THIS COMPOSITION IS NOT. The diffusion rows are measured at the 043 step (manifest loss
1.267624369070698). The trunk row is measured over the only pairformer boundary that exists, which
was captured from the 0.5.0 step (loss 1.6591175475821072) at a 64-token crop of a 384-token batch
holding 56 real tokens. Two steps, so the composition is an ORDER-OF-MAGNITUDE statement about
where the error mass sits and not a partition of one number. It is stated that way on purpose:
what it has to decide is whether the trunk is a rounding error beside the diffusion transformer,
and 0.4 % against 19 % is not a distinction that turns on the step.
"""
import json
import sys

AGREE = sys.argv[1]
OUT = sys.argv[2]

a = json.load(open(AGREE))
h = a["pairs"]["DEVICE_vs_UPSTREAM_BF16"]["sets"][0]
trunk_rel = h["mass_weighted_rel_l2"]
TRUNK_MASS = 0.058282          # SECTION_MASS_MEASURED.json, the 043 float64 reference
rows = [
    {"scope": "pairformer_stack", "rel": trunk_rel, "mass_frac": TRUNK_MASS,
     "measured_by": "of3t-pairformer, this row",
     "step": "0.5.0 r=0 dropout-disabled boundary, 64-token crop"},
    {"scope": "diffusion_module.diffusion_transformer", "rel": 8.025, "mass_frac": 0.436221,
     "measured_by": "of3t-trajectory, pass 175", "step": "043"},
    {"scope": "the rest of the diffusion arm (5 sections)", "rel": None, "mass_frac": 0.075137,
     "error_mass": 28.20471585719318 - 28.092905038125004,
     "measured_by": "THE_ERROR_MASS_IS_ONE_MODULE.json", "step": "043"},
]
for r in rows:
    if r.get("error_mass") is None:
        r["error_mass"] = r["mass_frac"] * r["rel"] ** 2
tot = sum(r["error_mass"] for r in rows)
for r in rows:
    r["share_of_the_measured_error_mass_pct"] = 100.0 * r["error_mass"] / tot

dit = rows[1]["share_of_the_measured_error_mass_pct"]
trunk = rows[0]["share_of_the_measured_error_mass_pct"]
rep = {
    "what": __doc__.strip().splitlines()[0],
    "caveat": __doc__.strip().split("WHAT THIS COMPOSITION IS NOT.")[1].strip(),
    "trunk_measurement": {
        "mass_weighted_rel_l2_vs_upstream_bf16": trunk_rel,
        "median_over_tensors": h["median_rel_l2_over_tensors"],
        "mass_weighted_norm_ratio": h["mass_weighted_norm_ratio"],
        "mass_weighted_cos": h["mass_weighted_cos"],
        "threshold_floor_over_r":
            a["PRE_REGISTERED_READING"]["threshold_a_perfect_port_would_read__floor_over_r"],
        "multiples_of_that_threshold":
            a["PRE_REGISTERED_READING"]["multiples_of_that_threshold"],
        "measured_zero_model_reading_A16":
            a["PRE_REGISTERED_READING"]["measured_zero_model_reading_A16"],
        "cos_between_the_two_error_vectors":
            a["ERROR_GEOMETRY"]["cos_between_the_two_errors"],
        "our_error_over_theirs": a["ERROR_GEOMETRY"]["ratio_ours_over_theirs"],
    },
    "for_comparison_same_statistic_other_scopes": {
        "diffusion_device_arm_547_tensors": 7.426742,
        "diffusion_module.diffusion_transformer": 8.025,
        "note": "mass-weighted rel_l2 of the device gradient against upstream's own bf16 step, "
                "each on its own scope. The trunk's is LARGER than either.",
    },
    "rows": rows,
    "verdict": (
        f"REFUTED. The trunk reads {trunk_rel:.4f} against upstream's own bf16 training step on "
        f"its own scope, larger than the diffusion arm's 7.4267 and larger than the diffusion "
        f"transformer's 8.025. Composing the error masses, the diffusion transformer holds "
        f"{dit:.1f} % of the measured error mass and the pairformer trunk {trunk:.1f} %, against "
        f"the 0.4 % that 'the failure is one module' leaves for everything outside it. The "
        f"headline has to be rewritten: it is at least two modules, and the second one is the "
        f"trunk."),
}
json.dump(rep, open(OUT, "w"), indent=1)
print(rep["verdict"])
for r in rows:
    print(f"  {r['scope']:52s} mass {r['mass_frac']:.6f}  error mass {r['error_mass']:9.4f}  "
          f"{r['share_of_the_measured_error_mass_pct']:6.2f} %")
print("worst tensor on the trunk scope (A14-filtered):", h["worst_tensor"], h["worst_rel_l2"])
