#!/usr/bin/env python3
"""p115 -- the optimistic-bound arithmetic for the RFD3 b=8 floor claim.

The brief allows exactly two conclusions: the bar met at <= 51.896 s/design, or a floor -- and a
floor "only after Jobs 1 and 2 are BUILT and MEASURED", with the re-run census, the per-op roof
fractions, the optimistic-bound arithmetic, and the measured accuracy cost of everything declined.

This is the arithmetic. Every row carries its instrument and its artifact, and the bound is
deliberately GENEROUS: where a lever has both a measured value and a larger earlier estimate, the
larger one is used, so the conclusion cannot be an artefact of pessimism. Rows that are only a
ceiling (the site deleted to zero) say so, and are still granted in full.
"""
import json
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p115/floor.json")

BASELINE = 94.087          # card-1 baseline, s/design at b=8, page fixture
BAR = 51.896               # 4x the GPU denominator of record
GPU = 12.974               # H200, 8 designs in 103.79 s -- the b=8 number of record

# (label, generous s/design, instrument, bit-exact?, note)
BOARD = [
    ("Job 1, block-sparse atom attention", 3.744,
     "cost model p103/p105 (+3.744); FOLD-MEASURED p114 median n=5 (+3.232). Granting the larger.",
     False, "built, default-off"),
    ("process_z collapse", 3.208,
     "fold A/B p91; independently corroborated by p111's 3.766 prediction", False, "built"),
    ("head_dim 48 -> 64", 3.118,
     "board's isolated 5.80 / 1.86 DiT calibration; SCREENED p112 at ~1.06. Granting 3.118.",
     False, "not built; weight-layout change"),
    ("pair Transition remainder", 2.5,
     "isolated ceiling / 2.95; 7.058 already landed", None, "unknown exactness"),
    ("batching b=2", 1.226, "fold A/B E6.3", True, "built, capped by _BATCH_SPEED_CAP"),
    ("DiT half of L5a", 0.58, "isolated 1.07 / 1.86", None, "BLOCKED, two kernels failed"),
    ("pairformer attn, DELETED TO ZERO", 2.288,
     "p113 2.777 isolated / 1.214 token-encoder calibration", None, "CEILING, not a lever"),
    ("pairformer s_transition + block residual, DELETED TO ZERO", 1.241,
     "p113 (0.230 + 1.277) isolated / 1.214", None, "CEILING, not a lever"),
]

# What a free, exact, perfect atom attention would be worth -- the rigorous bound from E5.3,
# measured twice (p83 fold-valid 17.364; p101's dense control 17.86 isolated at a 1.00 calibration).
ATOM_SITE_TOTAL = 17.364


def main():
    deficit = BASELINE - BAR
    granted = sum(r[1] for r in BOARD)
    reached = BASELINE - granted
    print("baseline %.3f s/design   bar %.3f   deficit %.3f   GPU denominator %.3f"
          % (BASELINE, BAR, deficit, GPU))
    print("current ratio %.3f x\n" % (BASELINE / GPU))
    print("%-56s %8s  %s" % ("lever (granted at its most generous value)", "s/design", "exact?"))
    print("-" * 92)
    for label, val, instr, exact, note in BOARD:
        print("%-56s %8.3f  %s" % (label, val,
                                   {True: "yes", False: "NO", None: "?"}[exact]))
    print("-" * 92)
    print("%-56s %8.3f" % ("TOTAL GRANTED", granted))
    print("%-56s %8.3f  = %.2f x" % ("s/design reached", reached, reached / GPU))
    print("%-56s %8.3f" % ("SHORTFALL against the bar", deficit - granted))
    print()
    print("So the floor on the identified board is %.2f x, and reaching 4x needs a further"
          % (reached / GPU))
    print("%.3f s/design that no pass in this lineage has identified." % (deficit - granted))
    print()
    print("Independent bound, needing no lever list (E5.3, measured twice): the ENTIRE atom")
    print("attention site is %.3f s/design fold-valid, so a free, exact, perfect atom attention"
          % ATOM_SITE_TOTAL)
    print("still leaves %.3f s/design of the deficit unpaid -- %.0f %% of it."
          % (deficit - ATOM_SITE_TOTAL, 100 * (deficit - ATOM_SITE_TOTAL) / deficit))
    print()
    exact_only = sum(r[1] for r in BOARD if r[3] is True)
    print("Of the granted total, only %.3f s/design is bit-exact (%.1f %%). Everything else needs"
          % (exact_only, 100 * exact_only / granted))
    print("the accuracy envelope, and the measured cost of the two largest is:")
    print("  block-sparse   sequence identity 0.9080 (63/685 residues), backbone RMSD 1.1669 A,")
    print("                 median per-atom shift 0.0000, p99 5.1186 (p106/p106b)")
    print("  head_dim 64    maxabs 3.125e-02, K accumulation regrouped 24 -> 32 tiles (p112)")
    print("  288-route      ~1 bf16 ulp, rel median 5.0e-3 -- DECLINED, subset of process_z (p110)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        baseline_s_per_design=BASELINE, bar_s_per_design=BAR, gpu_denominator=GPU,
        current_ratio=round(BASELINE / GPU, 4), deficit=round(deficit, 3),
        board=[dict(lever=l, granted_s_per_design=v, instrument=i, bit_exact=e, note=n)
               for l, v, i, e, n in BOARD],
        total_granted=round(granted, 3),
        s_per_design_reached=round(reached, 3),
        ratio_reached=round(reached / GPU, 4),
        shortfall_vs_bar=round(deficit - granted, 3),
        bit_exact_granted=round(exact_only, 3),
        atom_site_total_fold_valid=ATOM_SITE_TOTAL,
        unpaid_even_with_free_atom_attention=round(deficit - ATOM_SITE_TOTAL, 3),
    ), indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
