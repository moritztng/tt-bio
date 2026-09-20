#!/usr/bin/env python3
"""What the TMK campaign can be worth, priced from measurements other rows took.

CPU only, no device. Every input is a measured constant with the doc that took it named
beside it. The point is that the campaign's upper bound is arithmetic, not opinion: the
orchestrator refuses to dispatch a build row against a prize nobody has bounded.

Run: python3 perf/tmk_orchestrator/ceiling.py
"""

# --- measured inputs, each with its source -----------------------------------------------
# trix-scaffold-attribute, qb2 p300c card 1, sibling idle, AICLK FORCED and verified 1350 MHz,
# protenix-v2 cdk2x2_512 fixture, the same one the campaign's ground truth is measured on.
FOLD_S = 48.62          # whole fold
MODULE_IN_FOLD_S = 12.6881  # all 1208 trimul calls inside that fold

# trix-floor / trix-radical, qb2 p300c card 0, pinned + during-verified 1350 MHz.
MODULE_MS = 11.519      # standalone unmasked, per call
ARITH_AT_ROOF_MS = 2.223    # compulsory arithmetic at the measured 123.65 TFLOP/s roof
RATE_DEFICIT_MS = 3.491     # same arithmetic at the call sites' 54.97 / 42.20 / 43.34 TFLOP/s
TRANSPOSE_MS = 2.551        # the 64 B all-to-all, 3Z at 158 GB/s; a 2.5-3.7 ms band
TRANSPOSE_BAND = (2.5, 3.7)

# trix-orchestrator's split of the module, summing to 100.00 %.
CLASS_SHARE = {"projection": .3859, "gated channel move": .2079, "normalisation": .1429,
               "contraction": .1414, "pure layout": .0757, "gating": .0025, "residual": .0437}

# trimul-absolute-optimal S13, qb2 card 0: the only per-core no-multicast contraction anyone
# has measured, against the shipped 2D-mcast program config in the same session.
SHIPPED_CONTRACTION_TFLOPS = 41.54
BEST_MEASURED_TFLOPS = 47.15


def fold_ratio(module_ms_after: float) -> float:
    """Carry a module-level ratio to the fold at trimul's measured in-fold share."""
    after = MODULE_IN_FOLD_S * (module_ms_after / MODULE_MS)
    return FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + after)


def line(label: str, module_ms_after: float) -> str:
    return (f"{label:<46} module {MODULE_MS:6.3f} -> {module_ms_after:6.3f} ms "
            f"({MODULE_MS / module_ms_after:5.4f}x)   fold {fold_ratio(module_ms_after):6.4f}x")


share = MODULE_IN_FOLD_S / FOLD_S
print(f"trimul is {share * 100:.2f} % of the fold "
      f"({MODULE_IN_FOLD_S:.4f} s of {FOLD_S:.2f} s)\n")

print("CEILINGS -- what is available at all, before anyone writes a kernel")
print(line("delete trimul entirely (unreachable)", 1e-9))
print(line("compulsory arithmetic only (unreachable)", ARITH_AT_ROOF_MS))
print()

print("ROUTES -- each takes its own term to zero, perfectly")
rate_only = MODULE_MS - RATE_DEFICIT_MS
print(line("A  rate: every matmul class at the roof", rate_only))
print(line("B  transpose: the 64 B all-to-all deleted", MODULE_MS - TRANSPOSE_MS))
print(line("A+B stacked, both perfect", MODULE_MS - RATE_DEFICIT_MS - TRANSPOSE_MS))
print()

print("ROUTE A priced at rates anyone has actually reached, not at the roof")
print("  arithmetic at the call sites' own rates = "
      f"{ARITH_AT_ROOF_MS + RATE_DEFICIT_MS:.3f} ms; a rate lever of r shrinks that term by 1/r")
for r in (1.135, 1.20, 1.30, 1.50, 2.00):
    after = MODULE_MS - (ARITH_AT_ROOF_MS + RATE_DEFICIT_MS) * (1 - 1 / r)
    tag = "  <- measured, S13 no-mcast halved output" if abs(r - 1.135) < 1e-9 else ""
    print(line(f"   rate lever {r:.3f}x on all matmul classes", after) + tag)
print()

print("WHY THE MEASURED 1.135x WAS DISMISSED, AND WHY THAT IS STALE")
print(f"  S13 screened it on the contraction alone, which the campaign then priced at 8.9 % "
      f"of the module:\n    1.135x on 8.9 %  = "
      f"{1 / (1 - .089 * (1 - 1 / 1.135)):.4f}x on the module  -- correctly dismissed")
print(f"  today's split puts the contraction at {CLASS_SHARE['contraction'] * 100:.2f} % and the "
      f"projection at {CLASS_SHARE['projection'] * 100:.2f} %.")
arith_classes = CLASS_SHARE["projection"] + CLASS_SHARE["contraction"]
print(f"    the same lever on both matmul classes ({arith_classes * 100:.2f} %) = "
      f"{1 / (1 - arith_classes * (1 - 1 / 1.135)):.4f}x on the module, "
      f"fold {fold_ratio(MODULE_MS * (1 - arith_classes * (1 - 1 / 1.135))):.4f}x")
print(f"  so the mechanism is {arith_classes / .089:.2f}x better placed than when it was screened,")
print("  and that re-pricing is the campaign's case for a rate route.")
print()

print("TRANSPOSE BAND (route B is quoted on MIXED PARTS -- read it as a band)")
for t in TRANSPOSE_BAND:
    print(line(f"   transpose term {t:.1f} ms deleted", MODULE_MS - t))
print()

print("KILL ARITHMETIC")
need = 1.05
print(f"  a fold win must clear {need:.2f}x to be worth landing over this campaign's "
      f"0.00-2.43 % A/A twins.")
for r in (1.15, 1.30, 1.50):
    after = MODULE_MS - (ARITH_AT_ROOF_MS + RATE_DEFICIT_MS) * (1 - 1 / r)
    f = fold_ratio(after)
    print(f"  rate lever {r:.2f}x -> fold {f:.4f}x  "
          f"{'CLEARS' if f >= need else 'MISSES'} the {need:.2f}x bar")
print(f"  shipped contraction {SHIPPED_CONTRACTION_TFLOPS} TFLOP/s, best measured alternative "
      f"{BEST_MEASURED_TFLOPS} = {BEST_MEASURED_TFLOPS / SHIPPED_CONTRACTION_TFLOPS:.4f}x.")
print("  so the smallest artifact must beat 1.30x, not 1.135x: 1.135x is a repeat of a "
      "measured dead end.")
