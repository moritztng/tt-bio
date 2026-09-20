#!/usr/bin/env python3
"""The campaign's closing arithmetic: every route, priced on ONE part, against the charter.

CPU only, no device. Consolidates ceiling.py (fold share), roofcheck.py (roof self-consistency)
and verdict.py (tmk-assumptions' re-pricing), adds tmk-kernel's measured geometry lever, and
re-derives tmk-kernel's fold projection on p300c numbers throughout -- that row measured on qb1
p150a and carried its module time against a p300c fold, which this campaign's own standard
forbids ("a ratio from one part is not a ratio on the other").
"""

# --- p300c, AICLK forced and during-verified 1350 MHz, cdk2x2_512 -------------------------
FOLD_S = 48.62                # trix-scaffold-attribute
MODULE_IN_FOLD_S = 12.6881    # all 1208 trimul calls in that fold
MODULE_MS = 11.519            # trix-floor, standalone unmasked per call
CONTRACTION_SHARE = 0.1414    # trix-orchestrator's split of the module

# tmk-assumptions, qb2 p300c node 3: how much of the booked rate deficit survives
A_BEST, A_PLAUSIBLE = 1.774, 0.906          # ms/call recoverable
# tmk-assumptions granule arms; the baseline is disputed (see BASELINE-DISPUTE)
B_SHIPPED_MS = 2.121
B_HEADROOM = {"against ttnn.permute, 63.3 GB/s (disputed baseline)": 364.7 / 63.3,
              "against trix-radical's 158 GB/s": 364.7 / 158.0,
              "against our own kernel, perfwar's 221 GB/s": 364.7 / 221.0}
# tmk-kernel, qb1 p150a: the geometry lever, per core, and its op-level ceiling
GEOMETRY_OP_CEILING = 1.82


def fold(after_ms):
    return FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + MODULE_IN_FOLD_S * after_ms / MODULE_MS)


def show(label, after):
    print(f"  {label:<56} module {MODULE_MS / after:6.4f}x   fold {fold(after):6.4f}x")


print(f"trimul is {MODULE_IN_FOLD_S / FOLD_S * 100:.2f} % of the fold "
      f"({MODULE_IN_FOLD_S:.4f} s of {FOLD_S:.2f} s).\n")

print("THE HARD CEILING -- no kernel can exceed this, whatever it does")
show("trimul costs literally nothing", 1e-9)
print()

print("EVERY ROUTE THE CAMPAIGN FOUND, each priced at ITS OWN BEST CASE")
show("A  rate, all of tmk-assumptions' recoverable deficit", MODULE_MS - A_BEST)
show("A  rate, its plausible same-mix case", MODULE_MS - A_PLAUSIBLE)
for label, h in B_HEADROOM.items():
    show(f"B  exchange, {label}", MODULE_MS - B_SHIPPED_MS * (1 - 1 / h))
# tmk-kernel's surviving design, re-derived on p300c: it improves the CONTRACTION only
contr_ms = MODULE_MS * CONTRACTION_SHARE
show(f"A' tmk-kernel's sub-grid design, {GEOMETRY_OP_CEILING}x on the contraction",
     MODULE_MS - contr_ms * (1 - 1 / GEOMETRY_OP_CEILING))
print(f"     (the contraction is {CONTRACTION_SHARE * 100:.2f} % of the module = "
     f"{contr_ms:.3f} ms/call; tmk-kernel's own p150a-on-p300c figure was 1.0183x)")
print()

best_b = B_SHIPPED_MS * (1 - 63.3 / 364.7)
real_b = B_SHIPPED_MS * (1 - 221.0 / 364.7)
print("STACKED, both routes, perfectly built")
show("A best + B at the generous disputed baseline", MODULE_MS - A_BEST - best_b)
show("A best + B against our own kernel", MODULE_MS - A_BEST - real_b)
show("A plausible + B against our own kernel", MODULE_MS - A_PLAUSIBLE - real_b)
print()

print("AGAINST THE CHARTER")
print('  Moritz, 2026-09-21: "i want to get a big speedup ... make a big breakthrough."')
print(f"  The largest number in this document is {fold(1e-9):.4f}x, and it requires the operation")
print("  to cost nothing. Every route that can actually be built lands between 1.02x and 1.09x.")
print(f"  K-C, pre-registered: a new build below 1.05x on the fold stops the campaign.")
for label, after in (("A best", MODULE_MS - A_BEST),
                     ("A' tmk-kernel's design", MODULE_MS - contr_ms * (1 - 1 / GEOMETRY_OP_CEILING)),
                     ("B against our own kernel", MODULE_MS - real_b),
                     ("B at the generous baseline", MODULE_MS - best_b)):
    f = fold(after)
    print(f"    {label:<28} {f:.4f}x  {'CLEARS' if f >= 1.05 else 'FIRES K-C'}")
