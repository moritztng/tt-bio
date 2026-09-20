#!/usr/bin/env python3
"""Is the pessimistic roof even self-consistent with the module's own measured time?

`tmk-assumptions` owns K-A: whether 123.65 TFLOP/s is the right roof for the trimul's shapes, or
whether the binding roof is far lower -- `c13-matmul-rate` implies 43.9 from a 101.1 FLOP/byte
arithmetic intensity against a 276.8 crossover, and `perfwar-trimul-kernel` measured 35.20 at
K=256 on the trimul's own shape.

That question has an arithmetic pre-check that needs no device, and this script is it: a proposed
roof must be at least as fast as the rate the module ALREADY achieves. A roof the shipped code
already beats is not a roof.

CPU only. Run: python3 perf/tmk_orchestrator/roofcheck.py
"""

# trix-floor / trix-radical, qb2 p300c card 0, AICLK forced and during-verified 1350 MHz.
ARITH_GFLOP = 274.877        # 12*N^2*D^2 + 2*N^3*D at N=512, D=256, from the definition
ARITH_AT_ROOF_MS = 2.223     # the same arithmetic priced at 123.65 TFLOP/s
RATE_DEFICIT_MS = 3.491      # the extra the call sites pay at their own rates
MEASURED_ROOF = 123.65       # square 4096^3 pipelined, measured

CANDIDATE_ROOFS = {
    "123.65  trix-floor, measured square 4096^3 pipelined, pinned 1350 MHz": 123.65,
    " 43.9   c13-matmul-rate, implied by 101.1 FLOP/byte vs a 276.8 crossover": 43.9,
    " 35.20  perfwar-trimul-kernel S7.0, measured at K=256 on trimul's own shape": 35.20,
    "118.24  perfwar-trimul-kernel S7.0, measured at K=1024, same K-curve": 118.24,
}

# sanity: the arithmetic and the roof must reproduce the quoted floor
derived = ARITH_GFLOP / MEASURED_ROOF  # ms, since GFLOP/TFLOPs = ms
print(f"consistency: {ARITH_GFLOP:.3f} GFLOP / {MEASURED_ROOF} TFLOP/s = {derived:.4f} ms "
      f"against the quoted {ARITH_AT_ROOF_MS:.3f} ms  "
      f"({abs(derived - ARITH_AT_ROOF_MS) / ARITH_AT_ROOF_MS * 100:.2f} % apart)\n")

site_ms = ARITH_AT_ROOF_MS + RATE_DEFICIT_MS
site_rate = ARITH_GFLOP / site_ms
print(f"THE MODULE'S OWN ACHIEVED RATE, implied by the decomposition it is scored against:")
print(f"  the call sites deliver {ARITH_GFLOP:.3f} GFLOP in {site_ms:.3f} ms "
      f"= {site_rate:.2f} TFLOP/s, weighted over all three matmul classes.")
print(f"  (the per-class rates quoted separately are 54.97 / 42.20 / 43.34.)\n")

print("CANDIDATE ROOFS, each checked against that:")
for label, roof in sorted(CANDIDATE_ROOFS.items(), key=lambda kv: -kv[1]):
    ms = ARITH_GFLOP / roof
    if roof <= site_rate:
        note = (f"IMPOSSIBLE as the binding roof -- the shipped module already runs at "
                f"{site_rate:.2f}, which is {site_rate / roof:.2f}x this figure")
    else:
        deficit = site_ms - ms
        note = (f"leaves {deficit:.3f} ms of deficit, {deficit / 11.519 * 100:.1f} % of the "
                f"11.519 ms module, headroom {roof / site_rate:.2f}x")
    print(f"  {label}\n      {ms:7.3f} ms for the arithmetic -- {note}")

print()
print("WHAT THIS SETTLES AND WHAT IT DOES NOT")
print("  Settles: 43.9 and 35.20 cannot both be the trimul module's binding roof AND be below the")
print("  rate the module already achieves. Whatever they measure, it is not a ceiling this module")
print("  is sitting under -- the shipped code is already faster than both.")
print("  Does not settle: the roof could still be well below 123.65 without being below 48.11.")
print("  Any roof R in (48.11, 123.65) leaves a real but smaller prize, and that interval is")
print("  exactly what tmk-assumptions has to close with an arithmetic-intensity check at the")
print("  SHIPPED Kt, at a pinned and during-sampled clock.")
print()
print("THE PRIZE ACROSS THAT INTERVAL, so the decision is mechanical when the number arrives:")
print(f"  {'roof':>8}  {'arith ms':>9}  {'deficit ms':>11}  {'module':>8}  {'fold':>7}")
FOLD_S, MODULE_IN_FOLD_S, MODULE_MS = 48.62, 12.6881, 11.519
for roof in (50, 60, 70, 85, 100, 123.65):
    ms = ARITH_GFLOP / roof
    deficit = max(0.0, site_ms - ms)
    after = MODULE_MS - deficit
    fold = FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + MODULE_IN_FOLD_S * after / MODULE_MS)
    print(f"  {roof:8.2f}  {ms:9.3f}  {deficit:11.3f}  {MODULE_MS / after:7.4f}x  {fold:6.4f}x")

# where does a PERFECT rate route cross the 1.05x worth-building bar?
def fold_at(roof):
    deficit = max(0.0, site_ms - ARITH_GFLOP / roof)
    after = MODULE_MS - deficit
    return FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + MODULE_IN_FOLD_S * after / MODULE_MS)

lo, hi = site_rate, 400.0
for _ in range(200):
    mid = (lo + hi) / 2
    if fold_at(mid) < 1.05: lo = mid
    else: hi = mid
cross = hi
print()
print("READ AGAINST THE 1.05x WORTH-BUILDING BAR FOR A NEW BUILD")
print(f"  A PERFECT rate route -- every matmul class lifted from today's {site_rate:.2f} TFLOP/s")
print(f"  all the way to the roof -- crosses 1.05x on the fold at a roof of "
      f"{cross:.2f} TFLOP/s.")
print(f"  So the bar is clearable in principle, but only by a route that reaches the roof exactly,")
print(f"  and the best rate anyone in this lineage has ever measured on a trimul shape is 47.15,")
print(f"  which is BELOW today's weighted {site_rate:.2f}. At the highest rate ever achieved here")
print(f"  the deficit does not shrink at all.")
print(f"  What a REAL lever buys, priced from rates that exist rather than from the roof:")
for r in (1.135, 1.30, 1.50, 2.00):
    after = MODULE_MS - site_ms * (1 - 1 / r)
    f = FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + MODULE_IN_FOLD_S * after / MODULE_MS)
    print(f"    lever {r:5.3f}x -> {site_rate * r:6.2f} TFLOP/s -> fold {f:.4f}x  "
          f"{'clears' if f >= 1.05 else 'misses'} 1.05x")
lo2, hi2 = 1.0, 20.0
for _ in range(200):
    mid = (lo2 + hi2) / 2
    after = MODULE_MS - site_ms * (1 - 1 / mid)
    f = FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + MODULE_IN_FOLD_S * after / MODULE_MS)
    if f < 1.05: lo2 = mid
    else: hi2 = mid
print(f"  the rate lever that clears 1.05x on the fold is {hi2:.3f}x, i.e. "
      f"{site_rate * hi2:.2f} TFLOP/s sustained across every matmul class in the module.")
print("  That is the number a rate route has to beat, and it is the honest form of K-B.")
