# The fit that defines the C10 target, re-examined

The campaign's objective — work term 15,355 -> 9,584, a 37.6 % cycle cut — comes from a two-point
fit of `t = a + b/f` through folds 1 and 6 of a six-fold sweep. Three of those six folds ran with a
foreign TT holder on the chip, all six carry an arithmetic-mean clock label for a clock that swung
up to 1350 MHz *inside* the fold, and nobody checked the fit against a point it was not built from.
Two such points exist: the eight-fold pinned arm at a sampled min = max = 1350 MHz (14.554 s median)
and its unpinned sibling at a mean 1339.2 MHz (14.650 s median).

| fit | n | intercept s | slope MHz·s | error at 1350 | error on the unpinned arm |
|---|---|---|---|---|---|
| two-point, the campaign's | 2 | 2.901 | 15,355.3 | −0.279 s | −0.284 s |
| six-point least squares | 6 | 5.210 | 13,516.8 | +0.669 s | +0.653 s |
| **clean folds + pinned arm** | **4** | **3.478** | **14,960** | **+0.005 s** | **−0.002 s** |
| co-tenanted folds + pinned arm | 4 | 0.627 | 18,582.4 | −0.163 s | −0.148 s |

The unpinned-arm column is the honest test: that arm is in none of the fits. The clean fit — the
three co-tenant-free folds plus the pinned measurement — predicts it to **1.9 ms, 0.013 %**. The
campaign's two-point fit misses by 284 ms and the six-point least squares by 653 ms, because both
are dominated by folds whose clock label is an arithmetic mean of a quantity that swung from 850 to
1350 MHz during the fold.

So the inverse-clock model itself is sound, and cycles are a currency worth trading in. But the
target the campaign has been quoting is built on the least-supported of the four fits.

## What the target actually is

Under the clean fit, at a pinned 1350 MHz:

| goal | what it demands |
|---|---|
| 10.0 s, fixed cost untouched at 3.478 s | work 14,960 -> 8,805, a **41.1 %** cycle cut |
| 10.0 s, fixed cost cut to 2.0 s | work 14,960 -> 10,800, a **27.8 %** cycle cut |
| 10.0 s, fixed cost cut to 1.0 s | work 14,960 -> 12,150, a **18.8 %** cycle cut |
| every device cycle deleted | the floor is the 3.478 s fixed cost |

The fixed term is **3.478 s, 24 % of a 14.559 s fold, and completely clock-immune**. Cutting it to
1.0 s is worth 2.478 s on its own and would read about 12.08 s with no device cycle touched — larger
than any op-level lever this project has landed. It is also the one nobody has attacked since
`b2x-host-residual`, whose `host_residual.py` predates `PairConditioningDevice`, `PairAssemblyDevice`,
`ConfidenceHeadsDevice` and `forward_atoms` going device-resident, so its old residual is not today's
opportunity.

Read the two rows together: 10.0 s by device-cycle deletion alone needs 41 %, which is a very large
ask against a fold whose prior campaigns could not find a single lever worth more than a few percent.
10.0 s as *2.5 s of fixed cost plus a 19 % cycle cut* is a different, far more plausible campaign,
and it says where the chips should go.

## Caveats, and they are real

- The historical six ran on commit `46812858` and the pinned eight on `0df13ad9`. The clean fit
  crosses source trees, so its intercept carries whatever differs between them.
- The pinned and unpinned arms sit 11 MHz apart, so the out-of-sample check validates the model
  near 1350 MHz. The intercept's leverage comes from the 850 MHz folds, and it is not independently
  confirmed.
- An intercept is not measured host time. It is whatever does not scale with the clock, which
  includes clock-immune device and dispatch cost as well as CPU work. `c10-fold-census`'s CLOSURE
  line is what will actually split it.
- Nothing here is a new measurement. It is arithmetic on a banked audit of folds someone else ran.

## Reproduce

    python3 reexamine_fit.py            # prints fit_reexam.json
    python3 -m pytest test_reexamine_fit.py -q   # 6 known-answer controls

The controls recover an exact synthetic (a, b) from two and from five points, refuse a single point
and refuse two points sharing a clock label, and show the fitter visibly missing a linear-in-f curve
rather than reporting a false fit. The report test re-runs the script and compares it to the
committed JSON.
