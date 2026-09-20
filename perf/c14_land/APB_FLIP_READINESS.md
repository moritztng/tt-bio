# What flipping `TT_BIO_APB_CONCAT_HEADS` actually touches, and what the excluded gate arm would have said

Two questions this row had flagged as open and never answered. Both are settled from source and
arithmetic, no card.

## 1. The flip is one line of production code plus its docs entry

Searched every `.py`, `.md`, `.sh`, `.yaml` and `.json` in the repo for `APB_CONCAT_HEADS`. Outside
this row's own artifacts and `perf/roof_concat/`, every hit is a historical perf record —
`perf/ttx_a3/`, `perf/c14_bfp8/` census and A/B JSONs. There is:

* no second production read of the flag (`tt_bio/tenstorrent.py:1565` defines it, `:8168` consumes it),
* no test that pins the default — `grep -rn 'APB_CONCAT_HEADS\|_concat_heads' tests/` is empty, so
  the flip breaks no test and, equally, no test would have caught it going the wrong way,
* no catalog, platform or release-note reference.

So the flip is exactly:

    tt_bio/tenstorrent.py:1565   env_flag("TT_BIO_APB_CONCAT_HEADS", False) -> True
    docs/tuning-flags.md         heading `— off` -> `— on`, and the "why it is off" paragraph
                                 replaced by what a user needs instead

This matters because "it is one line" is the kind of claim that has been wrong here before
(`hf-revision-pin-fix-missed-three-direct-callers`). It was checked rather than assumed.

## 2. The `size-ladder` arm I excluded could not have red-flagged this lever

`gate_apb.sh` excludes `size-ladder`, the gate's one timed arm, because it takes no benchlock, runs
no guard, and has already produced two false reds in this campaign. That exclusion is honest but it
left an unquantified risk: this is a PERF lever, and `size-ladder` is the perf arm.

Quantified now, and it is negligible. The arm does not gate absolute seconds. It gates the **scaling
exponent** k between rungs, with

    tol = max(0.50, 3 * sqrt(2) * sigma / ln(N2/N1))

(`scripts/release_gate.py:655`; boltz2 reads sigma 6.5 %, giving tol 0.50 on 256->512).

A lever that scales the runtime of every rung by the same factor leaves k **exactly** unchanged: the
factor cancels inside `ln(T2/T1)`. APB is a per-call saving at the token head re-assembly — 5064
calls at 512 aa — so it is close to proportional, and the cancelling case is the realistic one.

Taking the worst case instead, a *constant* 0.07 s saved at every rung, with 256 aa near 8.0 s and
512 aa near 14.6 s:

    k before = ln(14.60 / 8.00) / ln(2) = 0.8682
    k after  = ln(14.53 / 7.93) / ln(2) = 0.8740
    shift    = 0.0058

against a tolerance floor of **0.50**. About 1 % of the band, in the arm's most sensitive direction.

**So excluding `size-ladder` does not hide a perf regression this lever could plausibly cause.** It
is still owed for completeness on a quiet host, and it should still be run before a tag — but it is
not a blocker for the flip decision, and this row should stop carrying it as an unknown.

Stated limit: the 0.05-0.08 s figure is measured at 512 aa only. If APB's saving were far larger at
one rung than another the exponent could move more, but the mechanism is per attention call, so the
saving tracks call count and therefore size.

## 3. There is no cheap substitute for the paired cross-model reading. I looked.

The gate prints an RMSD per arm, and with seven arms now running the flag it is tempting to read
those as cross-model accuracy evidence and skip the five-fold paired run. They are not, and the
reason is measurable rather than theoretical.

The gate's APB-ON numbers on qb2 p300c, prot.yaml, 200 steps / 5 samples, seed 0:

    boltz2 1.185   rf3 1.240   opendde 1.411 (TM 0.944)   protenix-v2 1.465
    openbind 1.493   protenix-v1 1.616   openfold3 1.755

To turn any of those into evidence about the flag you need the same fixture with the flag OFF. The
only stored gate table in the tree with per-model RMSDs is
`perf/inblockw/qb1/gate/release_gate_summary.txt` — same fixture, same protocol, same seed, but a
**qb1 p150a** on a different tree. It reads:

    opendde       3.144   TM 0.835      against qb2's 1.411 / 0.944 here
    protenix-v2   1.422   TM 0.948      against qb2's 1.465 here

**OpenDDE moves 1.73 A between board class and tree, with the flag off in both.** That is what an
unpaired comparison of this fixture carries before the lever is even considered, and it is roughly
8x the 0.2244 A that APB moves Boltz-2. So the gate's per-arm RMSDs cannot resolve this flag, and
the MODELS table does not help either: it stores floors (`max_rmsd`, `min_tm`), not a recorded
measured value to difference against.

This is `unpaired-cross-stack-rmsd-carries-full-seed-floor` with a number attached. The paired,
same-seed, same-box, same-tree five-fold run in `apb_xmodel_accuracy.sh` is not belt-and-braces —
it is the only design that can see an effect this size, which is why its A/A control must read
0.000000 A before anything else in it is readable.
