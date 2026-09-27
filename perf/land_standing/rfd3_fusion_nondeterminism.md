# `rfd3-fusion` is non-deterministic on main, and it is not the dividing-k candidate

Read 2026-09-26 23:0xZ from the two release-gate runs this row launched against the SAME commit,
`840c5f844`, on qb1. Nothing here needs a device: both runs already exist on disk.

## The reading

`scripts/release_gate.py`'s `rfd3-fusion` arm censuses `rfd3_R4.json` at 4 timesteps and compares
`RFD3_FC1_SPLIT_SILU`'s served/declined split against a closed form. Two runs, one tree:

    run                          FC1 served   FC1 declined   total   PV served/declined
    gate_dividingk.log           154          14             168     27 / 108
    gate_dividingk_resume.log     35          133            168     27 / 108
    _rfd3_fusion_expected(4)     154          14             168     27

Both runs report the **same three decline clauses**, verbatim:

    rows=45 w=704 hidden=256 no-pinned-config
    rows=64 w=704 hidden=256 no-pinned-config
    tensor_rows=685 w=704 hidden=512 chunk-height-would-move 64->63

So no clause appeared or disappeared, `RFD3_SOFTMAX_PV_FUSED` is identical in both, and the total
is identical in both. **Only the served/declined split moved**, and the first run matched the
expectation exactly.

## Why the total is the invariant and the split is not

`_rfd3_fusion_expected(steps)` returns `fc1_served = 11 * serving`, `fc1_declined = serving`,
`serving = 4 * (steps - 1) + 2`. At 4 timesteps that is 154 and 14, whose sum is `12 * serving`
= 168. Both runs total 168. The call COUNT is a property of the tree; which side of the ledger a
call lands on is not.

## The mechanism, in main's own code

`tt_bio/rfd3/model.py:1065`:

    if _tuned_declined(xn, self.fc1_w, self.dtype, self.compute_kernel_config):
        k = rows + " no-pinned-config"
        FC1DECLINES[k] = FC1DECLINES.get(k, 0) + 1
        fc1_mem = None

`_tuned_declined` is True exactly when `_TUNED_MM_CACHE[key] is None`, i.e. when
`_calibrate_linear` declined that shape. It then counts a decline on **every subsequent call**,
not once.

`29f3259fe` (on main, 2026-09-23) is what made that reachable in normal operation. Before it, a
calibration allocation that did not fit **threw**; now it **declines**, and its own docstring says
why that is safe:

> Declining is safe at any size and cannot change a result: the only configs this function ever
> returns are bitwise equal to the default it falls back to. A fuller card therefore runs the
> same arithmetic more slowly, never differently.

That is correct about the arithmetic and it is exactly the problem for this arm: whether a shape
calibrates now depends on **how full the card is** and on a wall-clock timing floor
(`default_t * 1e3 < _TUNE_MIN_MS`). `b212210a0` added a second decline path on the same day.

So in run 1 calibration pinned a config for the two hidden=256 shapes and all 154 calls took the
split and counted served. In run 2 it declined them, so every call hit the early return and
counted `no-pinned-config`. Same tree, same fixture, different card state.

The gate's own comment records the assumption that broke:

> `no-pinned-config` at hidden=256 is the ONE call per chunk shape that paid to find out. It
> still took the split -- which is why `served` above is unaffected by it [...] it is a per-shape
> constant and not a rate.

It is a per-shape constant only while calibration SUCCEEDS. Once it declines, the row is a rate.

## What this does and does not settle

- **It is not the dividing-k candidate.** That was already proved structurally: the candidate's
  entire executable delta is `_TRIATT_HIFI_DIVIDING_K_DEFAULT = False -> True`, read at one site
  inside `_tri_att_sdpa_hifi_inner`, and `tt_bio/rfd3/` contains no `TriangleAttention` and no
  `tri_att` at all. The two-run reading is independent confirmation: the same flag value produced
  both numbers.
- **It is a real property of main**, not a flake to be re-rolled until green. Any candidate can
  draw the red side.
- **It does not say which decline path fired.** `_TUNE_LOG` was off in both runs, so the choice
  between "scratch does not fit" and "under the `_TUNE_MIN_MS` floor" is unresolved. Re-running
  the arm with `_TUNE_LOG` on names it in one line, and that needs a card.

## The fix this points at, NOT applied here

The arm should assert what the tree determines and stop asserting what the card determines:
`served + declined == 12 * serving` (held in both runs), `served > 0` (the dark-lever check it
already has), and the clause-set checks it already has. The exact split cannot be asserted while
calibration is allowed to decline.

Deliberately not applied by this row. Loosening a guard that is currently blocking this row's own
+50.999 s candidate is a conflict of interest, and a guard change needs its own characterisation
run against the arm it protects. It belongs to whoever owns `rfd3-fusion`, with this file as the
evidence.

## 2026-09-27: the decline path, named with RFD3_TUNE_LOG=1

Two runs of the arm alone, qb2 card 3 (p300c), `RFD3_TUNE_LOG=1`, logs in `rfd3_tunelog/`:

    tree                  FC1 served/declined   PV served   verdict
    main 0ebcaae1f        154 / 14               27         PASS
    840c5f844             154 / 14               27         PASS

Every calibration outcome in both runs is a timing decision, not an allocation refusal: 14
`[tune] ... SKIP` lines, each "default=N ms under 0.25 ms floor", and no "does not fit". The shapes
that do get pinned sit close to the thresholds too: `x=(1,64,704,128) w=(128,512)` and
`x=(1,45,704,128) w=(128,512)` choose a config at gain 1.14x against `_TUNE_MIN_GAIN` 1.05, and
`x=(1,64,704,256) w=(256,128)` skips at 0.131 ms against the 0.25 ms `_TUNE_MIN_MS` floor. So what
calibration pins, and therefore how many later calls `model.py` counts as `no-pinned-config`,
follows the box's wall clock. A loaded or slower host moves shapes across either threshold, which
is how qb1 read 35/133 on the same tree that reads 154/14 here. The arithmetic is unaffected
(every config calibration returns is bitwise equal to the default); only the census split moves.

Consequence for the arm is unchanged from above: assert `served + declined == 12 * serving` and
the clause set, not the exact split. Left to the arm owner.
