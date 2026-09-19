# Pre-registration: region T composed onto current main, with cond-hoist shipping

Written and committed at 23:3xZ, minutes after the session started at 23:33:18Z and well before
any block completed. Session: `perf/c14_bfp8/regiont_compose_ab.json`, `--blocks 5 --folds 3`,
qb2 node 3 with node 2 idle, benchlock, `TT_BIO_AICLK=1350`, per-arm quiet guard on.

## Why this session exists and why the earlier +0.1750 s cannot be the landing number

`wk/c14-regiont-on-main` @ `d3427db84` reads `_B2_DIT_COND_HOIST = env_flag(..., False)`. It is a
pre-hoist tree, and `TT_BIO_DIT_COND_HOIST` has been the default on `origin/main` since
`15f1ab0ac`. So +0.1750 s is region T's value against a base that no longer exists. Stack
perturbations are strongly sub-additive and this campaign approves a stack as a stack or not at
all, so the landing number is region T against current main **with cond-hoist on**.

Merge: `origin/main@67f1ecaa8` into the branch at `37b988dd5`, no conflicts, and the production
delta against main is unchanged at 131+/39- across the same five files, so the merge added no
production delta of its own. `_B2_DIT_COND_HOIST` now reads True at `tenstorrent.py:1481` and
`_TRIATT_B8` False at `:1564` in both arms' tree.

## Registered predictions

Scoring rule, unchanged from the session it repeats: **block 0 is discarded unconditionally** as a
session warm-up, and the A/A floor comes from the remaining base blocks.

1. **base median 14.35-14.50 s.** The pre-hoist base measured 14.630 s in the same harness and
   cond-hoist is worth +0.2052 s, so a post-hoist base should sit near 14.43 s. This arm IS
   current main by construction — the branch's only delta is region T and the base arm has it off
   — so it is also a free check on main's own fold time.
2. **on median 14.18-14.35 s.**
3. **delta +0.10 to +0.20 s, ratio 1.007-1.015.** If region T's absolute saving survives the hoist
   unchanged it lands near +0.175 s and 1.0122; sub-additivity can only shrink it. A reading above
   1.018 would mean the two levers are super-additive, which nothing in the mechanism predicts and
   which I would treat as an instrument problem before believing it.
4. **A/A floor under 1.006**, as in both previous quiet sessions (1.00335 and 1.00432).
5. **The two arms' digests differ from each other**, and the base arm's digest is whatever current
   main computes. Whether either matches the pre-hoist pair (`45781db716ebf020` /
   `791696c3e443676d`) is a fact about cond-hoist's bit-exactness that this session reports for
   free; it is not predicted here.
6. **No foreign device holder in any scored block.** Moritz's TRIX campaign took qb2's other three
   chips at the 23:26Z tick and `trix-transaction` holds card 2, my board-pair sibling. Both guards
   read clean at 23:32:37Z (loadavg 0.05, no fd on either chip) and the per-arm guard re-checks
   before every arm, but if TRIX starts folding mid-session this check is what catches it.

## What makes this negative

The on median above the base median in two or more admissible blocks, or a ratio inside the
session's own A/A floor. Either outcome means region T does not survive composition with
cond-hoist and must not be landed on the +0.1750 s reading.

## The arithmetic nobody may do

**0.2052 + 0.1750 is not a number.** Whatever this session returns is region T's value on shipping
main, and the stack's value is the composed fold time against a pre-both baseline, not a sum.
