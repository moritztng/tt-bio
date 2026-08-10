# D2 (atom-window `_bmm`) vs main's `batched_matmul` — predictions, written before the device was opened

Leg `perfwar-atomwindow-batchedmm-reconcile`, qb1 card 0, ttnn 0.67.4, base `cc39a867`.

## What the two things actually are

Main already routes all four atom-window call sites through `tt_bio.tenstorrent.batched_matmul`
(`tt_bio/protenix.py:431/434/515/529`, merged by E7 `373038e2`). D2's branch
`wk/perfwar-atom-window-attention` never merged; its `AtomTransformer._bmm` rewrites the same four
lines. So this is not "naive vs D2" — it is "main's helper vs D2's helper", and the naive matmul is
only the shared denominator both quote against.

Production shapes, protenix-v2/opendde `AtomTransformer` (`N_HEADS=4`, `HEAD_DIM=32`, `nq=32`,
`nk=128`), at 298 aa `nb=75`:

| op | in0 | in1 | batch | Mt | Kt | Nt |
|---|---|---|---|---|---|---|
| QK^T | `[nb,4,32,32]` | `[nb,4,32,128]` | 4·nb | 1 | 1 | 4 |
| A@V  | `[nb,4,32,128]` | `[nb,4,128,32]` | 4·nb | 1 | 4 | 1 |

## Prediction 1 — the two helpers pick the SAME per-core config, and differ only in launch structure

`_batched_matmul_block_w` returns 1 for QK^T (`Kt=1` is odd) and 2 for A@V (`Kt=4` even, and
`min(height,width) <= 32` makes it narrow). Those are exactly the widths D2 hard-coded and exactly
the two D2 proved are the bit-exact ones. `_batched_matmul_search` with `m_tiles=1` can only pick
`per_core_M=1`, and `per_core_N` is forced to `Nt`. So both helpers issue
`MatmulMultiCoreReuseProgramConfig(in0_block_w=1|2, per_core_M=1, per_core_N=4|1)` on the same
11x10 grid. Two differences remain:

- `out_subblock_w`: main takes the widest legal (4 on QK^T), D2 hard-codes 1.
- launch structure: main issues **one** program with 300 blocks over 110 cores (legal because
  `per_core_M == m_tiles`, so one block is one batch element and the kernel's batch stride is
  correct); D2 issues **three** programs of 100 blocks each, built with `ttnn.slice` on both
  operands and joined with `ttnn.concat`.

**Predicted: main >= D2 at nb=75.** Both take 3 rounds of blocks through the grid
(ceil(300/110)=3 vs 3x1), and main pays no slice/concat. D2's own OWED note prices that movement at
12.3 MB per call against 11 MB of useful traffic, so the predicted margin is large: **main 1.3-2.2x
faster than D2's `_bmm`**, and D2 SUBSUMED at 298 aa.

Falsifier: if 3 blocks per core in one program serialises worse than 3 separate one-block-per-core
programs (per-core CB reserve/push not overlapping across blocks, or in1 re-read per block), D2 wins
despite the extra movement.

## Prediction 2 — the interesting gap is at SMALL nb, and it is a gate, not a chunking strategy

`_batched_matmul_search` opens with `if batch < 2 or batch * m_tiles < cores: return None`. With
`m_tiles=1` that is `4·nb < 110`, i.e. **main declines the config entirely for nb <= 27** and falls
back to the naive matmul, which D2 measured at 4 of 110 cores. D2's `_bmm` has no such gate: at
small nb it takes the `g == 1` path, one launch, no slicing, `nb·4` cores busy.

That gate is a saturation heuristic, not a correctness rule. The correctness escape here is
`p == m_tiles`, which holds for every nb because `m_tiles == 1`. So for nb <= 27 the config is legal
and unused.

Production sizes this covers: `nb = ceil(atoms/32)`, so 298 aa (2398 atoms) is nb=75 and clears the
gate, 117 aa `prot.yaml` (900 atoms) is nb=29 and *just* clears it (116 blocks over 110 cores, a
2-round tail with 104 cores idle in round 2), and ubq (~76 aa, ~600 atoms) is nb=19 and **does not
clear it at all**.

**Predicted: at nb=19 the naive path is ~1 us per window-head (D2's flat rate) and a single-launch
config with 76 of 110 cores busy is 3-5x faster, and main currently leaves all of it on the table.**
If that holds, the right change is to relax the gate in `_batched_matmul_search` — one shared
helper, no chunking, no second symbol — not to merge D2.

Falsifier: at 76 blocks the per-launch fixed cost (program dispatch, CB setup on 76 cores) may
dominate a 76 us naive call, in which case the gate is right and there is nothing to fix.

## Prediction 3 — bit-exactness

Both helpers should be `torch.equal` against the naive matmul on all four classes. Main's four
classes are already pinned in `tests/test_batched_matmul_hw.py:32-37` (fp32 and bf16) plus the
rank-5 openfold3 forms at lines 49-50. `out_subblock_w` changes how many output tiles sit in DEST,
not the K accumulation order, so E7's "`in0_block_w` alone decides bit-exactness" predicts main's
`osw=4` on QK^T is still exact. If it is not, that is a live bug in main, not a D2 finding.

## Method

`perf/atomwindow_reconcile/micro.py`. One process, one card, arms interleaved and rotated per trial
so a host load ramp cannot map onto arm order (L2 pass 2 §5 lost a whole measurement to that).
`ttnn.synchronize_device` on both sides of every timed region. Median of per-arm medians over 7
trials of 6 reps, 3 warm calls first. Denominator for every ratio is the naive `ttnn.matmul` on the
same operands in the same process.
