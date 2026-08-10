# P5 — `protenix-trunk--p2-msa-structural`, Phase 2: C1, C3, C4, C5

Protenix-v2, trunk only, 298 aa. qb2 card 1 (board 007 chip 1), ttnn 0.68.0, so **ratios, not
campaign absolutes**. T5's numbers are pc card 0 — a different card and a different chip — so every
baseline I A/B against is re-measured here this pass and said to be.

**Probes only. No production change**: everything lives under `perf/p2_msa_structural/` on
`wk/protenix-trunk--p2-msa-structural` and nothing under `tt_bio/` is touched.

**Site line numbers moved.** T5 worked a branch where the OPM tail sat at 3034-3037/3043 and the
producing matmul at 3012. On `origin/main` (`bbb5d85b`) the same chain is
`to_layout:3059`, `reshape:3060`, `to_layout:3061`, `permute:3062`, consumer `linear:3068`,
producer `matmul:3037`. Same ops, +25 lines. I use this branch's numbers below and give T5's in
brackets the first time.

---

## Predictions (before measuring)

Committed and pushed before the device was opened
(`perf/p2_msa_structural/PREDICTIONS.md`, commit on this branch). Every one names the number, the
unit, and what would count as having been wrong.

**Roofs, qb2 card 1.** DRAM read at 128 MB **380 ± 40 GB/s**, DRAM write **255 ± 35 GB/s**,
DRAM→DRAM copy **400 ± 50 GB/s**. Wrong if any lands outside its band; a write roof under 200 or
over 300 GB/s would mean I am measuring something other than the writer.

**C1 — the OPM layout chain.**

- **P1.1** The (i,c)x(d,j) → (i,j,cd) reindexing **cannot** be emitted by any matmul that keeps
  a on the M side and b on the N side, because M carries {i,c} and N carries {d,j} while the
  consumer needs {i,j} together. So the *interesting* arm is not "make the matmul emit it" but
  "make the matmul emit a layout the residual relayout is cheap from". I predict the b-side
  reorder arm (build `b_flat` as `(j*D+d, S)` instead of `(d*J+j, S)`) leaves the **matmul's own
  time unchanged within ±5 %** — identical operand shapes, identical K, only the tiny 0.68 MB b
  operand is permuted differently. Wrong if it moves more than 5 %.
- **P1.2** The batched-over-i arm (`a` as `(298, 32, 35)` @ `(35, 9536)`, output `(298, 32, 9536)`)
  is **at least 3x slower** than the flat `(9536, 35) @ (35, 9536)`, because M becomes one tile row
  per batch and ttnn does not distribute the batch dimension. Wrong if it lands within 1.5x.
- **P1.3** The chain is 4 passes over a 182-195 MB intermediate. A **single**-op relayout would move
  182 MB read + 195 MB write = 377 MB, which at a 400 GB/s copy roof is **~940 us/call = 37.6
  ms/fold at x40 OPM calls**. So if any single op can do the reindex, the prize is
  **179.2 - 37.6 ≈ 140 ms/fold**, not 179.2. Wrong if the measured 4-op chain on this card is not
  within 25 % of 4x the single-pass figure.
- **P1.4** Kill test as the brief states it: if the matmul's own time rises by more than **179
  ms/fold (4479 us/call at x40)** under any arm, the relayout is intrinsic and has only moved. I
  predict **no arm trips this** — the b-side reorder is free and the batched arm's penalty, while
  large in ratio, is bounded by the matmul's own 1.3 ms/call, i.e. it cannot cost 4.5 ms/call.

**C3 — the PWA per-head weight slices.**

- **P3.1** us/call is **flat within ±30 %** from a 16 kB slice to a 1.6 MB slice off the same
  tensor. Wrong if it rises roughly linearly (a 100x size step producing more than a 3x time step
  kills the fixed-cost reading and with it the 51.5 ms/fold).
- **P3.2** The fixed cost will be **~40-70 us**, which is **6-11x T4's measured 6.40 us launch
  floor**. So "essentially all of the time is dispatch" does **not** follow, and I name the third
  thing in advance: **sub-tile slicing**. `self.z_weight[:, i:i+1]` takes one logical column out of
  a 32-wide tile face, so the reader must fetch whole tiles and the packer must rebuild a padded
  face. Prediction: a **tile-aligned 32-column** slice of the same output byte count is **at least
  3x faster** than the 1-column slice. Wrong if aligned and unaligned come out within 1.3x — then
  the cost is host-side op overhead, not the tile face.

**C4 — the transition launch floor.**

- **P4.1** `linear@2071` [T5's 2046] and `layer_norm@2063` [2038] are **flat within ±25 % from 1x to
  2x rows** and rise clearly at 4x. Wrong if 2x rows costs 1.8-2.2x the time — that is bandwidth
  and the candidate dies.
- **P4.2** `minimal_matmul@1701` costs **within ±15 % of `@1695`** despite writing a quarter of the
  bytes, on this card as on pc. Wrong if the ratio is near 4x (i.e. byte-driven).

**C5 — the loop-invariant template z projection.**

- **P5.1** Hoisting `self._lin(zn, "…linear_no_bias_z.weight")` out of `for t in range(nt)` removes
  **30 of 40 calls/fold**. On pc that was 14.4 ms/fold. On qb2 card 1 I predict **10-20 ms/fold**,
  and a **`trunk_template` stage wall drop of 1.4-2.8 %** (the stage is ~72 ms/call). Wrong if the
  stage wall does not move by at least 1.0 %, which would mean the call was already overlapped with
  something.
- **P5.2** `torch.equal` on the `_template` stage output, baseline vs hoisted, returns **True**.
  Wrong if it returns False — the operand and weight are both loop-invariant, so any difference is
  a bug in my hoist, not a numerical property.
- **P5.3** C5 and `protenix-trunk--p2-matmul-ceiling`'s program-config fix to the same site are
  **not additive**. If the config fix makes the call k us and the hoist removes 30 of 40 calls,
  the combined figure is `10 x k x 40/1000` ms/fold of remaining cost, not `179 - C5 - C2`.

---
