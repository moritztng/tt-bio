# PREDICTIONS — protenix-trunk--z-survival-512

Registered before any fold arm ran. Committed first, on purpose: a prediction written down before the
number is what catches a wrong mental model, and a post-hoc explanation cannot be wrong (charter §2,
WARROOM 2.8). Everything below is falsifiable and names what would falsify it.

The no-fold probe (`surv_envelope.py`, results `surv_envelope_qb2c0.json`) had already run when these
were written, so its numbers are INPUTS here, not predictions. What is predicted is what the in-fold
arms will read. Card qb2 chip 0, ttnn 0.68.0, 11x10 grid, 110 cores, co-tenant on chip 1 (board 007).

## The inputs the predictions are built on, measured on this card this pass

L1 fit budget = `get_max_worker_l1_unreserved_size()` x 110 cores = **168 565 760 B (160.75 MiB)**.
The production helper `_l1_memory_config_if_it_fits` evaluated at every padded size:

| flag class | headroom | last padded N on L1 | first padded N on DRAM |
|---|---:|---:|---:|
| L1 `layer_norm` source, c_z=256 (`_PAIR_BIAS_L1_NORM`, `_PWA_L1_NORM`, `_TEMPLATE_L1_NORM`) | 1.5x | **448** | **480** |
| pair transpose, c_z=256 (C2FIX, not this leg's) | 2.5x | 352 | 384 |
| pair transpose, c=64 (C2FIX template track) | 2.5x | **704** | 736 |
| `_pair_proj_config(out_l1=True)`, c_z=256 (`_PAIR_PROJ_L1_OUT`) | static budget | 352 | 384 |
| `_pair_proj_config(out_l1=True)`, c=64 | static budget | **>800** | — |
| `_pair_proj_config(bw_cap=1)`, c_z=256 and c=64 (`_NARROW_PROJ_BW`) | — | **>800** | — |

Isolated price of the two live mechanisms, `[1,N,N,256] @ [256,8]`, median of 5 synced reps:

| form | padded 320 | padded 512 |
|---|---:|---:|
| production (tuned config, DRAM source) | 0.4016 ms | **0.9032 ms** |
| L1 source (what the norm flags hand it when the fit test passes) | 0.1209 ms | **0.2516 ms** |
| `core_grid=` baseline (`_NARROW_PROJ_BW = None`, the pre-X2 path) | 0.5081 ms | **1.4136 ms** |

Roofs, this card, this pass: `[1,512,512,256]` clone 380.6 GB/s to DRAM, 756.8 GB/s to L1 (1.988x);
`[1,512,512,64]` 355.9 / 654.7 (1.84x); 4096^3 bf16 matmul at production fidelity 111.48 TFLOP/s;
machine balance **292.9 FLOP/byte**.

## S1-S3 — the three L1 `layer_norm` flags are worth EXACTLY ZERO at 512 aa, by construction

Padded 512 needs 1.5 x 134 217 728 = 201 326 592 B against a 168 565 760 B budget, so
`_l1_layer_norm` refuses and falls to `ttnn.layer_norm(memory_config=DRAM)` — which is the same op the
OFF arm runs. Both arms emit byte-identical device work. The delta is not "small", it is identically
zero, and no timing argument is needed to establish it.

**Predicted: 0.0 ms/fold each. The census shows 0 of N calls on the L1 branch at every one of the
three sites.**

Falsified by any census row at `pairbias` / `pwa` / `template` reading branch `L1`, or by a
`norm|<site>|c256` or `lin|<site>|c256@*` delta larger than that key's own A/A spread.

## S4 — `_PAIR_PROJ_L1_OUT` reproduces the sibling leg's +29.7 ms/fold

`z-h5-infold` measured this flag alone, in-fold, doubled and interleaved: **+29.7 ms/fold** on
`body:TriangleMultiplication|c64` over 160 counted regions (A/A floor 8.7 % of the delta), +37.8 on
the widest wall including the residual `add_`, and **zero on the pair track** because
`_pair_proj_config(out_l1=True)` returns `None` at c_z=256 from padded 384 up (confirmed above).

**Predicted: +29.7 ms/fold ± (my own A/A spread + 2.6 ms) on the same wall, and zero at c_z=256.**
This is the cross-check on my instrument before the other four flags are trusted. Falsified if the
two disagree outside that band — in which case nothing else in this leg is quotable until the
disagreement is explained.

## S5 — `_NARROW_PROJ_BW` is the only one of the five that does not evaporate, and it is the biggest

Its mechanism is not capacity. It is a program config: `per_core_N=1` and a one-tile-wide output leave
`ttnn.linear(core_grid=)` on a flat core ladder engaging ~16 of 110 cores, and the tuned config takes
the whole grid. Nothing in that depends on the tensor fitting anywhere, and the sweep above confirms
the config is returned at every size to padded 800.

Isolated, the flag is worth 1.4136 − 0.9032 = **0.5104 ms/call at padded 512** against
0.5081 − 0.4016 = 0.1065 at padded 320 — a **4.8x** per-call growth against a 2.56x growth in bytes.

**Predicted: +0.35 to +0.55 ms/call in-fold at 512 aa** (the sibling leg's in-fold-to-isolated ratio
on a comparable region was 1.04x, so the in-fold figure should not be far below the isolated one),
**times the counted number of c_z=256 narrow calls**. If that count is near the sibling's 1048, this
flag alone is **370-580 ms/fold at 512 aa** — larger than everything else in the family put together,
and entirely absent from `size512-ab`'s A/B, which held it fixed in both arms.

Falsified below 0.10 ms/call or above 0.80 ms/call.

## S6 — C2FIX survives at 512 aa only on the template track, at ~93 ms/fold

`_transpose_memory_config` takes DRAM at c_z=256 from padded 384 and stays on L1 at c=64 to padded
704. So its 1010.9 ms/fold at 298 aa collapses to the 160 c=64 transposes. `size512-ab` measured those
at 0.64 ms/call isolated and 0.58 ms/call in-fold.

**Predicted: +80 to +120 ms/fold.** This is `z-rowblock`'s op. One bracketed arm, reported and handed
over; no further work on it in this leg.

## S7 — the headline: the 27.6 % survival figure falls to 6-9 %

`size512-ab`'s A/B moved C2FIX + `_PAIR_PROJ_L1_OUT` + the three norms (read off
`perf/size512/fold_ab512.py`, which states `_PAIR_PROJ_BW` and `_NARROW_PROJ_BW` are identical in both
arms). Summing the arms above, that set is worth **93 + 37.8 + 0 + 0 + 0 = ~131 ms/fold** at 512 aa,
against the **+476.96 ms** it reported.

**Predicted: the `off:ab5` arm reads 110-160 ms/fold, so 476.96 was 3-4x too large and the survival
fraction of that set is 131 / 1729.03 = 6-9 %, not 27.6 %.** Falsified above 300 ms/fold or below 40.

## S8 — the family total is the sum of its singles, and is dominated by the flag nobody measured

**Predicted: `off:family` (the five flags of the merged 685.1) equals the sum of its five singles
within 2x the A/A spread of that wall, and 90 %+ of it is `_NARROW_PROJ_BW`.** If the singles do not
add, the flags interact and that interaction is the finding — the two candidate couplings are
`_NARROW_PROJ_BW` x the norm flags (turning the cap off also removes the L1-output route inside
`_narrow_proj_linear`, which is dead at 512 aa but live at 298 aa) and `_PAIR_PROJ_L1_OUT` x C2FIX,
which touch the same trimul region.

## S9 — the block wall cannot resolve this and the site walls can

`z-h5-infold`'s doubled arms measured an A/A spread of **64.9 ms on `block:PairformerLayer`** at 512
aa, where `size512-ab` had quoted 2.10 ms from a single pair of ON arms and called its 476.96 "227x
the floor".

**Predicted: with >=6 ON arms the `block:PairformerLayer` spread comes back above 30 ms, so the block
wall cannot resolve S1-S4 or S6, while the per-site walls resolve all of them.** Falsified by a block
spread under 20 ms over six ON arms — in which case the block wall is the better instrument after all
and the site walls are a redundancy rather than the answer.

## S10 — where these sites sit, and whether compute overlaps communication

The narrow pair-track projection at `[1,512,512,256] @ [256,8]` moves 150 994 944 B (134.22 MB read,
16.78 MB written at the 32-wide padded output) for 1.074 GFLOP: **7.1 FLOP/byte** against this card's
**292.9 FLOP/byte** machine balance, i.e. **41x onto the memory side**, so a bandwidth roof binds and
there is nothing to argue about.

At 111.48 TFLOP/s the maths take 0.0096 ms against 0.9032 ms measured, so compute is **1.1 % of the
op**: the total is nearer `max(compute, comm)` than `compute + comm`, with `comm` binding, and no
overlap arrangement could be visible at that ratio. Consistent with the sibling leg, which reached the
same verdict from a different direction (its excess tracked bytes and not FLOPs across a 16x FLOP
swing).

**Predicted placements against roofs measured on this card:** production 44 % of the copy roof (DRAM);
the L1-source form 79 % of the copy roof (L1); the `core_grid=` baseline 28 % of the copy roof (DRAM).
Two of the three are under 70 %, so a mechanism is owed at transaction granularity, not at ttnn
argument level: the tuned config runs `in0_block_w=1`, `out_subblock_w=1`, `per_core_N=1`, so the
reader issues one-tile (2 KB) reads per K block and the packer writes one tile per pack, and small
transactions do not reach the bulk rate a clone's long bursts do. **Predicted: the limiter is
transaction size on both sides of the projection, and raising `in0_block_w` (i.e.
`_NARROW_PROJ_BW > 1`) closes part of the gap at the cost of bit-exactness** — which the org has
already decided separately and is not re-opened here.
