# protenix-trunk--z-narrowbw-512 — predictions, registered before any arm runs

Committed and pushed before the first fold arm opens a device. Charter §2 Phase 2: a prediction
written down before the number is what catches a wrong mental model; a post-hoc explanation cannot be
wrong. Every entry has a falsifier that does not depend on the outcome.

Scope: **protenix**-v2, the **trunk**. Card qb2 chip 0, ttnn 0.68.0, 11x10 grid, 110 cores. Every
absolute is a ratio owing a qb1/0.67.4 re-take (charter §4.8); a **ratio between caps on one card** is
what this card is good for and is what most of these predictions are.

The subject is `_NARROW_PROJ_BW` (`tt_bio/tenstorrent.py:106`), the `in0_block_w` cap for the three
NARROW-output pair-track projections. It ships at **1**, which is the production contraction order and
`torch.equal`; above 1 the partials fold through `packer_l1_acc` in K-block order and it is not
bit-exact. `z-survival-512` priced the constant at **407.89 ms/fold at 512 aa against 60.37 at
298 aa, 6.76x**, over **764 counted calls** (484 pairbias @ n=16, 240 pwa @ n=1, 40 template @ n=64).

---

## N0 — settled card-free in the planning pass, and it halves the sweep

**`cap = 16` and `cap = 8` are the same program config at every narrow site, at both sizes.** The
helper picks `in0_block_w = max(d in (k_tiles, 8, 4, 2, 1) : d <= cap and k_tiles % d == 0)` and
`k_tiles = 8` at `c_z = 256` at all three sites, so cap 8 and cap 16 both select 8. Enumerated over
`_pair_proj_config` directly (`perf/narrowbw/cfg_ladder.txt`): identical `in0_block_w`, `per_core_M`,
`out_block_h`, `out_block_w`, `out_subblock_h`, `out_subblock_w`, `per_core_N` in every one of the six
site x size combinations.

Consequences the exec pass must not re-derive:

- the constant's own comment reports "1.98x / 2.08x at 16" — that is a measurement of
  **`in0_block_w = 8`**. There is no cap 16 in this model, only an alias for 8.
- the five-cap sweep has **four distinct arms**, and `bw:16` is a **free A/A control** on `bw:8`.
- **no cap is capacity-gated.** The program config's L1 budget needs 186 368 B per bank at bw=1 and
  421 888 B at bw=8 against `_l1_bank_bytes() = 1 461 760`, i.e. **0.13x to 0.29x of one bank**, so
  the helper returns a config at every cap and no cap can silently fall back to `core_grid=`.
- **core utilisation is identical at every cap**: `per_core_M = 75` and **110 of 110 cores** at
  512 aa, `per_core_M = 30` and **100 of 110** at 298 aa, independent of the cap. Occupancy therefore
  cannot explain any part of the ladder, and this leg must not lead with it.

**Falsifier (the arms still check it):** any census row where `bw:16` and `bw:8` report different
`in0_block_w`, a plDDT or CIF sha differing between those two arms, or a site-wall delta between them
larger than that key's A/A spread.

## N1 — the speed ladder is monotone in the cap and saturates at bw = 8

Two things move with `in0_block_w` and both improve monotonically. The in0 NOC read transaction per K
block is `in0_block_w x 2 KB` → **2 / 4 / 8 / 16 KB** at bw 1/2/4/8, against the long bursts a clone
gets. And the number of `packer_l1_acc` partial folds per output tile is `k_tiles / in0_block_w` →
**8 / 4 / 2 / 1**, each one paying an in1 multicast barrier, a DEST clear and a packer pass; at bw = 8
the whole K = 256 contraction happens in one block and there is no partial fold at all. Nothing on the
write side moves: `out_subblock_h = 1` and `out_subblock_w = 1` (2 at template) at every cap, so the
packer still emits one tile per pack.

**Prediction, 512 aa, speedup against cap 1 at the `pairbias`/`pwa` shape:**

| cap | in0_block_w | in0 read transaction | partial folds / out tile | predicted speedup vs cap 1 | band |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 2 KB | 8 | 1.00 | — |
| 2 | 2 | 4 KB | 4 | **1.37x** | 1.20–1.55 |
| 4 | 4 | 8 KB | 2 | **1.64x** | 1.45–1.80 |
| 8 | 8 | 16 KB | 1 | **1.88x** | 1.65–2.00 |
| 16 | 8 | 16 KB | 1 | = cap 8 | inside the A/A floor |

**Falsifier:** non-monotone beyond the A/A floor, or cap 4 delivering under 70 % of cap 8's gain
(which would make 4 the wrong recommendation), or cap 8 outside 1.65–2.00x.

## N2 — cap 8 is worth 250–400 ms/fold at 512 aa, central 325, and 90–170 at 298 aa, central 126

From `z-survival-512`'s measured `on`-arm site walls at 512 aa — `lin pairbias c256@16` 440.46 ms,
`lin pwa c256@1` 215.95, `lin template c256@64` 38.01, **694.42 ms/fold over 764 counted calls** — and
N1's 1.88x: `694.42 x (1 - 1/1.88) = 325.1 ms/fold`. At 298 aa the constant's own comment gives
cap16/cap1 = 1.98/1.15 = 1.72 and 2.08/1.23 = 1.69 on qb1, so 1.70, against an isolated on-arm
`764 x 0.4016 = 306.8 ms/fold`: `306.8 x (1 - 1/1.70) = 126.4 ms/fold`.

Per cap, 512 aa: **cap 2 = 187 (120–260), cap 4 = 271 (200–350), cap 8 = 325 (250–400)**.

The trade therefore scales **2.6x from 298 aa to 512 aa**, against a **2.56x** growth in bytes.

**Falsifier:** any cap outside its band, or a 512/298 scaling outside 2.0–3.2x.

## N3 — the binding roof is the bandwidth roof at every cap, and cap 8 reaches 76–90 % of it

Bytes per call at 512 aa, `pairbias`/`pwa`: **134 217 728 B read + 16 777 216 B written** (the output
is one tile wide, so 32 padded columns) = **150 994 944 B**. FLOPs at the padded output width the
hardware actually computes: `2 x 512 x 512 x 256 x 32 = 4.295 GFLOP`. **Arithmetic intensity
28.4 FLOP/byte** against the machine balance this card measures (292.9 FLOP/byte, sibling's figure,
re-measured this pass) — **10.3x onto the memory side**, so the binding roof is the bandwidth roof at
every cap and the op is **memory-bound** with nothing to argue about. Using the most generous compute
roof available (a square 4096³ matmul at production fidelity, 111.48 TFLOP/s) makes that placement
conservative rather than optimistic, which is the point of charter §4.6.

At the measured DRAM copy roof of 380.6 GB/s (bidirectional bytes) the floor for this call is
**0.3967 ms**. Cap 1 measures 0.9032 ms = **43.9 % of the copy roof (DRAM)**. Predicted cap 8:
**0.48 ms, 82.6 % of the copy roof (DRAM)**, band 76–90 %.

**Compute and communication overlap.** Compute is `4.295 / 111.48e3 = 0.0385 ms`, **4.3 % of the
0.9032 ms cap-1 op** and ~8 % of the predicted cap-8 op, so the total is nearer **max(compute, comm)**
than `compute + comm` at every cap, with `comm` binding. What the cap changes is not the compute time
but how many times the reader stalls: at bw = 1 each output tile pays 8 in1 multicast barriers and 8
packer passes interleaved with 2 KB reads, and the circular-buffer depth bounds how many of those are
in flight. **Prediction: cap 8's measured time exceeds the pure-bandwidth floor by more than its
0.04 ms of compute**, i.e. part of the residual is phase count and not bytes.

**Falsifier:** a measured placement outside ±10 points of 82.6 %, or a cap-1 placement outside
39–49 % of the copy roof (DRAM), which would mean the sibling's roof does not reproduce on this card.

## N4 — accuracy does NOT get worse with the cap

The reference is a torch fp32 matmul over the **same bf16 operands**, so this is arithmetic only, with
no diffusion noise and no sampling. `k_tiles / in0_block_w` partials fold through `packer_l1_acc` per
output tile: **8 at bw = 1, exactly 1 at bw = 8**. Fewer fold steps cannot add error, and at bw = 8 the
whole K = 256 contraction accumulates once in an fp32 DEST and is packed to bf16 once. The direction
already has independent support: `_PAIR_PROJ_BW = 16`'s own comment records opendde moving 5.54 Å from
the bw = 1 arm but **toward** its reference on every metric.

**Prediction:** `max_abs_vs_fp32` and `rms_vs_fp32` are **non-increasing** in the cap over
{1, 2, 4, 8}, at all three site shapes and both sizes. cap 1 is `torch.equal` to the `core_grid=`
reference (the comment's claim, never actually measured). cap 8 and cap 16 are `torch.equal` to each
other. No other pair is `torch.equal`.

**Falsifier:** any cap whose `max_abs_vs_fp32` **exceeds** cap 1's. The device is deterministic and
both arms read the same operands, so there is no noise floor to clear and any excess is real. If this
is falsified, the trade is a real trade and the ranked table is a genuine parity decision; if it holds,
the only thing cap 8 costs is the *label* `torch.equal`, and that is the finding.

## N5 — on `protenix-hsa-msa`, cap 8 lands at or below 0.45 of the envelope bound

`scripts/full_parity_gate.py` in **DEFAULT** mode (never `--legacy-rdx`, never `--seeds`), scored by
`scripts/integration_envelope.py`: `bound = envelope x (1 + 0.50) + 0.05`, fraction of bound =
`num / bound`. `protenix-trunk--y-envelope-audit` established that of the three protenix legs only
**`protenix-hsa-msa` (585 aa) is well-conditioned** — samples 6.99 Å apart, `identification_safe: true`,
numerator 0.055 Å — and that `prot` (116 aa) and `ubq` (76 aa) carry a pre-existing accepted GAP 10x
and 22x the bound with `identification_safe: false`, so they cannot score a protenix trunk change at
all. On `hsa` main today reads **0.4245 of bound** with a bit-exact pair-cap control at **0.4025**.

**Prediction:** the cap 1 control arm reproduces **0.42 ± 0.03** of bound on `hsa` (a different card
and a different ttnn minor, so a ratio and not a match); cap 2, 4 and 8 each land **at or below 0.45**;
every leg returns the **same verdict in every arm** (PASS on `hsa`, GAP-reproducing on `prot`/`ubq`);
no metric on any leg crosses from inside the bound to outside it.

**Falsifier:** any cap above 0.50 of bound on `hsa`, or a verdict flip on any leg, or a control arm
outside 0.39–0.45 (which would mean the instrument does not reproduce and nothing after it is
quotable).

## N6 — the three caps are independent at 512 aa

`_NARROW_PROJ_BW`, `_PAIR_PROJ_BW` and `_PAIR_PROJ_L1_BW` are three separate `bw_cap` arguments to one
helper, and `z-survival-512`'s census shows their call sites are disjoint: `_narrow_proj_linear` serves
`pairbias` / `pwa` / `template` only, `_pair_proj_linear` serves `trimul` / `triatt` only. The shared
`_pair_proj_program_config` lru_cache is keyed on `(m_tiles, k_tiles, n_tiles, in0_block_w, elem_bytes,
out_l1)`, so no cap can read another's entry. The one channel that could couple them is L1: bw = 8
raises the narrow matmul's static circular-buffer footprint from 186 368 B to 421 888 B per bank
alongside whatever `_PAIR_PROJ_L1_OUT` has resident, which is the `protenix-v2-448aa-l1-cb-clash`
failure class.

**Prediction:** the `bw:8+pairbw1` arm's three narrow site walls match `bw:8`'s within each key's A/A
spread, and no arm throws a circular-buffer clash at either size.

**Falsifier:** a narrow-site difference above the floor, or any arm raising
`statically allocated circular buffers ... clash with L1 buffers`. Either outcome is the finding and
outranks the table.

---

## What is deliberately NOT predicted

- **The decision.** A non-bit-exact trunk change is Moritz's call by charter §4.7 and this leg produces
  the table, not the verdict.
- **`prot` and `ubq` resolving anything.** `y-envelope-audit` settled that they cannot; reporting
  "2 GAP, 1 PASS" and stopping there would be learning nothing from two thirds of the run.
- **Anything about the pair transpose, the permute flip or the crash band.** Other legs' ops.
