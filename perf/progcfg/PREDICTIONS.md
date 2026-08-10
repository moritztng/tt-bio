# protenix-trunk--z-progcfg-h5 — H5: is the L1-output projection's win program config, traffic, or overlap?

TASK TYPE: ACCELERATE (Phase 2 experiment) | PLAYBOOKS loaded: ACCELERATE + VERIFY/BENCHMARK |
memories read: perfwar-programconfig-gate-output-not-subtracted, tt-bio-matmul-dram-write-serialized-l1-residency-fix,
roofline-roof-must-be-measured-not-asserted, ttnn-sync-before-every-timed-region,
tt-bio-l1-residency-guard-dead-in-real-folds, ttnn-batched-matmul-programconfig-rules,
tt-bio-worktree-run-recipe, donecheck-hostspecific-path-unsatisfiable-on-remote-host

Scope: **protenix**-v2, the **trunk**. Card qb2 chip 0, ttnn **0.68.0**, 11x10 grid.
**Every number this leg produces is a ratio owing a qb1/0.67.4 re-measurement before it drives a
decision** (charter §4.8). 0.68.0 removes the unaligned-contraction penalty and regresses SDPA 1.76x,
so a mechanism visible only here is not a lever.

**Phase 2. Probes and prototypes only. No production change.**

---

## STATE OF THIS DOCUMENT

**PLANNING PASS COMPLETE — MEASUREMENT NOT STARTED.** Written by the opus5 planning tier before the
device was opened. Two verdicts below are settled by code reading and arithmetic and are final; the
rest are predictions with the probe that tests each one. Nothing here is a device measurement yet.

| deliverable | state |
|---|---|
| predictions registered before the device opens | **DONE** (this commit) |
| V1 config term is identically zero | **SETTLED by code read** — confirm by config diff in the probe |
| V2 the "11x over bytes" is a mis-specified byte model and a mislabelled denominator | **SETTLED by arithmetic** — confirm by census |
| four-cell result at the fold's own shapes, 298 + 512, c=64 + c_z=256 | probe validated on device at **one** shape (`298:64`), see "first light"; three shapes owed |
| roofs measured on this card this pass | probe written, **not run** (`--skip-roofs` was used to smoke the cells) |
| H5 verdict + the ms/fold hand-off to Phase 3 | blocked on the above |

### First light — `298:64` only, smoke run after the predictions were committed at `e8a6307b`

Run to validate the harness, not to answer the leg. One shape, no roofs, so nothing here is a
deliverable. Two things are worth carrying forward anyway.

**P1 is CONFIRMED on the device.** `cfg_fields_identical: true` at `[1,298,320,64]`: both arms return
`in0_block_w=2, out_subblock_h=1, out_subblock_w=2, out_block_h=5, out_block_w=2, per_core_M=30,
per_core_N=2` on an 11x10 grid, 100 of 110 cores engaged. The code read is confirmed by the device.

| cell | proj ms | region ms | region+consume ms | `torch.equal` vs C1D |
|---|---:|---:|---:|---|
| C1D tuned / DRAM (production OFF) | 0.1055 | 0.2991 | 0.3830 | ref |
| C1L tuned / L1 (production ON) | 0.0709 | 0.2285 | 0.2941 | **True** |
| C0D untuned / DRAM | 0.1009 | 0.2816 | 0.3659 | False, max abs 0.25 |
| C0L untuned / L1 | 0.0729 | 0.2301 | 0.2969 | False, max abs 0.25 |

Clone roof at this shape: 334.4 GB/s to DRAM, 592.8 GB/s to L1. Arithmetic intensity 32.0 FLOP/byte.

**The config term is not merely zero, it is slightly negative at this shape**: the tuned DRAM cell is
0.0175 ms/region *slower* than `core_grid=`, while the destination term is +0.0706 ms/region. The best
DRAM-output config in the sweep reaches 0.1033 ms on the projection against C1L's 0.0709 — **1.46x
short**, which is P6's direction. And a bit-exact L1-output config production never tries
(`bw=2, obh=6, obw=2`, 26.9 % of a bank) reaches 0.0684 against production's 0.0709, a 1.04x the
production path leaves on the table because it pins `out_block_h = 5`.

None of this is the leg's answer: it is one shape, the small one, with no roofs and no 512 aa cell.

---

## The two verdicts that do not need the device

### V1 — H5 as literally posed is KILLED. The program-config term in the measured win is exactly zero.

The win `size512-ab` measured is the `_PAIR_PROJ_L1_OUT` flag flip. Read what that flag actually
switches (`tt_bio/tenstorrent.py:753-790`):

```python
if l1_out and _PAIR_PROJ_L1_OUT:                       # ON arm
    cfg = _pair_proj_config(x, w, bw_cap=_PAIR_PROJ_L1_BW, out_l1=True)
    if cfg is not None:
        return ttnn.linear(x, w, memory_config=ttnn.L1_MEMORY_CONFIG, ..., program_config=cfg)
cfg = _pair_proj_config(x, w)                          # OFF arm
return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, ..., program_config=cfg)
```

`_PAIR_PROJ_L1_BW = 16` and `_PAIR_PROJ_BW = 16` (lines 96 and 123), so both arms enter
`_pair_proj_config` with the same cap and derive the same `in0_block_w`. And `out_l1` does not appear
in any field of the returned `MatmulMultiCoreReuseMultiCast1DProgramConfig` — it appears **only** in
the `need <= _l1_bank_bytes()` budget at line 697. Every field the two arms produce is identical:
`in0_block_w`, `out_subblock_h/w`, `out_block_h=5`, `out_block_w=n_tiles`, `per_core_M`, `per_core_N`.

**So the A/B that produced 2.30 ms/call holds program config fixed and varies only `memory_config`.
Its program-config content is zero by construction, not by measurement.** H5 asked whether the win is
config or traffic; at the site it was measured on, it cannot be config.

That does not make the leg empty — it relocates it. The decision question underneath H5 is worth more
than H5: **can a DRAM-output program config recover the win, so the pair track keeps it past the
capacity cliff?** That is a different experiment from the one H5 names, and it is the one below.

The probe still runs the four cells, because "the config fields are equal" is a claim about the code
and the cells are the claim about the device — including whether the *untuned* `core_grid=` path
behaves the same way, which the code read cannot tell you.

### V2 — the "11x what the traffic justifies" is FALSIFIED. It is ~4x, from two independent errors.

**Error 1, the denominator.** `size512-ab` attributes `TriangleMultiplication`'s +367.4 ms at 512 aa
to "160 c=64 L1-output projections, 2.30 ms per call". 160 is the count of **trimul executions**, not
projections. `TriangleMultiplication.__call__` calls `_trimul_out_proj` twice — `p_out` at line 1751
and `g_out` at line 1754 — and `_trimul_out_proj` is the only `l1_out=True` site in the body. Two
trimuls per block x 80 template-block executions = **160 regions and 320 projections**. The leg's own
in-fold counter agrees: it counted **480** c=64 L1-output projections per fold, and 480 = 320 (TriMul)
+ 160 (TriangleAttention `x_out`, one per tri-attention, 2 x 80).

So 2.30 ms is a per-**region** cost, and the per-projection cost is **1.148 ms**. Charter rule 3.

**Error 2, the byte model.** 0.21 ms/call prices **one** 33.554 MB output: its DRAM write at the
measured 272.9 GB/s plus one consumer read at 399.9 GB/s. A trimul output region moves **six** tensors
of that size across the DRAM boundary in the OFF arm, and the ON arm removes all six:

| # | transfer | direction | bytes |
|---|---|---|---:|
| 1 | `p_out` writes its result | write | T |
| 2 | `g_out` writes its result | write | T |
| 3 | `multiply_` reads `p_out` | read | T |
| 4 | `multiply_` reads `g_out` | read | T |
| 5 | `multiply_` writes in place into `p_out`'s buffer | write | T |
| 6 | the layer's residual `add_` reads that result | read | T |

`T = 512 x 512 x 64 x 2 = 33 554 432 B`; the region moves `6T = 201.33 MB`, split 3T write / 3T read.
Rows 1-4 are certain from `tenstorrent.py:1745-1758`; rows 5-6 are the assumption the probe checks
(if the residual add is outside the flag's reach the region is 5T and the floor drops 17 %).

| model | floor per region |
|---|---:|
| directions serial: `3T/272.9 + 3T/399.9` | **0.621 ms** |
| directions overlapped at the 397.7 GB/s r+w roof | **0.506 ms** |

**Against 2.30 ms/region measured, the over-bytes factor is 3.7x to 4.5x, not 11x.** The corrected
number is still a defect by charter rule 5 and still needs a mechanism. It is not an 11x mystery.

Both errors are corrections to a sibling leg's arithmetic, not to its measurement: the +367.4 ms
itself is a paired body wall against a 2.10 ms A/A floor and is not in doubt.

---

## The contradiction that is worth more than H5, and the measurement that settles it

Apply the same region byte model to the win the org actually banked, at 298 aa on the pair track:

`T256 = 298 x 320 x 256 x 2 = 48 824 320 B`, region = `6T256 = 292.9 MB`, floor at the 397.7 GB/s
combined roof = **0.736 ms/region**. There are `2 x 524 = 1048` c_z=256 trimul regions per fold, so
the byte floor for those regions alone is **771 ms/fold**.

The org's entire X7 line is **561.8 ms/fold**, and that figure also contains the L1 `layer_norm`
source and the tri-attention `x_out` projections. So on the pair track at 298 aa the measured saving
is **below** its own byte floor, while on the template track at 512 aa it is 4x above it.

**One mechanism cannot produce both.** Either the pair-track DRAM traffic is largely hidden behind
compute at 298 aa, or the 512 aa template-track region is paying for something other than its bytes.
The two shapes differ in exactly one relevant way and it is the one the roofline cares about:

| site | K (tiles) | FLOPs | bytes | arithmetic intensity |
|---|---:|---:|---:|---:|
| pair track `[1,N,·,256] @ [256,256]` | 8 | `2·M·256·256` | `4·M·256` | **128 FLOP/byte** |
| template track `[1,N,·,64] @ [64,64]` | 2 | `2·M·64·64` | `4·M·64` | **32 FLOP/byte** |

Against the 260.3 FLOP/byte machine balance `size512-ab` measured on this same chip (**to be
re-measured this pass, charter §4.1**), the pair projection is 2.0x below balance and the template
projection is **8.1x** below. The template projection has a quarter of the compute per output byte
to hide its DRAM write behind.

**H5' (the reformulated hypothesis this leg actually tests): the excess is lost compute/communication
overlap at low arithmetic intensity, not program config and not traffic volume.** At `c_z=256` the
matmul has enough work per output tile that the writer's drain overlaps the next K block's compute
and the region approaches `max(compute, comm)`; at `c=64`, `k_tiles = 2` means a core finishes an
output tile after two inner blocks, the DRAM write is exposed, and the op degenerates to a copy
carrying a matmul's transaction pattern — 2 KB per tile, round-robined across DRAM banks — so the
region lands near `compute + comm` with `comm` itself far under the copy roof.

Stated as the limiter rather than in passing: **what holds the template-track projection at ~20 % of
the DRAM copy roof is not bandwidth and not core occupancy — it is that the writer's transactions are
tile-granular (2 KB, round-robined across DRAM banks) and, with `k_tiles = 2`, there is no next K
block whose unpack and math phases the drain can hide inside.** Core occupancy cannot be the limiter:
`cores = ceil(m_tiles / per_core_M)` is 110 of 110 at 512 aa and 100 of 110 at 298 aa, so the grid is
full or nearly so at both sizes, and `k_tiles < num_cores` — the failure that left 109 of 110 cores
idle elsewhere in this codebase — does not apply to an M-split 1D matmul. The DRAM controller reaches
397.7 GB/s on the 4 KB bursts of a plain clone and the same silicon carries this op at a fraction of
that, which is a transaction-rate limit with a circular-buffer cause, not a bandwidth limit.

That is falsifiable three ways and the probe does all three: absolute per-cell timings (not deltas),
the same cells at both channel widths on one card in one process, and the DRAM arm's achieved GB/s
against the clone roof **at the op's own shape**.

---

## PREDICTIONS — registered 2026-08-10 before the device was opened

Numbers are qb2 chip 0, ttnn 0.68.0, 110-core grid. Each names its falsifier.

**P1.** Every field of the program config is identical between the L1-output cell and the DRAM-output
cell at all four shapes. *Falsified* by any differing field. (Code read says this is certain; it is
here so the probe prints the diff rather than assuming it.)

**P2.** A shape-tagged census of `_pair_proj_linear(l1_out=True)` in a live 512 aa fold returns
**320** calls at `w=[64,64]` inside `TriangleMultiplication` and 160 at `TriangleAttention`, totalling
the 480 the sibling counted. *Falsified* by 160 in TriMul.

**P3.** The isolated OFF-minus-ON region delta at `[1,512,512,64]` reproduces the in-fold 2.30
ms/region to within 20 %. *Falsified* outside 1.84-2.76 ms — in which case the fold body wall is
contaminated by allocation pressure the isolated probe does not see, and **that** is the finding.

**P4.** The DRAM arm of the 512 aa template region runs at **15-25 % of the DRAM r+w clone roof
measured at its own shape**, the same band as the permute's 16.8 %; the L1 arm at 20-45 % of the L1
clone roof. *Falsified* if the DRAM arm exceeds 50 % of the copy roof, which would mean the traffic
model rather than the transaction rate is what is wrong.

**P5.** The same region at `[1,298,320,256]` runs at **45-75 %** of the DRAM r+w clone roof at its own
shape — materially higher than P4's band — because K=8 gives the writer something to hide behind.
**This is the discriminating prediction for H5'.** *Falsified* if the two channel widths land in the
same band, which kills H5' and sends the residual back to transaction size alone.

**P6.** No DRAM-output program config closes the gap. Sweeping `in0_block_w` over the divisors of
`k_tiles` and `out_block_h` over the divisors of `per_core_M` and `out_block_w` over the divisors of
`n_tiles`, the best DRAM-output cell is within **1.25x** of the production tuned DRAM cell and leaves
**at least 2.5x** of the L1 win on the table. *Falsified* if any DRAM-output config comes within 1.15x
of the L1 cell — which would confirm H5 in its useful form and be the leg's biggest result.

**P7 — the one nobody has looked at.** At `c_z=256, N=512` the production path returns `None` for an
L1 output (`need = 2 031 616 B` against a 1 461 760 B bank, 139 %) **only because it holds
`in0_block_w = 8` and `out_block_w = n_tiles = 8` fixed.** Two legal configs do fit:

| config | in0/in1 CBs | out CB | fixed | L1 out term | need | % of bank |
|---|---:|---:|---:|---:|---:|---:|
| `bw=1, obh=1, obw=8` | 36 864 | 49 152 | 131 072 | 1 228 800 | **1 445 888** | 98.9 % |
| `bw=1, obh=5, obw=2` | 28 672 | 61 440 | 131 072 | 1 228 800 | **1 449 984** | 99.2 % |

(`per_core_M = 75` is forced: `ceil(8192/110) = 75` and no smaller value keeps
`ceil(m_tiles/per_core_M) <= 110`. `obh` must divide 75; `obw` must divide `per_core_N = 8`.)

**Prediction: at least one of these allocates, and beats the tuned DRAM cell at
`[1,512,512,256]`.** *Falsified* if both throw, or if both are slower than the tuned DRAM cell. If it
holds, X7 has a route across its own 366-token cliff that costs no new capacity test, applies to
**3144 calls per fold** at 512 aa, and is the ranked hand-off Phase 3 asked for. If it fails on
allocation rather than on speed, the finding is that the static budget and the allocator disagree at
99 % of a bank, which is the same class of defect as the 448 aa circular-buffer clash.

---

## The four-cell design (and why two cells cannot do this)

Two axes, crossed, at each of four shapes. A two-cell A/B measures `config+destination` as one number
and cannot attribute it; the cross gives both main effects **and their interaction**, which is the
term that decides whether a DRAM-output config can inherit the L1 config's behaviour.

| | output DRAM | output L1 |
|---|---|---|
| **untuned** `core_grid=CORE_GRID_MAIN` | C0D | C0L |
| **tuned** `_pair_proj_config(...)` | C1D *(= production OFF)* | C1L *(= production ON)* |

- config term at DRAM = `C0D - C1D`
- destination term at the tuned config = `C1D - C1L` — **this is the whole production win**
- destination term at the untuned config = `C0D - C0L`
- interaction = `(C0D - C0L) - (C1D - C1L)`; non-zero means the two levers are not separable and a
  config-only fix cannot be priced by subtraction.

**Shapes are the fold's own** (`[1, N, ceil32(N), c] @ [c, c]`, bf16, source in DRAM as production
leaves it — the trimul's `ttnn.layer_norm` has no `memory_config` and writes DRAM). `[1,320,320,32]`
overstated two retracted projections by 3.2x and 1.45x in this org; it is not used.

| shape | m_tiles | k_tiles | n_tiles | per_core_M | cores engaged | in0_block_w | need DRAM | need L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `[1,298,320,64]` | 2980 | 2 | 2 | 30 | **100 of 110** (90.9 %) | 2 | 249 856 | 372 736 |
| `[1,512,512,64]` | 8192 | 2 | 2 | 75 | **110 of 110** (100 %) | 2 | 249 856 | 557 056 |
| `[1,298,320,256]` | 2980 | 8 | 8 | 30 | **100 of 110** (90.9 %) | 8 | 802 816 | 1 294 336 (88.5 %) |
| `[1,512,512,256]` | 8192 | 8 | 8 | 75 | **110 of 110** (100 %) | 8 | 802 816 | 2 031 616 — **refused** |

Core utilisation is exact from the config, not inferred: `cores = ceil(m_tiles / per_core_M)`. The op
**gains** the grid as it grows and loses per-core L1, which is the inversion `size512-ab` found and
the reason `k_tiles < num_cores` (109 of 110 cores idle) is not the failure mode here.

Each cell is measured three ways so the region and the op can be told apart:

1. **`proj`** — the projection alone.
2. **`region`** — `p_out`, `g_out`, `multiply_` with the sigmoid activation, exactly as
   `tenstorrent.py:1751-1759` runs them. This is the unit the 2.30 ms belongs to.
3. **`region+consume`** — the region plus a downstream `add_` reading the result, to price row 6 of
   the byte table and settle whether it belongs in the model.

Plus, at every shape: `ttnn.clone` of the same tensor to DRAM and to L1 as the **copy roof at the op's
own shape**, which is the only fair roof for an op whose output dominates its traffic.

---

## Roofs to measure on this card this pass (charter §4.1 — never inherited)

`size512-ab`'s roofs were taken on this same chip and are still not reused; they are listed only as
the prior the predictions were written against.

| roof | method | prior (qb2 c0, 0.68.0) |
|---|---|---:|
| DRAM read | 128 MB DRAM->L1 clone | 399.9 GB/s |
| DRAM write | 128 MB L1->DRAM clone | 272.9 GB/s |
| DRAM read+write | DRAM->DRAM clone, both directions counted | 397.7 GB/s |
| L1 op roof | block-sharded bf16 `add` over 110 cores, 3N-byte convention | 7951.1 GB/s |
| compute, square bf16 HiFi4 | 4096^3, DRAM output | 104.11 TFLOP/s |
| **machine balance** | compute / DRAM read | **260.3 FLOP/byte** |
| K-corrected compute roof at K=64 and K=512, at `nt=2` and `nt=8`, **both output buffer types** | charter §4.6 requires both columns | not measured — **owed** |

The charter's 338 FLOP/byte is another card's number. Placement does not turn on which is used: at 32
and 128 FLOP/byte both projections are on the **memory side**, so a **bandwidth roof binds**, and the
`TFLOP/s = write_GB/s x K` identity means the compute column here is a write roof wearing a FLOP/s
label — at K=64 a 272.9 GB/s DRAM write caps the template projection at 17.5 TFLOP/s no matter what
the array can do.

**Compute/communication overlap** is the leg's central question, not a checkbox. It is answered by
comparing each measured cell against `max(compute, comm)` and `compute + comm` built from roofs
measured this pass, at two arithmetic intensities that differ 4x. P5 is that prediction.

---

## Exactly what the execution pass runs

Everything below runs on **qb2, chip 0**, from the worktree
`/home/ttuser/.coworker/wt/protenix-trunk--z-progcfg-h5`, with:

```
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-progcfg-h5
MESH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
ENV="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-progcfg-h5 TT_MESH_GRAPH_DESC_PATH=$MESH PYTHONPATH=$WT"
```

`PYTHONPATH=$WT` is mandatory: the env has `tt_bio` installed editable and a script run resolves
`import tt_bio` to the shared checkout, not this worktree. `TT_MESH_GRAPH_DESC_PATH` is what lets a
single chip of a p300 board open (charter §4.8); without it the open fails with
*"Board ... has 1 chips, but expected 2 chips for board type p300"*.

**Step 1 — roofs and the four cells, one process, ~12 min.** No fold, no weights.

```
cd $WT && env $ENV $PY perf/progcfg/h5_cells.py --out perf/progcfg/h5_cells_qb2c0.json
```

Order inside the process: roofs first, then per-shape clone roofs, then the 2x2 cells, then the
DRAM-output config sweep, then the P7 L1-output cells at `[1,512,512,256]`. Every timed region
synchronises on both sides; median of 5 reps of 4 piped calls after 3 warm-ups (an unsynced `to_torch`
drain has inverted rankings in this codebase before). Each cell records `torch.equal` against C1D so a
config that changes values is caught rather than credited.

**Step 2 — the in-fold census and the region wall, ~25 min.** Settles P2 and P3.

```
cd $WT && env $ENV $PY perf/progcfg/h5_infold.py --size 512 --out perf/progcfg/h5_infold_qb2c0.json
```

One process, `build_fold` once, `_PAIR_PROJ_L1_OUT` flipped between folds so both arms share weights,
MSA cache and warm program cache. Instrument `_pair_proj_linear` to tally `(x.padded_shape, w.shape)`
and the branch taken, and wrap `_trimul_out_proj` + the following `multiply_` in a synchronised region
wall. Report the region wall per arm, the census, and the CIF sha256 + plDDT per arm — both arms must
produce **byte-identical** output, since only a memory config moves. At 512 aa the fixture is
`perf/size512/fixtures/cdk2x2_512.yaml` (already on branch `wk/protenix-trunk--size512-ab`, cherry-pick
`perf/size512/fixtures/` rather than rebuilding it).

**Step 3 — write the verdicts.** Fill V3 (H5' confirmed or killed by P5), the four-cell table, the
roof table, and either the priced Phase-3 hand-off (if P6 or P7 holds) or the plain statement that
there is nothing here. A well-evidenced "nothing here" is a full pass.

**Budget guard.** If step 2 cannot finish in the turn, commit step 1's JSON and the verdicts it
settles. Step 1 alone settles P1, P4, P5, P6 and P7 — that is H5' and the Phase-3 hand-off. Step 2
only settles the sibling's attribution.

---

## Decided against, so execution does not relitigate

- **Re-running the fold-level ON/OFF A/B.** `size512-ab` already has it with an A/A floor and
  byte-identical CIFs at three sizes. Repeating it produces the same number and no new term.
- **Sweeping `_PAIR_PROJ_BW` / `_NARROW_PROJ_BW`.** `_PAIR_PROJ_BW = 16` shipped at `bbb5d85bd`,
  `y-envelope-audit` scored it at 0.42 of the integration-parity bound and recommended keep. It is the
  baseline, not a variable. A control against `in0_block_w = 1` is a control against code main does
  not run.
- **Re-deriving the `in0_block_w` ladder.** `trimul-rescore` has it: 587.9 / 469.0 / 424.7 / 385.0 us
  at bw = 1/2/4/8. Reuse it.
- **Testing whether in1 residency in L1 helps.** `trimul-rescore` killed it: 384.81 vs 385.04 us. The
  writer is not starved by sharing BRISC with the in1 multicast sender.
- **`[1,320,320,32]` or any synthetic shape.** Priced at the fold's own shapes or the number is void.
- **The row-blocked transpose.** `z-rowblock` owns it on qb1 card 1. Different op, different cliff.
  If both legs succeed the CTO reconciles the overlap.
- **SDPA chunking.** Dead: 2.941 / 3.601 / 2.973 / 1.725 Å against a 0.1136 Å bound.
- **Fusing `p_out` and `g_out` into one `[c, 2c]` projection.** Considered and rejected on reading the
  code: they take **different inputs** (`norm_out(x)` and `x_norm_in`, `tenstorrent.py:1746` and
  `1754`), so no single matmul expresses them. The symmetry with the fused `_gp_in_chunks` input
  weights does not carry to the output side.
- **Changing anything in `tt_bio/`.** Phase 2. The probes import the helpers and build configs beside
  them; they do not edit them.

---

## Parity

Nothing in this leg changes production, so there is no parity verdict to give. The probes still assert
`torch.equal` per cell, for one reason: a **memory config cannot change a value**, so any cell that
differs from C1D has changed the contraction order and is a different parity class, not a faster
version of the same op. That distinction is what `_PAIR_PROJ_L1_BW`'s docstring records and what the
DRAM-output config sweep in step 1 could otherwise quietly cross.

---

## Files

| path | what |
|---|---|
| `perf/progcfg/h5_cells.py` | roofs, per-shape clone roofs, the 2x2 cells, the DRAM config sweep, the P7 L1 cells |
| `perf/progcfg/h5_infold.py` | 512 aa census + region wall, both arms, one process |
| `perf/progcfg/h5_cells_qb2c0.json` | step 1 results — **not yet produced** |
| `perf/progcfg/h5_infold_qb2c0.json` | step 2 results — **not yet produced** |

Branch `wk/protenix-trunk--z-progcfg-h5` on qb2. Nothing merges from this leg.
