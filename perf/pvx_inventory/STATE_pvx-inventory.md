# pvx-inventory — which shipped levers fire on Protenix v2, counted at run time

TASK TYPE: VERIFY/BENCHMARK (a firing census, not an A/B) | PLAYBOOKS loaded: ACCELERATE "THE PERF
METHOD" + ALWAYS-ON | memories read: `eligibility-firing-condition-is-not-a-code-fact`,
`one-size-tuning-is-a-standing-defect-class`, `census-key-label-is-not-an-executed-shape`,
`unified-solution-not-per-model-patches`, `merged-lever-defaults-off-is-not-a-landed-win`,
`perf-page-cell-is-historical-not-live-baseline`, `perf-ab-session-is-the-independent-unit`,
`benchlock-protects-co-tenants-not-just-the-caller`,
`l1-budget-derived-from-live-grid-makes-output-host-dependent`.

Branch `wk/pvx-inventory`, worktree `/home/ttuser/.coworker/wt/pvx-inventory` on qb1, artifacts
`perf/pvx_inventory/`. Device work on **qb1 card physical 0, Blackhole p150a, 11x10 grid**,
**AICLK sampled DURING every fold**: 19 of 20 Protenix samples and 3 of 3 Boltz-2 samples read
1350 MHz, the part's `asic_fmax`; the remaining one read 1343. Nothing here was measured on a
decayed governor.

---

## The headline: bucket 2 is POPULATED, and the design doc's main conclusion is wrong on counts

`state/pvx/DESIGN.md` states its own falsifier: *"If `pvx-inventory` comes back with a populated
bucket 2 — several shared levers with counted firing rates near zero on Protenix and materially
above zero on Boltz-2 — then world 2 is live after all."* **That is what the counts say.** Seven
shared, default-ON levers fire on every Boltz-2 call they are offered and on 0-37 % of
Protenix's, and three of them fire on Protenix at **exactly zero**:

| lever | Boltz-2 | Protenix v2 | Protenix rate |
|---|---|---|---|
| `TRIATT_FUSED_QKVG` (K3) | **560 / 0** | **0 / 1208** | **0.0 %** |
| `TRIATT_FUSED_QKVGB` | **560 / 0** | **0 / 1208** | **0.0 %** |
| `PAIR_PROJ_MINIMAL_MATMUL` | 0 / 0 (never reached) | 0 / 1208 | 0.0 % |
| `TRIMUL_MASK_L1` | **528 / 0** | **0 / 0** (never reached) | **0.0 %** |
| `TT_BIO_RESIDUAL_L1` | **560 / 0** | **80 / 1048** | **7.1 %** |
| `TT_BIO_TRANSITION_L1_ROWS` (LEDGER N1) | **296 / 0** | **110 / 526** | **17.3 %** |
| `QKV_MM_CONFIG` (`_MM_BLOCK` lookup) | **560 / 0** | **2256 / 3784** | **37.4 %** |

The E6 correction in the ledger stands and this pass confirms it by count: `REBLOCK_PERMUTE_GATED`
serves **2416 / 0** on Protenix, the exact number `protenix.py:1940` records. E6 was never the
example. **K3 is**, and it has the identical signature E6 had before it was fixed: a shared fused
kernel, default ON, 100 % served on Boltz-2, **ineligible on 100 % of Protenix's 1208 triangle
attentions**.

**The gap this closes is small, and that part of the design doc survives.** Section SCREEN prices
the whole of bucket 2 at an upper bound of **~1.06x** on Protenix's fold against a measured
same-board gap of ~3.19x. World 3 is still the world we are in *by magnitude*. It is world 2 *by
count*, and the two rows downstream of this one need the counts, not the magnitude.

---

LEVERS: 58 registered levers, read off the shipped tree at `bd643929a` rather than the git log,
each with the module that executes it and the counter that proves it ran. `scripts/lever_census.py`
carried 38; this pass added 20 that already kept a `[served, declined]` counter next to their guard
and had no census row, and gave counters to the three `eltwise_fusion` fusions that had none.

### 1.1 What was already registered, and what this pass added

The registry lives in `scripts/lever_census.py:LEVERS` as `(flag, module, resolved attribute,
counter, how)`. That is the canonical list and this doc does not restate it row by row; what
matters here is the **coverage gap it had**, because an inventory that cannot see a lever reports
it as absent.

Added this pass, all default-ON unless marked, all reaching more than one model through
`tenstorrent.py`:

| lever | module that executes it | what it does |
|---|---|---|
| `TRIMUL_MASK_L1` | `tenstorrent.py:471` | the trimul pair mask in L1 instead of DRAM |
| `RESIDUAL_L1` | `tenstorrent.py:478` | the Pairformer residual's update operand produced into L1 |
| `PWA_BATCH_HEAD_WEIGHTS` | `tenstorrent.py:509` | one PairWeightedAveraging projection for all heads |
| `ATOM_SHIFT_GATHER` | `tenstorrent.py:878` | slice+concat atom key gather instead of a one-hot matmul |
| `SDPA_RAGGED_PAD` | `tenstorrent.py:1739` | pads a ragged SDPA axis so the fused kernel serves |
| `TRIMUL_FUSED_GOUT` | `tenstorrent.py:5972` | the trimul gate folded into the out projection |
| `TRIMUL_TAIL_L1` (OFF) | `tenstorrent.py:625` | the F1 tail's operands held in L1 |
| `TRIMUL_TAIL_F1_L1_OUT` | `tenstorrent.py:5948` | the F1 tail packing its product straight into L1 |
| `TRIATT_FUSED_QKVG` | `triatt_qkv.py:248` | q, k, v **and the gate** from one pass over the normed pair tensor |
| `TRIATT_FUSED_QKVGB` | `triatt_qkv.py:358` | the same, with the pair-bias projection joined |
| `TRIATT_GATE_EPILOGUE` (OFF) | `triatt_sdpa.py:302` | the output gate folded into the fused SDPA's pack stage |
| `TRIATT_FUSED_HIFI` (OFF) | `tenstorrent.py:2221` | the fused SDPA at fp32 accumulation and HiFi4 |
| `OPM_ROW_BLOCK` | `tenstorrent.py:112` | c14's OuterProductMean output-stage row blocking |
| `PWA_DEPTH_BLOCK` | `tenstorrent.py:68` | the PWA depth block, read the same way |
| `OPM_SMALL_DEPTH` (OFF) | `tenstorrent.py:125` | `proj_o` folded into the outer product |
| `FP32_SOFTMAX_FUSED` | `tenstorrent.py:3060` | the fused branch of the fp32 softmax |
| `FUSE_SCALE_ADD` | `eltwise_fusion.py:43` | attention's scale-then-bias as one `addalpha` |
| `FUSE_MASK_ADD` | `eltwise_fusion.py:47` | the gated-residual write-back as one `addcmul` |
| `FUSE_NORM_RESIDUAL` | `eltwise_fusion.py:52` | an add whose only consumer is a norm, folded into it |
| `PROTENIX_RELP_SCATTER` | `protenix.py:394` | the relative-position one-hot built by scatter |

The three `eltwise_fusion` levers are the ones that make the point about registries. They are
default ON, they are the highest-count fusion in the tree, and **`scale_add` declines on the
OPERAND DTYPE** — a property of the call site, invisible in the flag. Counted this pass on Protenix:
**6000 served, 3 declined.** Those 3 are exactly the bf16 calls the shipped docstring predicts and
deliberately refuses, because fusing them moves a structure 1.475 A. A registry that reads flags
could not have produced that line; a counter produces it for free.

### 1.2 What the inventory deliberately does NOT claim

This is a list of **shipped levers**, not a reconstruction of Boltz-2's 23.504 s -> ~14.4 s. The
brief asks for the second and the tree does not support it honestly: 220 perf-ish commits, several
levers landed and later subsumed, and at least one — `triatt_sdpa_hifi` — that reads like a Boltz-2
win and ships `False` at both of its two sites (LEDGER C4). Attributing seconds to levers from the
log is exactly the error `perf-page-cell-is-historical-not-live-baseline` names. What is defensible
is: **this is what is on today, this is which module runs it, and this is how many times it ran.**

---

FIRING: measured in process, one cold fold and one warm fold per model, every counter zeroed
between the two, the warm fold's counts reported. Both models at **512 aa on the same fixture**
(`perf/size512/fixtures/cdk2x2_512.{yaml,a3m}`, 35-row a3m) and **at their own shipped protocol**,
because folding one model at the other's recycling count would move every trunk count on this page
by that ratio with nothing saying so.

| | Boltz-2 | Protenix v2 |
|---|---|---|
| trunk cycles x blocks | 4 x 64 = **256** | 10 x 48 = **480** |
| sampling steps, samples | 200, 1 | 200, 1 |
| warm fold (contended, see below) | 17.727 s | 52.193 s / 52.291 s (two passes) |
| AICLK DURING | **1350 MHz** (3/3 samples) | **1350 MHz** (19 of 20 samples over two passes; one 1343) |
| plDDT | 0.844645 | 0.810638 |
| artifact | `perf/pvx_inventory/firing_boltz2_512_qb1c0.json` | `perf/pvx_inventory/firing_protenix-v2_512_qb1c0.json` |

**The fold seconds above are NOT admissible as perf numbers and nothing in this campaign may score
against them.** Both runs were taken outside `benchlock` while `pvx-eligibility` was folding
protenix-v2 on card 2 of the same host. That is deliberate: a call count does not move with host
load, the campaign ledger admits a census under contention where it refuses a timed A/B, and taking
the lock for a census would have blocked `pvx-baseline`, whose clocked cell genuinely needs it.
`pvx-baseline` owns both anchors. The seconds are recorded only so the counts have a fold to belong
to. The **counts** are reproducible: each model was folded twice in separate processes, and every
one of the 58 rows read identically both times, plDDT 0.844645 and 0.810638 to the last digit. The
two Protenix folds landed 52.193 s and 52.291 s, 0.19 % apart, which is the closest thing to an
A/A floor this row is entitled to claim and is not offered as one.

### 2.1 Reading a zero

A row is one of three things and the instrument separates them:

* `served > 0, declined = 0` — fires on every call it is offered.
* `served = 0, declined > 0` — **offered and refused**, with the guard's own clause recorded. This
  is bucket 2's signature.
* `served = 0, declined = 0` — **never reached**. Not "declining", not "inactive": the code path
  that would offer it does not execute in this model. That is a different finding and it needs a
  different fix.

`eligibility-firing-condition-is-not-a-code-fact` is why this row exists, and the three-way split
is the whole content of it.

### 2.2 The per-fold counts, both models

### 2.3 Every lever, both models, one table

The updated brief asks for the count on **every** lever including the ones that fire, because
bucket 1 is what tells `pvx-protenix-specific` which levers are **already priced into the shared
stack's 1.1988x and must not be counted again**. Generated from the two artifacts by
`perf/pvx_inventory/fulltable.py`, so it cannot drift from them.

The bucket column is **not** derived from the counts alone. `served=0, declined=0` means the code
path never executed, and whether that is bucket 2 or bucket 3 is a judgement about the tree; every
such judgement is written into `fulltable.py:OVERRIDE` with its reason and shows up in the last
column. Everything without a reason fell out of the counts.

Totals: **12 bucket 1, 7 bucket 2, 4 reverse-direction, 35 bucket 3.**

| lever | resolved | Boltz-2 s/d | Protenix s/d | px rate | bucket | why, where the count alone cannot say |
|---|---|---|---|---|---|---|
| `SPLIT_SWIGLU` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `SPLIT_SWIGLU_SMALL_GRID` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_FFN_L1_FC1` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_FFN_L1_LN` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_FFN_L1_SLICE` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_FFN_FUSED_RESIDUAL` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_FFN_FILL_ASSEMBLY` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `TRIMUL_IN_PROJ_DUAL_NOC` | `True` | 560 / 0 | 1208 / 0 | 100.0 % | **1** |  |
| `ADALN_S_HOIST` | `True` | 2400 / 0 | 13209 / 0 | 100.0 % | **1** |  |
| `FP32_SOFTMAX_BIAS_HOIST` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `FP32_SOFTMAX_L1_GRID` | `(8, 8)` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PAIR_TRANSPOSE_VIA_ROW_MAJOR` | `True` | 0 / 560 | 0 / 1208 | 0.0 % | 3 | l1_dest_is_faster on both; a correct decline |
| `PAIR_PROJ_MINIMAL_MATMUL` | `True` | 0 / 0 | 0 / 1208 | 0.0 % | **2** | 0/1208 on protenix, no_mm_block:(8,1) -- same table as QKV_MM_CONFIG |
| `TRIMUL_TAIL_F1` | `True` | 0 / 0 | 1048 / 0 | 100.0 % | **2-rev** |  |
| `QKV_MM_CONFIG` | `True` | 560 / 0 | 2256 / 3784 | 37.4 % | **2** | 2256/6040 on protenix, 3784 missing _MM_BLOCK keys |
| `DEVICE_LM_HANDOFF` | `-` | not-imported | not-imported | - | 3 |  |
| `REBLOCK_PERMUTE` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `REBLOCK_PERMUTE_BACK` | `True` | 560 / 0 | 1208 / 0 | 100.0 % | **1** |  |
| `REBLOCK_PERMUTE_GATED` | `True` | 1120 / 0 | 2416 / 0 | 100.0 % | **1** |  |
| `TRIMUL_MASK_AFTER_MOVE` | `True` | 1120 / 0 | 2416 / 0 | 100.0 % | **1** |  |
| `TRIATT_PERSISTENT_MASK` | `True` | 560 / 0 | 1208 / 0 | 100.0 % | **1** |  |
| `SDPA_WIDE_K` | `False` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `SDPA_FUSED_LARGE_S` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `RFD3_SPARSE_BIAS` | `-` | not-imported | not-imported | - | 3 |  |
| `RFD3_FUSED_SCORES` | `-` | not-imported | not-imported | - | 3 |  |
| `TRIATT_HEAD_MAJOR_QKV` | `True` | 560 / 0 | 1048 / 160 | 86.8 % | **1** | 1048/1208; the 160 declines are the _MM_BLOCK miss, not this gate |
| `TRIATT_HEAD_MAJOR_TAIL` | `True` | 560 / 0 | 1048 / 0 | 100.0 % | **1** |  |
| `TRIATT_TAIL_OVER_L1` | `True` | 560 / 0 | 1048 / 0 | 100.0 % | **1** |  |
| `B2_BIAS_SLICE_HOIST` | `True` | 3 / 0 | 0 / 0 | never reached | 3 | boltz2.py-exclusive; protenix has its own diffusion module |
| `B2_ADALN_S_MEMO` | `True` | 2400 / 0 | 0 / 0 | never reached | 3 | same |
| `B2_TOKEN_DIT_SDPA` | `True` | 4800 / 0 | 0 / 0 | never reached | 3 | same |
| `APB_CONCAT_HEADS` | `False` | 0 / 5064 | 0 / 5284 | 0.0 % | 3 |  |
| `ATOM_AXIS_BUCKET` | `True` | 200 / 0 | 0 / 0 | never reached | 3 | same |
| `TRANSPOSE_L1_RESIDENT` | `1.25` | 560 / 0 | 1208 / 0 | 100.0 % | **1** |  |
| `SDPA_Q_CHUNK_FITS` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `RFD3_SOFTMAX_PV_FUSED` | `-` | not-imported | not-imported | - | 3 |  |
| `RFD3_FC1_SPLIT_SILU` | `-` | not-imported | not-imported | - | 3 |  |
| `TRANSITION_H_CHUNK` | `16` | 296 / 0 | 110 / 526 | 17.3 % | **2** | 110/636 on protenix, base-unraised x526 at c=256 (LEDGER N1) |
| `TRIMUL_MASK_L1` | `True` | 528 / 0 | 0 / 0 | never reached | **2** |  |
| `RESIDUAL_L1` | `True` | 560 / 0 | 80 / 1048 | 7.1 % | **2** | 80/1128 on protenix, capacity refusal at c_z=256 |
| `PWA_BATCH_HEAD_WEIGHTS` | `True` | 16 / 0 | 30 / 0 | 100.0 % | **1** |  |
| `ATOM_SHIFT_GATHER_OFF` | `False` | 1200 / 0 | 0 / 0 | never reached | 3 | same |
| `SDPA_RAGGED_PAD` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `TRIMUL_FUSED_GOUT` | `True` | 560 / 0 | 160 / 1048 | 13.2 % | **1** | 160/1208, the 1048 declines read f1_tail_serves and are correct |
| `TRIMUL_TAIL_L1` | `False` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `TRIMUL_TAIL_F1_L1_OUT` | `True` | 0 / 0 | 0 / 1048 | 0.0 % | **2-rev** | 0/1048 on protenix, never offered on boltz2 |
| `TRIATT_FUSED_QKVG` | `True` | 560 / 0 | 0 / 1208 | 0.0 % | **2** |  |
| `TRIATT_FUSED_QKVGB` | `True` | 560 / 0 | 0 / 1208 | 0.0 % | **2** |  |
| `TRIATT_GATE_EPILOGUE` | `False` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `TRIATT_FUSED_HIFI` | `False` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `OPM_ROW_BLOCK` | `256` | 0 / 16 | 0 / 40 | 0.0 % | 3 |  |
| `PWA_DEPTH_BLOCK` | `268435456` | 0 / 16 | 0 / 30 | 0.0 % | 3 |  |
| `OPM_SMALL_DEPTH` | `False` | 0 / 16 | 0 / 40 | 0.0 % | 3 |  |
| `FP32_SOFTMAX_FUSED` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `FUSE_SCALE_ADD` | `True` | 0 / 0 | 6000 / 3 | 100.0 % | **2-rev** |  |
| `FUSE_MASK_ADD` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `FUSE_NORM_RESIDUAL` | `True` | 0 / 0 | 0 / 0 | never reached | 3 |  |
| `PROTENIX_RELP_SCATTER` | `True` | not-imported | 1 / 0 | 100.0 % | **2-rev** |  |

**The bucket-1 rows are the double-counting hazard.** Every one of them already fires on Protenix
at 100 % (or, for `TRIATT_HEAD_MAJOR_QKV` and `TRIMUL_FUSED_GOUT`, at a rate whose shortfall has a
different owner). Their value is inside whatever `pvx-baseline` anchors Protenix at, exactly as the
shared stack's 1.1988x is. **No row may propose them as an opportunity.** `REBLOCK_PERMUTE_GATED`
at 2416/2416 is the specific one to watch: it is E6, it is closed in the ledger, and it reads like
a headline lever to anyone who has not seen the count.

**Fires on both (bucket 1).** Counts scale with the trunk, as they should: Protenix runs 1208
trimuls and 1208 triangle attentions against Boltz-2's 560, a ratio of **2.157x**, against a trunk
block-execution ratio of 480/256 = 1.875x.

| lever | Boltz-2 served/declined | Protenix served/declined |
|---|---|---|
| `TRIMUL_IN_PROJ_DUAL_NOC` | 560 / 0 | 1208 / 0 |
| `REBLOCK_PERMUTE_BACK` | 560 / 0 | 1208 / 0 |
| `REBLOCK_PERMUTE_GATED` (E6) | 1120 / 0 | **2416 / 0** |
| `TRIMUL_MASK_AFTER_MOVE` | 1120 / 0 | 2416 / 0 |
| `TRIATT_PERSISTENT_MASK` (K2) | 560 / 0 | 1208 / 0 |
| `TRIATT_HEAD_MAJOR_QKV` (K1) | 560 / 0 | 1048 / 160 |
| `TRIATT_HEAD_MAJOR_TAIL` | 560 / 0 | 1048 / 0 |
| `TRIATT_TAIL_OVER_L1` | 560 / 0 | 1048 / 0 |
| `TRANSPOSE_L1_RESIDENT` | 560 / 0 | 1208 / 0 |
| `ADALN_S_HOIST` | 2400 / 0 | 13209 / 0 |
| `PWA_BATCH_HEAD_WEIGHTS` | 16 / 0 | 30 / 0 |

**Fires on Boltz-2, not on Protenix (bucket 2).** The table at the top of this doc. Blocking
conditions in BUCKETS below.

**Fires on Protenix, not on Boltz-2.** The brief does not have a bucket for this and it is worth a
line, because it is the same defect class pointing the other way:

| lever | Boltz-2 | Protenix | note |
|---|---|---|---|
| `TRIMUL_TAIL_F1` | 0 / 0 | **1048 / 0** | Protenix's trimul takes the fused F1 tail; Boltz-2's never reaches it |
| `FUSE_SCALE_ADD` | 0 / 0 | **6000 / 3** | Boltz-2 calls no `eltwise_fusion` site at all |
| `PROTENIX_RELP_SCATTER` | n/a | 1 / 0 | protenix-only by construction |

**Neither (bucket 3).** Model-exclusive, structurally absent, or shipped OFF. The `boltz2.py`
family — `B2_BIAS_SLICE_HOIST` 3, `B2_ADALN_S_MEMO` 2400, `B2_TOKEN_DIT_SDPA` 4800,
`ATOM_AXIS_BUCKET` 200, `ATOM_SHIFT_GATHER` 1200 on Boltz-2 — all read **0 / 0** on Protenix,
because Protenix has its own diffusion module and does not route through `tenstorrent.py`'s atom
and token DiT path at all. The seven `esmc` rows read 0 / 0 on both. `SDPA_FUSED_LARGE_S`,
`SDPA_WIDE_K` and `SDPA_RAGGED_PAD` read 0 / 0 on both, all three for the reason their own comments
give: 512 is below the large-S cap, `512 % 256 == 0` so the wide-k ladder is never consulted
(LEDGER A1), and token bucketing leaves 512 aligned so nothing is ragged.

---

BUCKETS: 1 fires on both, 2 fires on Boltz-2 and not on Protenix, 3 cannot apply. Every bucket-2
lever below carries the exact clause that blocks it, taken from the guard's own reject dict where
it keeps one.

### 3.1 Bucket 2, with the blocking condition named

**B2-a. `TRIATT_FUSED_QKVG` / `TRIATT_FUSED_QKVGB` — 0 of 1208, the campaign's prize.**
The guard's own reject dict, counted: **`dtype_or_memory_or_config` x 1208, i.e. 100 %.** That
clause is `if not _common_ok(x, w, dtype) or mm_config is None` (`triatt_qkv.py:277`), and the
`_common_ok` half cannot be what fires, because the earlier `head_dim_or_width` clause passed on
all 1208 (the concatenated weight exists and is the right width) and the unfused head-major
projection serves 1048 of the same calls through the same dtype and memory config. **It is
`mm_config is None`, and the reason is one missing pair of `_MM_BLOCK` entries:**

    _MM_BLOCK = {                       # tenstorrent.py:7078
        (8, 24): (4, 8, 1, 4, 1),       # protenix-v2 qkv            PRESENT
        (8,  8): (4, 8, 1, 4, 1),       # protenix-v2 gate + pair    PRESENT
        (4, 12): (4, 4, 1, 4, 1),       # boltz2 qkv   at c_z=128    PRESENT
        (4,  4): (4, 4, 1, 4, 1),       # boltz2 gate  at c_z=128    PRESENT
        (4, 16): (4, 4, 1, 4, 1),       # boltz2 qkv+gate      FUSED, PRESENT
        (4, 17): (4, 4, 1, 4, 1),       # boltz2 qkv+gate+bias FUSED, PRESENT
        # (8, 32) protenix-v2 qkv+gate       FUSED -- ABSENT
        # (8, 33) protenix-v2 qkv+gate+bias  FUSED -- ABSENT
    }

Boltz-2's *fused* keys were added when K3 landed; Protenix's were not. `_fused_qkvg` asks
`_qkv_mm_config(x, self.qkvg_weight)` over the concatenated weight, which is 4 x 256 = 1024 wide,
i.e. `nt = 32`, the table misses, `mm_config` comes back `None` and the fusion declines. Every
time, on every call, for the whole fold.

**This is E6 again, mechanism and all.** A shared fused kernel, shipped ON, measured on Boltz-2,
serving 560 of 560 there and **0 of 1208** on Protenix, and the thing that blocks it is not the
algebra but a lookup table that was extended for one channel width and not the other. The
bit-exactness argument the shipped comment makes for `(4, 16)` transfers verbatim to `(8, 32)`:
`(8, 24)` and `(8, 8)` both carry `K_block = 8`, which is the whole contraction at `kt = 8`, so a
fused key folding K the same way accumulates every output element in the order the two separate
matmuls accumulate it today. That argument still has to be **verified** rather than asserted, and
`pvx-eligibility` owns the file and the check.
Boltz-2 serves 560 of 560. K3 deletes one full read of the normed pair tensor per triangle
attention — 134.2 MB of a block's 8.05 GB at 512 aa, measured by walking device allocations
(`perf/b2z2_byte_floor/`). On Protenix that read happens 1208 times a fold and the fusion is
offered and refused every single time. This is E6's signature exactly.

**B2-b. `TRIMUL_MASK_L1` — 0 / 0, never reached.** Boltz-2 528 / 0. The lever puts the trimul's
`[1,1,L,L]` pair mask in L1 instead of DRAM, priced at 0.8566 -> 0.6238 ms a call on Wormhole at the
production shape (`perf/k10_binaryng/`), bit-exact. Protenix's trimul reaches the site zero times,
which is a **structural** statement and not a gate: the fix is not to widen a predicate, it is to
find out which path Protenix's trimul takes instead. `pvx-eligibility` owns that.

**B2-c. `TT_BIO_RESIDUAL_L1` — 80 of 1128 (7.1 %).** Boltz-2 560 / 0. The Pairformer residual's
update operand produced into L1 rather than DRAM, 0.4923 -> 0.3805 ms a call on Blackhole
(`perf/util_op_deletes/`). The shipped comment says both producers fall back to DRAM when the pair
tensor does not fit, "which is what happens above 512 aa" — on Protenix it happens **at** 512 aa,
because the pair tensor is c_z 256 and therefore twice Boltz-2's bytes at the same token count.
A size threshold fitted at one channel, read at another: `one-size-tuning-is-a-standing-defect-class`.

**B2-d. `TT_BIO_TRANSITION_L1_ROWS` — 110 of 636 (17.3 %). LEDGER N1, step 1, CONFIRMED.**
The Blackhole Transition row raise is gated `_TRANSITION_L1_ROWS and _c <= _BH_TRANSITION_L1_ROWS_MAX_C`
with the bound at **128** (`tenstorrent.py:306`, `:8626`). Counted per call, by the channel each
`Transition` call presents:

| model | c=64 | c=128 | c=256 | c=384 | c=768 | 4-D calls that take the raise |
|---|---:|---:|---:|---:|---:|---|
| Boltz-2 | 16 | 280 | — | 264 | 400 | **296 of 296 (100 %)** |
| Protenix v2 | 80 | 30 | **526** | 884 | — | **110 of 636 (17.3 %)** |

`TRANSITION_H_CHUNK` reads `served=296, declined=0` on Boltz-2 and `served=110, declined=526` on
Protenix, and 296 = 280 + 16 and 110 = 80 + 30 exactly, so the 4-D chunking path is reached by the
`c <= 256` calls in both models and the c=384/768 calls belong to a different (single-track) shape
class that never reaches it. **Protenix's 526 refusals all carry the reason `base-unraised`** and
all sit at `c == 256`, which is Protenix's pair channel. N1 is confirmed by count: **the raise is
barred on 82.7 % of the Transition calls Protenix makes, by a channel-width constant.**

N1's own falsifier — "if Protenix's Transition calls do not reach `:8626` with `_c == 256`, N1 is
wrong and dies here" — does not fire. They reach it 526 times.

One correction to N1's framing, and it matters for the fix being unified rather than
Protenix-shaped: the bound is **not** a Boltz-2-versus-Protenix line. It bars Boltz-2's own c=384
and c=768 Transition calls too; those simply take a different code path that never asks. The
channel that separates the two models is the **pair track**: Boltz-2's is 128 and passes,
Protenix's is 256 and does not. A per-core budget measured at c=256 with the core count the matmul
actually uses — which is what the shipped comment demands — serves c=256 and c=384 alike and needs
no model branch.

**B2-e. `QKV_MM_CONFIG` / `_MM_BLOCK` — 2256 of 6040 (37.4 %), and it is the ROOT of B2-a.**
Boltz-2 560 / 0, a perfect hit rate. Protenix's 3784 misses, keyed and counted:

| missing `(kt, nt)` | calls | what it is |
|---|---:|---|
| `(8, 32)` | 1048 | protenix-v2 qkv+gate **fused** — blocks K3 |
| `(8, 33)` | 1048 | protenix-v2 qkv+gate+bias **fused** — blocks K3b |
| `(8, 1)` | 1048 | the one-tile pair-bias projection at c_z=256 |
| `(2, 6)` / `(2, 8)` / `(2, 9)` | 320 / 160 / 160 | the narrow (c=64) triangle-attention sites |

The same table blocks `PAIR_PROJ_MINIMAL_MATMUL` (`no_mm_block:(8,1)` x 1048) and
`TRIATT_HEAD_MAJOR_QKV` (`no_mm_config` x 160). **This is one defect, not five.** A lookup table
swept on one model's widths, read by five levers, applied to a second model's widths. It is the
purest instance of `one-size-tuning-is-a-standing-defect-class` in this census and it is the entry
point to B2-a.

**B2-f. `TRIMUL_FUSED_GOUT` — 160 of 1208, and this one is NOT a loss.** Protenix declines 1048
with reason `f1_tail_serves`: the fused F1 tail already deletes that read, and `TRIMUL_TAIL_F1`
serves those same 1048 calls. Protenix is taking the better of the two paths. Recorded here so it
is not mistaken for a gap by the next reader; **excluded from the SCREEN**.

**B2-g. `TRIMUL_TAIL_F1_L1_OUT` — 0 of 1048 on Protenix, 0 / 0 on Boltz-2.** The one bucket-2 row
where Boltz-2 is the model that never reaches the site. Protenix reaches it 1048 times and is
refused every time. Small, but it is free to look at while B2-f's path is already being read.

### 3.2 Bucket 3, and why each is genuinely inapplicable

* **`boltz2.py`'s five device-residency levers** and the shared atom/token-DiT levers
  (`ATOM_AXIS_BUCKET`, `ATOM_SHIFT_GATHER`, `B2_TOKEN_DIT_SDPA`, `B2_ADALN_S_MEMO`,
  `B2_BIAS_SLICE_HOIST`): Protenix has its own diffusion module in `protenix.py`. Porting them is
  not an eligibility fix, it is a second port, and it belongs to `pvx-protenix-specific`.
* **`esmc` (7 rows), `rfd3` (4 rows), `DEVICE_LM_HANDOFF`**: other models. `not-imported` on both.
* **`SDPA_WIDE_K`, `SDPA_FUSED_LARGE_S`, `SDPA_RAGGED_PAD`**: 0 / 0 on both, each for a documented
  size reason, at 512 aa specifically. They are not dark; 512 is simply not their size.
* **Shipped OFF on both** (`APB_CONCAT_HEADS` 0/5064 and 0/5284, `TRIATT_GATE_EPILOGUE`,
  `TRIATT_FUSED_HIFI`, `TRIMUL_TAIL_L1`, `OPM_SMALL_DEPTH`): default-off flags, not levers this
  campaign may count as shipped. `merged-lever-defaults-off-is-not-a-landed-win`.
* **`PAIR_TRANSPOSE_VIA_ROW_MAJOR`**: 0/560 and 0/1208, reason `l1_dest_is_faster` on both. A
  correct decline in both models, and a reminder that a bare `served=0` is not a defect.

---

SCREEN: if every bucket-2 lever fired on Protenix at the rate it reaches on Boltz-2, Protenix's
512 aa fold lands at an upper bound of **~1.06x faster**, and only three of the seven can be priced
at all. Rate provenance is named per line; the arithmetic is a screen and not a promise.

| bucket-2 lever | Boltz-2 rate provenance | applied to Protenix | s |
|---|---|---|---:|
| `TT_BIO_TRANSITION_L1_ROWS` | **1.0352x** on the Boltz-2 512 aa fold, 15.270 -> 14.750 s, 4 paired reps, A/A floor 0.005 s (104x), `state/roof-transition-chunk-bh-ship.md` | ratio transfer at 100 % firing | **-1.77** |
| `TRIATT_FUSED_QKVG` | **1.00693x (n=6) / 1.01705x (n=8)** on the Boltz-2 fold, `state/b2z2-everything-union-wh.md` (Wormhole) | upper of the two, ratio transfer | **-0.87** |
| `TT_BIO_RESIDUAL_L1` | **0.4923 -> 0.3805 ms/call** on Blackhole, `perf/util_op_deletes/` | 1048 refused calls x 0.1118 ms | **-0.12** |
| `TRIMUL_MASK_L1` | 0.8566 -> 0.6238 ms/call, Wormhole, `perf/k10_binaryng/` | **unpriceable** — 0 calls reached, so there is no rate to transfer | — |
| `QKV_MM_CONFIG` | no fold-level figure exists for the table's coverage | **unpriced** | — |
| `TRIMUL_TAIL_F1_L1_OUT` | no fold-level figure | **unpriced** | — |
| `TRIMUL_FUSED_GOUT` | — | **excluded**: the decline is correct (B2-f) | 0 |

    priced bucket-2 total     1.0352 x 1.01705 x 1.0023  ->  1.056x     UPPER BOUND, as a RATIO
    expressed in seconds ONLY for scale, against this pass's own contended 52.2 s fold:  ~-2.8 s
    against the same-board-class gap                       ~3.19x     (LEDGER R3, itself DERIVED)

**The ratio is the claim; the seconds are an illustration.** The updated brief forbids scoring
against 54.760 s and this row does not: the 1.056x is composed from three per-lever ratios, none of
which needs a Protenix anchor to state. The moment `pvx-baseline` lands a clocked, floored anchor,
multiply it through and the seconds follow. Nothing downstream should quote the -2.8 s.

**Every caveat on that number, stated rather than implied:**

1. **It is an upper bound and the terms do not add.** Stack perturbations on this fleet are
   strongly sub-additive; three levers that each delete a DRAM read of the same tensor cannot each
   delete it. The campaign standard is to approve a stack as a stack, and this screen is not an
   approval of anything.
2. **Two of the three prices are ratio transfers across board class.** The QKVG figure is Wormhole;
   the Transition figure is a Blackhole fold but a different session and a different model.
   `perf-ab-session-is-the-independent-unit` forbids comparing across sessions, and an op-level
   ratio carried to a fold has failed here four times with two levers flipping sign. Each of these
   owes a fold A/B with its own A/A floor before it is a result.
3. **A lever measured at c_z 128 says nothing about c_z 256.** The Transition raise's whole
   blocking condition is that Protenix's channel is wider. Its value at that channel could be
   larger (more bytes to keep resident) or negative (the clash the bound exists to prevent).
   Pricing it at Boltz-2's ratio is the most optimistic reading available, which is what makes it
   an upper bound.
4. **The denominator is contaminated and this row does not lean on one.** 52.193 s is this pass's
   contended fold, not an anchor, and 54.760 s is a p150a with no recorded AICLK against a p300c
   14.4 s. The screen is stated as a ratio for exactly that reason. `pvx-baseline` owns both
   anchors (LEDGER R3, and the brief's own instruction not to score against 3.4x).
5. **The two largest-count bucket-2 rows are the two that cannot be priced.** 3784 `_MM_BLOCK`
   misses and 1208 refused K3 fusions are the biggest call counts in the table and neither has a
   fold figure. The screen is therefore an underestimate in coverage and an overestimate in
   transfer, and the honest summary is that **it sizes the campaign at "worth a pass, not worth a
   quarter"** rather than at any particular number.

### 4.1 What the screen means for the campaign's shape

The design doc put Protenix in world 3 and it is right about the magnitude: 1.06x of recoverable
eligibility against a 3.19x gap leaves the gap essentially where it was, and
`pvx-protenix-specific`'s surplus arithmetic (5.545x the trunk work in 3.149x the wall) remains the
campaign's central result. What changes is that **world 2 is not empty and is not cheap to
dismiss** — there are seven real, countable, default-ON levers that Protenix pays for and does not
get, and `pvx-eligibility` now has a ranked list instead of a hypothesis.

Rank for `pvx-eligibility`, by count x plausibility of a unified fix:

1. **`_MM_BLOCK` coverage** (3784 misses, five readers downstream). Fix the table, not the models.
2. **K3 / `TRIATT_FUSED_QKVG`** (1208 refusals). Likely unblocked by 1; read `QKVG_REJECTS` after
   any table change rather than assuming.
3. **`TT_BIO_TRANSITION_L1_ROWS`** (526 refusals). LEDGER N1 step 3 stands: a per-core budget
   measured at c=256 with the real core count, and the OpenDDE capacity leg green, before the bound
   moves. Do not raise the constant.
4. **`TT_BIO_RESIDUAL_L1`** (1048 refusals). A capacity threshold, so the same measurement
   discipline as 3.
5. **`TRIMUL_MASK_L1`** (0 reached). Structural, not a gate. Find the path first.

---

## 5. Instrument, and what it cost to make it trustworthy

`scripts/lever_census.py` is the shipped census and it was the right instrument; it was also
**20 levers short of the tree**. Every one of those 20 already kept a `[served, declined]` counter
next to its guard, so the gap was a registry gap and not a measurement gap — and a registry gap
reads exactly like a lever that does not exist. Registering them is the durable part of this pass:
`state/pvx/LEDGER.md` and every future size-ladder exemption now read a 58-row table.

Two defects found and fixed in the census itself:

* **`TRIATT_FUSED_QKVG` and `QKVGB` were pointed at `triatt_qkv.REJECTS`**, which belongs to the
  head-major lever, so the first pass reported the head-major lever's `no_mm_config` clause against
  a fusion that was never offered. Both now read their own `QKVG_REJECTS` / `QKVGB_REJECTS`. This
  is the misattribution the table's own comment on `REBLOCK_PERMUTE` warns about, made twice.
* **The AICLK reader used a tt-smi key that moved** (`chip_telemetry` -> `telemetry`) and recorded
  `"unreadable"` on every sample of the first Boltz-2 pass. That pass is banked as
  `firing_boltz2_512_qb1c0_pass1_noclock.json` and is explicitly a count artifact, not a perf one.
  A clock that reads "unreadable" is not a clock.

`perf/pvx_inventory/firing.py` is the runner: one model, one process, one cold fold, every counter
zeroed, one warm fold, counts reported from the warm one. Folding twice and resetting is what makes
a number "calls per fold" rather than "calls per process" — `B2_BIAS_SLICE_HOIST` reads 3 on both
folds rather than 6, which is the check that the reset works.

The `Transition` channel census (LEDGER N1) wraps the class from the harness rather than editing
`tenstorrent.py`, which belongs to `pvx-eligibility`.

### 5.1 File ownership, declared

This pass touched `tt_bio/eltwise_fusion.py` (three counters, no branch moved) and
`scripts/lever_census.py` (20 rows, two reject-attribution fixes). Neither appears in the ledger's
ownership table. **Claiming both for `pvx-inventory`** and flagging it to the orchestrator; if
`pvx-eligibility` needs `eltwise_fusion.py`, it is theirs and this row rebases.

Everything is on `wk/pvx-inventory` and pushed. **Nothing is merged and nothing here proposes a
merge.** The counter additions are behaviour-neutral by construction and would be safe on main, but
that is `pvx-land`'s call and its gate, not this row's.

---

SHIPPED: 0.0000 s. This row measures; it lands nothing and proposes no default flip. The honest
answer for an inventory row is zero and the campaign's named failure mode is pretending otherwise.

VERDICT: PARTIAL — bucket 2 is populated and counted, seven levers with their blocking clauses
named, LEDGER N1 confirmed at 526 of 636 refusals; the SCREEN prices it at an upper bound of 1.056x
and two of the seven cannot be priced at all, so the campaign's magnitude conclusion (world 3)
survives while its count conclusion does not. `pvx-eligibility` is unblocked with a ranked list.
