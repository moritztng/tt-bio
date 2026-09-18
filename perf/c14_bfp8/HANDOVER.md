# Handover: the triangle-attention bfp8 region, for the rows that now own the files

Written by `c14-bfp8-fastpath` after the 15:4xZ boundary re-cut moved `tt_bio/**` out of this row's
slice. The region was built and measured before the re-cut; the production edits have been reverted
off `wk/c14-bfp8-fastpath` so a merge of this branch carries no production diff. The exact patch is
`HANDOVER_region_patch.diff` in this directory (324 lines, against `c294b2573`). Apply the hunks
that belong to your slice; do not take the whole file.

## The measurement that matters most, and it is the answer to the premise of the whole campaign

**Nothing fell back.** `scripts/lever_census.py` around a real 298 aa `cdk2x2` fold, flag off vs
flag on, **0 of 38 levers changed served-vs-declined**. Every fast path that served 560 of 560 calls
with bf16 operands still served 560 of 560 with bfp8 q/k/v/gate/bias. Positive control on the same
pair, so this is not a flag that quietly did nothing: the two arms wrote **different structures**,
CIF sha256 `73f152558f7fddc8` (off) against `9e94a5de94003d42` (on).

So the recorded cause of the old 0.949x result — "the block format disqualifies fused kernels" — is
a property of the Python gates and not of the kernels. With the gates widened the kernels take the
format and keep serving.

## Eligibility by firing retires three of the five gates before anyone builds on them

Same census, decline reasons read off the levers' own counters:

| lever | file | served/declined at 298 aa | the reason it declines |
|---|---|---|---|
| `TRIATT_PERSISTENT_MASK` (fused SDPA) | `triatt_sdpa.py:338` | **560 / 0** | serves; reachable |
| `TRIATT_HEAD_MAJOR_QKV` | `triatt_qkv.py:53` | **560 / 0** | serves; reachable |
| `TRIATT_HEAD_MAJOR_TAIL` | `triatt_qkv.py` (`out_proj`) | **560 / 0** | serves; reachable |
| `TRIMUL_IN_PROJ_DUAL_NOC` | `mm_dualnoc.py:86` | 0 / **2240** at 298 aa but **560 / 0** at 512 aa | `memory` at 298 aa; **serves every call at the cell** |
| `TRIMUL_TAIL_F1` | `trimul_tail.py:128` | 0 / **560** | **`k_tiles=4`**, 560 of 560 — c_z = 128 gives kt = 4 and the fork wants 8 |
| `PAIR_TRANSPOSE_VIA_ROW_MAJOR` | `reblock_permute` | 0 / **560** | `l1_dest_is_faster:320x320x128` — a deliberate choice, not a dtype |

Two consequences for the re-cut, offered rather than acted on:

* **`mm_dualnoc.py:86` is NOT a null — widen it.** The 298 aa census alone would have retired it:
  0 served, 2240 declined, every one on `memory`, never reaching the dtype clause. At **512 aa it
  serves 560 of 560**. `_triangle_mul_memory_config(H)` is H-dependent and the decline does not
  survive to the cell, so a row that screened this gate at 298 aa only would have dropped a
  reachable site. That is the same shape of error as scoring a precision change at 298 aa: the
  control size does not cover the cell, in either direction.
* **`trimul_tail.py` and `tt_bio/tenstorrent.py` have no owner in the re-cut table**, which is
  therefore not gap-free over our bf16 gates. The good news is the gap is worthless: `trimul_tail`
  declines 560 of 560 on `k_tiles`, the same structural reason that made the ~1.17x once credited to
  widening `_pair_proj_minimal_matmul` a phantom. `tenstorrent.py` matters for a different reason —
  it is where a region FLAG has to live, and `_triatt_dtype()` in the patch is that flag.

## The region, and why it is shaped this way

`TT_BIO_TRIATT_B8`, default off. Under it the fused qkv+gate(+bias) pass writes its five
destinations in bfp8, the fused SDPA's destination follows `q.dtype`, the gate `multiply_` runs
bfp8 x bfp8, and the out projection keeps `_dtype()` so **`z_update` lands in bf16**.

* **No `typecast` anywhere in the region.** Every producer in it is a matmul whose destination format
  is a program argument, and `mm_generic.build`'s accumulator is a separate `interm_fmt` (fp32 under
  `fp32_dest_acc_en`), so the narrowing is one rounding at the pack stage and touches no
  contraction. This is why the region pays where a per-op cast cannot: a cast costs 0.2577 ms at this
  key to buy a 0.166 ms saving.
* **`z`, every `z_update`, the stored weights and the normed pair tensor stay bf16 on purpose.** bfp8
  fails on this model when it quantises an ACCUMULATION. Five residual sites a layer through 64
  layers and 3 recycles is what turned `_TRIATT_BIAS_B8`'s op-level 0.002547 into a block-level
  0.04179, and what produced the 1.4965 A of the whole-track arm and the 13.21 A of b2z's
  `transition` site. The region never touches one.
* `ttnn.layer_norm` has **no output-dtype argument** and returns the input's format, so the normed
  pair tensor cannot be narrowed without a cast. Rejected on paper with measured rates.
* One conservatism left in: `_fused_qkvgb` still asks `_qkv_l1_config` with `_dtype()`, so the L1
  budget is computed for bf16 while the destination is narrower. That can only refuse an L1 path
  that would have fit, never admit one that does not.

## What the patch actually changes, per slice

    tt_bio/mm_generic.py    ALREADY LANDED as 44bf24365 on wk/bfp8-orchestrator (bfloat8_b: 1088).
                            The patch's own version is identical in value; take the orchestrator's.
                            The patch additionally adds FAST_DTYPES + fast_dtypes_ok, the one
                            predicate the five gates share. That is the piece with two would-be
                            authors -- it belongs wherever tile_bytes went.
    tt_bio/triatt_sdpa.py   -> bfp8-sdpa-unlock. Three hunks: :338 q/k/v/bias, :398 the gate
                            operand (compare against q.dtype, not a pinned bf16), :481 _FUSE_QKV,
                            plus the two allocate_tensor_on_device destinations that were pinned
                            bf16 and now follow q.dtype / x.dtype.
    tt_bio/triatt_qkv.py    -> bfp8-qkv-matmul. _common_ok, and FIVE destination allocations that
                            were pinned bf16 and now follow the `dtype` argument. Note qkvgb_heads
                            must write ONE format across all five outputs: generic_minimal_matmul
                            sizes the output CB from outs[0], so a bias destination of a different
                            format would be written through the wrong page size.
    tt_bio/mm_dualnoc.py    -> bfp8-qkv-matmul. One clause. Measured null, see above.
    tt_bio/trimul_tail.py   -> unowned. Membership widened only, pair tests untouched. Measured
                            null, see above.
    tt_bio/tenstorrent.py   -> unowned. _TRIATT_B8, _triatt_dtype(), the bias travelling with
                            q/k/v, and the six producer sites that pass _triatt_dtype() instead of
                            _dtype(). `_dtype()` MUST stay bf16: TriangleAttention.__init__ only
                            concatenates the fused qkv+gate weight when `_dtype() == bfloat16`, so
                            a mode flag disables the very kernel the region exists to feed.

## One source reading of mine to distrust, and one correction to a correction

I wrote in pass 1 that `sdpa_generic._uniform_dataformat` is "the kernel's constraint, not ours".
**It is not a constraint at all** — `sdpa_generic.py:440` passes it as the `is_uniform_dataformat`
compile-time arg, a hint the kernel takes either way. That is what lets the region exit in bf16 at
the out projection instead of carrying block float across the residual, so it matters.

And the orchestrator's correction is right that `tile_bytes` **raised `ValueError`** on bfp8 rather
than being generic; my pass-1 table named the missing `_TILE_BYTES` entry as a blocker but my prose
around it said "the machinery underneath is already dtype-generic", which overstates it. Parameterised
but floored on bf16/fp32 is the accurate description.


## The accuracy result, which is what the region dies of

Measured after the re-cut, on the structures the two censuses wrote, all-atom Kabsch
(`perf/other512/cif_rmsd.py`, the lineage's own scorer):

| | off vs on | A/A control | plDDT off -> on | vs the bar |
|---|---|---|---|---|
| 298 aa | **0.602376 A** | — | 0.897201 -> 0.893966 | at/over the 0.60 A hold line |
| 512 aa | **1.216172 A** | **0.000000 A** exactly (two independent base processes, bit-identical CIFs) | 0.850185 -> 0.844167 | **2.03x over the 0.60 A kill bar** |

Context the standing policy requires beside it: the 512 aa **seed floor is 1.82527 A** (the reference
compared to itself with only the noise realisation changed, `perf/roof_shared`), so 1.216172 A is
**0.67x of variation already accepted** — but the bar is 0.60 A and this is over it, so the region
does not ship as scoped. For proportion: full-track bfp8 read **1.4965 A at 298 aa** where this reads
0.602376, and b2z's union read **13.07 A at 512 aa**. plDDT FALLS slightly here rather than rising,
which is the tell that this is the same fold perturbed and not the different global arrangement that
b2z's `transition` site produced.

**The obvious next narrowing, for whoever owns the destination.** The region currently carries q, k,
v, gate, bias AND the SDPA's own output in bfp8. The output is the one rounding that lands directly
on the path to `z_update`; q/k/v feed a softmax that attenuates. Splitting them — q/k/v/bias in bfp8,
the SDPA destination back to bf16 — is legal precisely because `_uniform_dataformat` is a hint and
not a precondition, and it keeps three of the four pair tensors in the 268.4 MB buffer narrow. That
arm is one line in `triatt_sdpa.py` (the destination presently follows `q.dtype`) and belongs to
`bfp8-sdpa-unlock` with `bfp8-accuracy-envelope` scoring it. This row cannot run it without a re-cut.

## What is still unmeasured, and it is the timed arm

No trustworthy fold A/B of the region exists. The 298 aa session read **1.01174x against its own A/A
floor of 1.00834** — a null inside noise, and doubly unusable: the box was co-tenanted (benchlock
recorded `WARNING after 300s load=1.82 foreign_folds=1`) and the clock was not pinned, sampling
1312-1350 MHz. The 512 aa timed arm was deliberately not run, because a release gate has been folding
since 12:14Z. Priced instead from this row's own measured realization and byte census, as a **band and
not a book entry**: the two 268.4 MB qkv buffers are 15.0 Z a layer, 265.7 GB a fold, **0.70 s** at
the 380.77 GB/s this card measures, times the 0.84-0.98 realization = **0.59-0.68 s**. The full
ours-to-fix bucket, if every one of our gates were widened, is 29.062 Z a layer = 1.14-1.32 s.
