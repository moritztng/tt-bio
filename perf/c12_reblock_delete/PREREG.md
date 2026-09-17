# c12-reblock-delete: pre-registration, written before the first arm

Clock: every number below is at **1350 MHz**, forced and sampled during the fold. The in-situ
source table is `perf/c12_genop_rate/headroom.json` @ `4ed84475d`. Part: qb2 p300c Blackhole,
11x10 = 110 cores. Reproduce with `python3 perf/c12_reblock_delete/sites.py` (CPU only).

## What the two sites actually are

    site            calls   ms/call   fold s    Mc   floor ms   % of own roof
    reblock_gated    1120    0.6448   0.7225   975.4   0.5119      79.39
    reblock_back      560    0.4941   0.2775   374.6   0.3414      69.10

Executed graph, not a census label: 14 `generic_op` programs per PairformerLayer / MSALayer,
264 + 16 layer invocations, 3,920 calls total, identified by `COMPUTE KERNEL SOURCE`
(`perf/c12_genop_rate/insitu_sites.py`). Per trimul call that is **1 in-projection matmul,
2 gated moves, 1 back move**, and 560 trimul calls reproduces 1120 and 560 exactly.

Shapes at 512 aa, from the channel-loop plan (`sites.py`): chunk 32, group 4, `slice_c` 128, one
channel-loop iteration, fused projection 512 channels wide.

  * `reblock_gated` issues `[1,512,512,512] -> [1,128,512,512]` twice, reading a 128-channel value
    slice and a 128-channel gate slice out of the fused projection.
  * `reblock_back` issues `[1,128,512,512] -> [1,512,512,128]` once.

**The "zero FLOP" label is half wrong and the brief is right to correct it.**
`compute_reblock_permute_gated.cpp:5` computes `out = transpose_wh(p * sigmoid(g))`; only
`reblock_back` is arithmetic-free. So on 0.7225 s nothing is being deleted - the sigmoid and the
multiply still have to run. What is deleted is a DRAM round trip.

## Mechanism: the producer's writer, not the move's reader

`file:line`, all of it:

  * The gated move's input is written by `_in_proj_matmul` (`tt_bio/tenstorrent.py:5653`), which
    routes to `mm_dualnoc.in_proj` (`tt_bio/mm_dualnoc.py:71`) and thence to
    `mm_generic.build` (`tt_bio/mm_generic.py:122`). Its two **dataflow kernels are tt-bio's own**
    (`tt_bio/kernels/mm_split/dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`), generated from the
    wheel's `minimal_matmul` kernels by `tt_bio/kernels/mm_split/patch_mm_split.py`. The output
    writer is therefore already ours.
  * The gated move itself is `_transform_chunk_gated` (`:6161`) -> `reblock_permute_gated`
    (`tt_bio/reblock_permute.py:785`).
  * The back move's input is written by `ttnn.matmul` (`tt_bio/tenstorrent.py:6572`) under
    `_triangle_mul_program_config`. That writer is **not ours**.

**The lever.** Have the in-projection write `a` and `b` directly, gated and reblocked, and the two
`reblock_permute_gated` programs cease to exist. Three pieces, each already written somewhere in
the tree:

1. *Gating must be core-local.* `mm_generic.build:207` gives each core `N_tiles_per_core = 2` of
   the 16 output channel-tiles. Under the shipped role order `(p_a, g_a, p_b, g_b)`
   (`tenstorrent.py:5891`) each role is 4 consecutive tiles, so core 0 holds two `p_a` tiles and
   no gate - **no core holds a (p, g) pair**. Re-laying the fused weight as tile-interleaved
   `(p t0, g t0, p t1, g t1, ...)` puts exactly one value tile and its gate tile on each of the
   8 N-cores that carry real output. The weight is laid out once at load and every consumer reads
   its order from `gp_roles()` alone, which is the mechanism `TRIMUL_GP_BANK_SPLIT` (`:5890`)
   already uses, so this costs nothing at runtime.
2. *The arithmetic is already written.* `compute_reblock_permute_gated.cpp` is the sigmoid, the
   SFPU multiply and the `transpose_wh`, in that order, with its rounding points documented.
3. *The scatter is already written.* `reblock_permute`'s writer assembles a destination tile from
   a group of `GROUP_TILES = 32` source tiles sharing one j-tile (`reblock_permute.py:38`). The
   matmul's M assignment has to change from contiguous runs to those groups: at 512 aa there are
   **256 groups over 11 M-axis cores, 23-24 each, a 3.1 % imbalance**, and groups taken I-major
   keep each core's activation range contiguous, so `in0` is still read once and multicast once.

**Predicted rate, not a call count.** Per trimul call, in units of `Z = H*H*slice_c*2 =
67.1089 MB`:

    today   in-projection   read in0 1 Z, write the fused projection 4 Z          5 Z
            gated move x2   read p 1 Z + g 1 Z, write 1 Z, twice                  6 Z
            total                                                                11 Z = 738.2 MB
    fused   read in0 1 Z, write a 1 Z, write b 1 Z                                3 Z = 201.3 MB

8 Z per trimul deleted. **Disclosed disagreement:** the inherited `genop_audit` books the
in-projection at 402.9 MB/call, which is 6 Z, where the kernel's own addressing above gives 5 Z -
`in0` is read once by one sender core per M index and multicast down the column
(`mm_generic.py:252`), and the output is 4 Z. `headroom.json`'s `floor_ms = 0.9246` and the 61.9 %
efficiency the central arm borrows are computed from the audit's 6 Z, so if 5 Z is the true count
the in-projection's real efficiency is 51.6 % and the central arm is optimistic by that much
(0.8264 -> 0.9920 ms/call, saving 1.0062 s instead of 1.0990 s). It does not cross the kill bar
either way, and arm 1 resolves it directly by measuring the fused op's own bytes. The fused op's
traffic floor is **0.5119 ms/call** at the measured 393.3 GB/s 1r1w roof - numerically the same floor `reblock_gated` already runs against, because
both move 3 Z. Arithmetic intensity rises from 106.6 to ~160 FLOP/byte against a measured
260.9 FLOP/byte machine balance, so the op stays traffic-bound, but the margin over the matmul's
own 0.3778 ms of arithmetic falls from 2.04x to 1.36x and the gate's SFPU work now has to hide
under a smaller traffic time. That is the risk the band below brackets.

## PREDICTED, before the first arm

Today: `trimul_in` 0.8392 s + `reblock_gated` 0.7225 s = **1.5617 s / 2108.3 Mc** at 1350 MHz.

    arm           ms/call   fold s   fold Mc   saves s   saves Mc   assumption
    optimistic     0.6448   0.3611     487.4    1.2007     1620.9   hits reblock_gated's 79.4 %
    central        0.8264   0.4628     624.7    1.0990     1483.6   hits trimul_in's 61.9 %
    pessimistic    1.1206   0.6276     847.2    0.9342     1261.1   gate stops hiding

**Central prediction: 1.0990 s / 1483.6 Mc off the 512 aa fold.** This is larger than the brief's
1.0000 s because it also eats part of `trimul_in`, and it leaves `reblock_back`'s 0.2775 s /
374.6 Mc untouched. 1.0000 s remains the upper bound for *deleting the two reblock sites*; it is
not this lever's ceiling and it is not its target.

**KILL CRITERIA.**

  * Arm 1, op level, before any fold arm: kill if the fused op's in-situ time exceeds
    **1.1206 ms/call**, the pessimistic edge. No tuning pass to rescue it.
  * Arm 2, fold level: kill if the interleaved benchlocked 512 aa delta is under **0.550 s
    (742.5 Mc)**, half the central prediction.
  * Any delta under 3x the session's own A/A floor is not a result and is reported as such.

## Why this is not step 2, stated as an accounting rather than an assertion

Step 2 lost because it deleted 8 Z of DRAM traffic and bought back L1 traffic no DRAM byte counter
sees. So the discriminator for this lever is its ON-CHIP accounting, not its DRAM accounting:

    stage                        today                        fused
    matmul out CB fill/drain     M_block x N_block tiles      same
    gate: sigmoid, mul, WH       3 CB round trips per tile    same, same kernel body
    writer gather                2048 local transactions      same, same writer
                                 per 32-tile group
    DRAM in between              4 Z write + 4 Z read         none

Every on-chip stage is the same stage, in the same order, through the same circular buffers. The
fused op does not add staging traffic; it removes a DRAM round trip from between two stages that
already exist. Step 2 by contrast KEPT both programs and added `ttnn.slice` copies of the LN'd
pair tensor (`ttnn.slice` is a copy, not a view) on top. That is the whole difference, and it is
the reason a +0.907 ms result does not transfer.

The one genuinely new cost is L1 residency: the writer needs 32 gated tiles staged before it can
emit a destination tile, and the matmul's `M_block_tiles = 8` emits 8 at a time, so 4 output
blocks have to be held. 32 gated + 32 destination tiles is **131 kB per core** on top of the
matmul's own circular buffers - the same order as the 148 kB `reblock_permute_gated` already
carries. That is the number arm 1b has to survive, and it is why the row is built in two arms
rather than one.

## BUILD, in two arms, because the second one is the risky half

**Arm 1a - the gate epilogue only.** The matmul writes `p * sigmoid(g)` into `a` and `b` in the
ORIGINAL `[1,H,H,slice_c]` layout. Its writer keeps its normal tile order and only the destination
addressing changes, which is the machinery `MM_SPLIT_LAST_TILES` and `write_tile_to_chunk` already
carry. The two plain forward moves then run as today's `reblock_permute` - arithmetic-free and
bit-exact by construction. Traffic 3 Z + 2 x 2 Z = **7 Z**, deleting 4 of the 8.

    arm            ms/call   fold s   fold Mc   saves s   saves Mc
    optimistic      1.5044   0.8425    1137.4    0.7192      971.0
    central         1.8145   1.0161    1371.8    0.5456      736.5
    pessimistic     2.1088   1.1809    1594.2    0.3808      514.1

The plain move's cost is **estimated**, not measured: it is taken from `reblock_back`'s measured
0.4941 ms/call, which moves the same 2 Z between the same two shapes in the inverse direction.
The forward move at this shape has never been timed, because the shipped path uses the gated one.
Arm 1a's own kill bar is **0.300 s / 405 Mc** - below that it is not worth keeping as a standalone
even if arm 1b then fails.

**Arm 1b - the reblock in the writer.** Adds the 32-tile group M assignment and the gather, taking
traffic to 3 Z and the prediction to the table above.

Both arms need the same two things first, and both are inert on their own:

  * the **tile-interleaved fused weight order** `(p t0, g t0, p t1, g t1, ...)`, because at
    `N_tiles_per_core = 2` the shipped role-major order puts no (p, g) pair on one core. A pure
    column permutation laid out once at load, the same mechanism `TRIMUL_GP_BANK_SPLIT` uses.
  * a **compute-kernel source override** in `mm_generic`, since `compute_src` was pinned to the
    wheel's own `compute.cpp`. LANDED this pass as `compute_dir=None`, threaded through `build`,
    `_key` and `generic_minimal_matmul`; `None` is the wheel's kernel so every existing caller is
    byte-identical. It is in the CACHE KEY as well as the build, because two arms of one A/B
    session differing only in their compute kernel would otherwise share the first arm's
    descriptor, and the arms of a perf comparison have to interleave inside one process.

## BUILD: the wheel, and that is why this route was chosen

`patch_mm_split.py` already generates tt-bio's dataflow kernels from the wheel's own
`minimal_matmul` sources by exact-match patching, and `ttnn.generic_op` JIT-compiles them against
the shipped wheel. `MM_DUAL_NOC` shipped that way. The one extension needed is a **patched compute
kernel**: `mm_generic.py:258` pins `compute_src` to the wheel's own `compute.cpp` ("never patched"),
so hosting the gate epilogue needs that path to become overridable the same way `kernel_dir`
already overrides the dataflow pair. That is a tt-bio change, not a tt-metal one.

**So no tt-metal source build, and the 1.115x wheel-vs-source cross-build control is not owed.**
If the epilogue ever does need a source build, that control comes first: `c12-genericop-rate`
measured 1.115x on a 2048 cube, stable to 1.7 % over three sessions, and missed its own
pre-registered band because of the build rather than the part.

`reblock_back` has **no wheel route**: its producer is `ttnn.matmul`, whose writer lives in
`MatmulDeviceOperation`. Re-hosting that matmul in `mm_generic` first is a larger build than the
0.2775 s it would unlock, so this row prices it and does not build it.

## SIZES: this is a 512/768 aa lever and it is worth nothing at 298 aa

From `sites.py`, on the 11x10 grid:

    aa    H  path  chunk group slice_c iters  gated/tri  plainfwd/tri  back/tri  branch
    298  320  L1      32     1      32     4          0             4         0  four-way-split + plain move
    512  512  DRAM    32     4     128     1          2             0         1  gated-move
    768  768  DRAM    32     4     128     1          2             0         1  gated-move

298 aa takes the **L1** channel-loop path (`TRIANGLE_MULT_L1_MAX_SEQ = 352`, `tenstorrent.py:580`),
and both `eligible_gated` and `__call__`'s own `gated` condition require a DRAM channel-loop
memory config. So **`reblock_gated` serves zero calls at 298 aa** and `reblock_back` serves zero
(`eligible_back` is DRAM-only, `reblock_permute.py:579`). The lever is worth **0.0000 s** there.
That is not a gap in the lever, it is a free negative control: 298 aa must come out bit-exact,
because the fused kernel takes no call at that size.

768 aa is structurally identical to 512 aa: chunk 32, group 4, `slice_c` 128, one iteration.
Same two gated calls and one back call per trimul. Its call count still has to be read off an
executed graph on the measurement arm - the 512 aa counts above are measured, the 768 aa ones are
derived from the plan.

## The epilogue is expressible, checked in the kernel rather than assumed

Read at `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp` in the
pc source checkout. That checkout is **v0.72.0-dev** and the wheel is **v0.68.0**, so this is a
structural reading, not the bytes the generator will patch - `patch_mm_split.py`'s exact-match
asserts are what pin the anchors to the wheel's own copy, and they fail loudly on drift.

  * The output stage is `copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles)`, a
    plain (m, n) walk that copies each accumulated tile into DST, optionally applies
    `SFPU_OP_FUNC_ACTIVATION`, and packs to `out_cb`. **There is already an SFPU hook in the
    output stage.** A unary macro cannot express a gate, but this loop is the right place and the
    edit is local to it.
  * The exact idiom the gate needs is **already in this file**: `add_bias_and_addcmul_block`'s
    `TERNARY_B_IS_FLOAT32` branch does two `copy_tile`s into two DST slots and then
    `mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID)`, which is
    `compute_reblock_permute_gated.cpp`'s stage 2 with the sigmoid removed.
  * **Plumbing trap worth stating before the build:** gating halves the tiles the output stage
    pushes per row, and the writer's `write_block_sync_granular` pops `N_block_tiles` from the
    out CB. The gated width has to be halved consistently in the writer's compile-time args and
    in the destination tensor's N, or the writer walks off the end and writes plausible garbage.

## The precision fact this reading turned up, which changes the accuracy pre-registration

`mm_generic.py` sizes `intermediate_cb` as `float32 if fp32_dest_acc_en else bfloat16`, and the
in-projection runs under `fp32_dest_acc_en=True`. `copy_block` therefore reads **fp32**
accumulated tiles and packs them to bf16 on the way to `out_cb`.

So a gate placed in that stage would read **fp32** `p` and `g`, where today's path packs the
projection to bf16 into DRAM and the gated kernel reads bf16 operands. The digest moves for three
reasons, not one, and the operand precision is the largest of them:

    1. operands carry fp32 mantissa instead of bf16   <- new, and the biggest
    2. `calculate_sigmoid` takes its accurate branch under the flag (10.4 % of elements, 1 ulp)
    3. packer ties break away from zero rather than to even (0.91 % at a 1.85 % tie rate)

All three push toward *better* float64 accuracy, which is precisely the
`c12-fused-eltwise-at-pin` failure signature - a transform more accurate than what it replaced
still failed. So the default is the matched-rounding variant:

  * **Variant B, the default.** Pack `intermediate_cb` to a bf16 CB first - which is exactly what
    `copy_block` already does into `out_cb` - and gate from that bf16 CB. Today's operand
    precision is reproduced and the exposure narrows to item 2 alone. Costs one extra CB round
    trip per tile and no DRAM byte.
  * **Variant A, measured against it.** Gate straight off the fp32 `intermediate_cb`. Cheaper by
    that round trip, more accurate against float64, and a larger digest move.

Pre-registered: **B is what arm 1a ships if it ships**, and A is a same-session comparison, scored
on both time and Angstrom. Choosing A on its float64 accuracy alone would be the mistake
`c12-fused-eltwise-at-pin` already paid for.

Also settled by this reading: at 512 aa `K_tiles = 4` against `K_block_tiles = 8`, so
`K_blocks = 1` and the `llk_pack_reconfig_l1_acc` accumulation never folds across K blocks. There
is no contraction-order question at this shape.

## ACCURACY: pre-registered, and this lever is not bit-exact

`compute_reblock_permute_gated.cpp:11-27` is explicit that its bit-exactness against ttnn depends
on running a **16-bit DST** (`fp32_dest_acc_en=False`): `calculate_sigmoid` branches on that flag
at compile time, and the accurate branch is a full bf16 ulp from ttnn's on **10.4 % of elements**,
measured over 3.28 M. The in-projection matmul runs under `fp32_dest_acc_en=True` (`_MM_DEFAULT`
is what `determine_default_block_sizes` returns under it, `tenstorrent.py:5656`), and the flag is
per kernel, not per stage. Packing the product from an fp32 DST is wrong a third way: the packer
breaks ties away from zero where ttnn breaks to even, **0.91 % of elements at a 1.85 % tie rate**.

So the fused epilogue will move the digest, on roughly 11 % of gate elements at 1 ulp, 560 trimuls
deep. That is allowed - the standing bar is accuracy against the seed floor - but it has to be
scored, not argued:

  * Å deviation of the final structure at 512 and 768 aa, against a seed-scatter floor measured on
    **this row's own fixture and metric**, with an A/A control in the same session. `1.84 A` is the
    wrong bar and is not quoted here: `c12-fused-eltwise-at-pin` measured 5.35-9.90 A of all-atom
    seed scatter on its own fixture.
  * plDDT beside it, which carries no frame and so is not confounded by basin choice.
  * 298 aa as the negative control: the kernel takes no call there, so anything but bit-exact at
    298 aa means the change leaked outside its gate.
  * Read `c12-fused-eltwise-at-pin` as the warning it is: transforms that were *more* accurate than
    what they replaced against a float64 reference still failed, because a one-ULP touch of a
    conditioning path feeding 200 diffusion steps saturates. A trunk reblock is not that path.
    Check rather than assume.

A bit-exact variant exists if it is ever wanted: call the cheap sigmoid explicitly instead of
letting `calculate_sigmoid` pick on the flag. It is kernel surgery for a property that is not the
bar, so it is named and not built.

## What this row must not be sold as

The whole device-side book is silu 0.2843 s + cond-hoist 0.2415 s + this row's 1.0000 s upper
bound. 12.5 s needs 2.3810 s. This row is the larger half of the only surviving route to 12.5 s
and it is not a route to 10.0 s, which four independent derivations put outside the current op set.

## Refuted already, do not spend an arm on it

`TRIMUL_INPROJ_ROWBLOCK` (`tenstorrent.py:5846`, ships **off**) is the *neighbouring* lever and it
is **measured to lose**. `_gated_rowblocked` (`:6218`) projects one row block into L1 and lets the
gated move read it there, deleting the same 8 Z per trimul - and at R = 64 on the 512 aa cell it
adds 29 device ops and costs **+0.907 ms**, giving back a third of the mask-after-move win.
R = 128 collides with the gated kernel's circular buffers on the 110-core grid and R = 256 is a
flat L1 OOM.

It is a different lever from this one and its loss does not price this one. It keeps the separate
`generic_op` move and only changes where that move's **reader** gets its bytes, so the move's own
0.7225 s survives and its call count rises by `H/R`. This row deletes the move. But its cost
breakdown is the sharpest warning available: dispatch was at most two thirds of the +0.907 ms, the
rest being L1 traffic no DRAM byte counter sees. A fused epilogue pays that same on-chip staging
traffic inside one program instead of across two, which is the reason the pessimistic arm above
exists.
