# The unowned layout block, triaged against the shipped source. Three sub-leads close; one survives.

`layout_screen.py` sized an unowned layout block at **1.2596 s (ten ops) to 1.5603 s (with the head
split)**, 2.5-4.9x the campaign's remaining gap to 12.5 s, and named its own blocker: *"whether any of
these transposes is CONTRACTUALLY REQUIRED by the layout its consumer demands is a source question
nobody has asked."* That question needs no card. Asked here, against `tt_bio/tenstorrent.py` and
`tt_bio/reblock_permute.py` on this branch. **Everything below is a source reading, not a
measurement** — each one names what a row would have to confirm.

## CLOSED 1 — widening `reblock_permute.eligible` is worth 0.000 s at 512 aa

`_channel_move_back` falls back to `transpose(1,2)` then `transpose(2,3)` — two full passes — when
outside `eligible_back`, so "widen the window" looks like a lever. `eligible()`'s own docstring
already measured it and reverted it:

> the widening is worth 0.000 s/fold at 512 aa (the fit test already routes the pair tensor to DRAM,
> where the leg is open, so **52224 of 52224 moves were already served and 0 declined**)

At 512 aa the custom kernel serves every channel move. The chained-transpose fallback does not fire,
so it contributes none of the layout block. The upper edge (352, qb2-evidenced to 544 but reverted on
non-reproducing qb1 data) is a cross-grid question, not a 512 aa second.

## CLOSED 2 — the trimul's decompose transpose is already deferred into the matmul, on every call

`_transform_chunk`'s `decompose` path appends `(_channel_move,)` then `(ttnn.transpose, -2, -1)`, and
`defer_transpose` pushes that inner swap into the matmul's own `transpose_a`/`transpose_b`, which
*"on the DRAM route deletes the separate `ttnn.transpose` below outright"*. So: is it on?

    TRIMUL_MM_TRANSPOSE = True                      # tenstorrent.py:4004, default ON
    _TRIMUL_MM_TRANSPOSE = env_flag("TT_BIO_TRIMUL_MM_TRANSPOSE", TRIMUL_MM_TRANSPOSE)

and the per-branch census exists precisely because *"a dead flag and a correct layout change look
identical from the outside"*, and it *"showed the deferral **already reaches every call at 512 aa**"*.
The `ending` variant needs no separate transpose either — it swaps the permute dims in place
(`perm_a = (0, 3) + ((2, 1) if self.ending else (1, 2))`). Measured cost of the deferral itself:
`transpose(b) + matmul` 1.2463 ms -> `transpose_b=True` 0.9421 ms, **1.3228x**, the matmul paying
0.8490 -> 0.9421 ms to re-read tile-transposed while a whole program and a 67.1 MB allocation go.
Not a lever: landed.

## CLOSED 3 — the 0.2850 s `1x512x512x128` transpose is already on its best route, and 46.5 % is structural

This is the screen's largest sized sub-lead: 46.5 % of the 1R+1W roof against a sibling shape at
95.3 %, *"up to 0.1525 s if it reached the rate its sibling shape demonstrably reaches"*. The source
identifies it as the **ending variant's second transpose**, on the module's output, consumed by the
caller's `ttnn.add_` into the pair tensor. `_transpose_memory_config`'s docstring is unusually
explicit about why it is slow, and it is not a tuning miss:

> ttnn's dim0/dim1 permute is a **real element transpose, not a tile-block copy**: tiling covers the
> last two dims, so swapping the untiled batch dim with the tile-row dim moves single rows between
> tiles. Its writes are therefore **row-granular scatter**, and DRAM punishes that.

Measured there at the 298 aa pair shape: **1.479 ms to DRAM = 70.9 GB/s against 0.281 ms for a plain
`ttnn.clone` = 373.3 GB/s, i.e. 19 % of the copy roof**; into L1 the same permute is 0.600 ms,
**2.47x**. And *"`ttnn.permute`, `ttnn.transpose(0,1)` and the 4-D `(0,2,1,3)` form all land on the
same kernel and the same 1.48 ms, so **this is the only lever short of a new kernel**."*

**So the sibling-shape comparison the screen rested on is not valid.** `1x512x1024x512` at 95.3 % is
a last-two-dims transpose, which tiling covers; `1x512x512x128` is the dim0/dim1 swap, which it does
not. They are different kernels with different achievable rates, which is exactly the screen's own
caveat 1 ("a transpose's achievable rate is ACCESS-PATTERN dependent... The 46 % may be intrinsic")
resolved in favour of intrinsic.

**And the L1 lever is already taken for Boltz-2 at 512 aa, for free.** `_transpose_memory_config`
tries `_l1_memory_config_if_it_fits(t, _TRANSPOSE_L1_HEADROOM=1.25)` first and returns L1 whenever it
fits, *before* any `reserve_per_core` is consulted. The 512 aa pair tensor is
512x512x128x2 B = **67.1 MB**, needing 83.9 MB at 1.25x headroom, against the documented edge of
"budget/1.25 ... 118-147 MB" on an 11x10 Blackhole grid. 67.1 MB is well under it, so it takes the L1
route without `transpose_l1_reserve`, which Boltz-2 leaves at its 0 default. The screen's measured
46.5 % of roof corroborates: it is 2.4x the 19 % the DRAM route reads, in line with the 2.47x the L1
route was measured to buy.

**What a row would have to confirm** (this is the inference in this document most likely to be
wrong): that the 512 aa ending transpose really is on the L1 route in the fold. Read it off the
executed graph, not off this arithmetic — `l1_stage_reserve` at `:7272` and the per-shape-class DRAM
fallback in `_pair_transpose` are both live paths, and `budget` is grid-derived, so a
`l1-budget-derived-from-live-grid-makes-output-host-dependent` hazard applies.

## SURVIVES — the one remaining mechanism, and it is continuous with `c12-reblock-delete`

What is left after the three closures is **not a config or a deletion but a kernel**: there is no
hand-written mover for the **dim0/dim1 swap**, while `reblock_permute` is exactly that for the
`(0,3,1,2)` / `(0,2,3,1)` moves, and it wins by *"issuing the 64 unavoidable NOC transactions per
source tile from 100 cores"*. The dim0/dim1 swap's row-granular scatter is the same class of problem
that kernel already solves in the other axis order.

    ending output transpose   <= 0.2850 s   dim0/dim1, 46.5 % of roof, L1 route, needs a new kernel
    1x512x1024x512            <= 0.1712 s   95.3 % of roof -> AT the roof, deletion only, no rate lever
    remaining eight ops         ~0.6034 s   unscreened by shape

Two cautions before anyone sizes that. `c12-genop-triatt-slack` measured `reblock_back` — the
closest existing analogue — as **22.5-30.9 % width-independent transaction-issue cost, 7.9-10.8
cycles per 32-byte gather read, structural at 64 transactions per output tile**. A new kernel for a
row-granular scatter inherits that floor rather than escaping it, so the prize is a fraction of
0.2850 s, not 0.2850 s. And every rate-gap lever this campaign has sized from a roof ratio has
collapsed (kblock 0.3423 -> 0.035 s, cross-family 0.9454 -> 0.081 s, genop slack 0.4544 -> 0.0184 s),
because **a resource gap is not headroom**. This should be entered, if at all, *after*
`c12-reblock-delete` reports, since that row is building the same mechanism at 3.5x the size and its
result is the only honest prior for this one.
