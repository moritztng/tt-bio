# Three Blackhole-fitted constants, measured on the Wormhole Galaxy

`roof-wrong-part-envelopes` audited all 48 empirically-fitted constants in `tt_bio/tenstorrent.py`
and found three that were fitted on a Blackhole part but reach every part, including the Wormhole
Galaxy the live JapanFold service runs on. This directory measures all three there, plus the two
gaps that audit flagged.

Everything below is a **Wormhole** measurement on a Wormhole target, so no transfer multiplier is
applied to any of it and none of it may be quoted as a Blackhole result.

## The part and the roof

Host `j10glx02`, card 0, pinned with `TT_VISIBLE_DEVICES=0`. Compute grid 8x9 = 72 cores,
`dram_grid_size()` = (x=12, y=1) so 12 DRAM banks, L1 bank 1 395 424 B, max worker L1 unreserved
1 466 080 B.

Dense bf16 HiFi4 4096-cube, measured in the same session as every ratio here: **54.3698 TFLOP/s**
(session A) and **54.8214 TFLOP/s** (session B), against 54.2769 TFLOP/s measured independently by
`roof-shape-honest-roofs`. Reproduced to 1.0 %. Own-session A/A floor 0.596 % / 0.434 % on the cube
and 0.015 % to 0.184 % on the swept ops.

## 1. `_FP32_SOFTMAX_L1_GRID = (8, 8)` — a real defect on the production part

Its comment names the part it was fitted on: "this p150a refuses more than 110 shards". It is used
as the tuned grid on every part.

It is not accidentally safe here, because the block height is derived from the core count, so the
rectangle is a byte count per core and not just an occupancy. 64 cores asks for 786 432 B/core and
this part refuses it; 72 cores asks for 524 288 B/core and it does not.

AF2-IG triangle attention (4 heads, head_dim 32), production defaults, two interleaved sessions
with the shipped arm run twice as its own A/A:

| tokens | shipped (8,8) | live grid (9,8) | ratio A | ratio B | A/A |
|---|---|---|---|---|---|
| 256 | 9.953 ms | 9.932 ms | 1.0022x | 0.9974x | 0.049 % / 0.172 % |
| 384 | 30.682 ms | 30.661 ms | 1.0007x | 1.0000x | 0.184 % / 0.041 % |
| **512** | **109.589 ms** | **74.034 ms** | **1.4803x** | **1.4804x** | 0.017 % / 0.043 % |
| 768 | 305.414 ms | 305.433 ms | 0.9999x | 0.9996x | 0.029 % / 0.015 % |

512 aa is the one rung where the rectangle refuses. Elsewhere the floating core count already lands
on 72 and the constant is neutral inside the A/A floor, so this is a refusal, not a tuning.

Bit-exact: `torch.equal` and max_abs 0.0 at every rung and every grid.

Served, not "it did not crash": on the live grid the counters read `l1_blocks` 448 of 456 with
`l1_refused` 0, against 8 of 344 on the fitted rectangle.

The measured envelope on this part is **72 shards**, not the 110 the comment names. At 80 the
allocator refuses outright ("Expected number of shards 80 to be less than or equal to total number
of L1 banks 72 in compute cores"), so the fitted ceiling never binds here and the live grid needs
no extra clamp.

### Fold-level cost

An op ratio is not a result, so the same change was priced on a whole fold. `fold_ab.py`, ABBA
inside each rep, both arms in one process with the plan caches cleared between them, 200 sampling
steps and 3 recycles at 512 aa, two sessions:

| | session A (n=4/arm) | session B |
|---|---|---|
| fitted (8,8) | 96.784 s | see `fold_ab_512_B.json` |
| live grid (9,8) | 76.480 s | |
| **fold ratio** | **1.2655x** | |
| A/A floor | 0.043 % | |
| paired deltas | 20.019 / 20.465 / 20.165 / 20.403 s | |

Every fold in both arms wrote the same CIF (sha256 `a0aa3b72…`) and the same 0.84482 pLDDT, so the
lever is bit-exact at the fold level, not just on the op.

The fold carries `BOLTZ2_FP32_SOFTMAX=1`, and that flag is the **path**, not the arm — both arms
have it. Boltz-2 ships the fused SDPA instead, so this prices the constant on a model whose weights
are on the box. OpenFold3 (trunk, template, MSA embedder, confidence head) and AF2-IG reach
`_fp32_softmax_attention` by default, and `_TRIATT_FUSED_HIFI` defaults off, so the path is live in
production rather than a fallback. Their weights are not on this box, and that caveat travels with
the number.

### Fix

`_apply_grid_thresholds` takes the rectangle from the live grid inside its small-grid branch. Both
Blackhole parts (13x10 p150a, 11x10 p300c) return above that branch and keep the fitted value byte
for byte. `TT_BIO_FP32_SOFTMAX_L1_LIVE_GRID=0` pins the old value for an A/B.

By kind this is **eligibility** — a grid widening that changes which plans the part accepts — which
does not transfer to another part at all.

### A second-order defect, found on the way and not fixed

Once the tuned rectangle refuses, `_FP32_SOFTMAX_L1_FLOAT_CORES` (on by default) re-proposes the
rejected height, because the free search reads `_FP32_SOFTMAX_L1_FREE_ROW_CAP` while `shard_for`
tests the height against `_FP32_SOFTMAX_L1_ROW_CAP`. The call then loses L1 residency entirely
instead of stepping down one row: 331 resident blocks with the float off against 9 with it on,
94.481 ms against 109.589 ms. That is why the production default was 0.862x its own float-off
configuration on this part. The fix above removes the trigger (`l1_refused` goes to 0), but the
mismatched cap is still there for the next part that refuses.

## 2. `_PAIR_FFN_FC1_BLOCK_W = 16` — accidentally safe, and dark at three of four sizes

Swept on qb2 card 2 (p300c, 11x10) with a documented clash one rung up at `obw = 32`. Reaches
ESMFold2's pair FFN on every part.

Driven here through the real `SwiGLUFFN.__call__`, so the row-block shrink, the L1 slice, the L1
layer-norm and the per-shape refusal memo are all inside the arm. Of the 320-1024 aa window the row
block rides in, **320 aa is the only size where the L1 fc1 destination is served at all** (160 of
160 calls), and there 8/16/24/32/48/64 all land within 0.17 % of each other on a 0.14 % A/A floor.
**`obw = 32`, the rung documented as a clash on the p300c, serves every call here.**

At 384 aa and 512 aa the class takes one refusal and retires (1 served, 1 refused, 190 and 254
blocked; 48.733 and 48.774 ms at 512 aa across two sessions), and at 768 aa the config gate declines
all 384. No width is a lever on this part.

The refusal that retires it is worth keeping verbatim: `L1 clash: grid=(8, 6) cores=48
buffer_addr=331776 cb_end=681248 shortfall=349472`.

Watch the two counters disagree: `L1_FC1_STATS` reads 256 served at 512 aa while the latch reads 1.
Counting requests is not counting service.

## 3. `_BATCHED_MATMUL_SATURATION_BLOCKS = 32` — accidentally safe by exhaustion

Comment says "measured on qb1 card 0", a 130-core p150a.

Same `perf/bmm_reconcile/pcm_sweep.py` that fitted it, two sessions agreeing within 0.3 %. The
legality predicate `blocks <= cores` already deletes the 80-block rung on 72 cores, so 8 of the 11
classes have exactly one legal `per_core_M` and the target is inert. In the three that have a
choice the legal set is {32, 16} blocks and 32 is faster in all three: DiT attn@v 0.0629 against
0.1064 ms (1.691x), DiT q@k^T 0.1109 against 0.1389 ms (1.252x), OF3 AttentionPairBias attn@v
0.0486 against 0.0855 ms (1.759x). Every rung bit-exact. There is no legal rung above 32 blocks
here to be wrong about.

## 4. `PAIR_ROW_BLOCK = 128` — no envelope, arbitrary, and not badly chosen

The audit flagged it as fitted on no part at all, its comment claiming only "a tile multiple".

Swept 32 to 512 at 512, 640 and 768 tokens through `_pair_bias_from_z`, the one of its three use
sites that runs without model weights, two sessions. **Nothing refuses at any height and every
height is `torch.equal` against the whole-tensor result**, so there is no envelope to have missed.
The curve is monotone and flat past the shipped value: at 640 aa, 5.1476 ms at 32, 4.5366 at 128,
4.4155 at 320, on a 0.65 % A/A floor. 32 costs 1.1346x and 320 buys 1.0274x, under the 1.05x bar.
128 sits just past the knee, so "a tile multiple" is a sufficient justification.

Bounded claim: its other two sites are the trimul's row-blocked projections, where the height bounds
peak allocation on sizes that would otherwise refuse. That is a footprint question and a timing
sweep says nothing about it, so raising the constant there would need the footprint measured first.

## 5. `TRIMUL_GP_BANK_SPLIT` — the argued Wormhole neutrality, now measured

The argument on main was that Wormhole has 12 DRAM banks and 256 % 12 = 4, so the reader's row
stride already walks banks and the reorder is free. Sound reasoning about a shipped default, never
run on Wormhole.

The premise reads off the part: `dram_grid_size()` returns (x=12, y=1). `perf/k10_b1_permute/
b1_equiv.py`, the same rig that measured it on Blackhole, run twice here. The C=128 cells, worth
1.2990x to 1.5146x on the p150a, read 1.0015x/1.0221x (N=298), 1.0138x/1.0015x (N=320),
1.0045x/1.0036x (N=512) and 1.0035x/1.0010x (N=640). The C=32 cells run 0.8870x to 1.0830x and
change sign between sessions, so that width sits inside the rig's own cross-session floor and
nothing there is a lever. 16/16 cells bit-exact against a live negative control.

**A lever this campaign landed on main costs the production part nothing.** By kind it is
**layout**, and its Blackhole reading is the one already on main.

## Running it

`run.sh` pins the card and the lease and execs the tree's python:

    sh perf/roof_bh_env/run.sh perf/roof_bh_env/bench.py --out results/fp32_grid_ladder_A.json
    BOLTZ2_FP32_SOFTMAX=1 sh perf/roof_bh_env/run.sh perf/roof_bh_env/fold_ab.py \
        --reps 2 --out results/fold_ab_512_A.json

An unpinned open on this box brings every chip up and breaks its co-tenants, so the pin is not
optional.

## Artifacts

`results/`: `fp32_grid_ladder_{A,B}.json`, `fp32_softmax_grid_s512.json`, `fold_ab_512_{A,B}.json`,
`b1_equiv_whglx_{A,B}.json`, `verify_fix.jsonl`, `pair_ffn_bw_{A,B,L320,L384,L768}.json`,
`pcm_sweep_whglx_{A,B}.json`, `pair_row_block_{A,B}.json`.
