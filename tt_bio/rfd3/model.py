"""RFD3 (RFdiffusion3) ttnn component ports.

Includes the TokenInitializer, dense-mask LocalAtomTransformer encoder,
CompactStreamingDecoder (device Upcast/Downcast cross-attention), and
LinearSequenceHead. The atom attention mask is mathematically equivalent to
upstream's gather-sparse path.

Design (per p1 §4 / state §2c.3): the index/one-hot/scatter/gather feature
engineering runs on HOST (pure torch, cheap, index-heavy — no matmul); the heavy
linears / RMSNorm / Transition / pair-bias attention / Downcast cross-attention run
on the TT device via ttnn. Decoder atom grouping uses device gathers, keeping the
three-block decoder resident.

Weight remap is a trivial prefix-strip: the 118 `model.token_initializer.*` ckpt keys
are canonical and load 1:1 (verified 0 missing / 0 extra vs the faithful reference).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

import ttnn

from .. import rfd3_bias, softmax_generic
from .. import sdpa_generic as _SG
from ..envflags import env_flag
from . import block_sparse as _BS
from . import distbins as _distbins
from .tiles import TILE, align_tile, pad_axis
from .. import pair_swiglu_ws as _WS
from .. import dit_pair_bias as _DPB
from .. import gather_tiled as _gather_tiled
from ..addcmul_at import addcmul_at as _addcmul_at
from .. import eltwise_fusion as _EF
from ..tenstorrent import (Module, get_device, CORE_GRID_MAIN, attn_value_matmul, silu_ckc, mm_sigmoid,
                           atom_pair_budget_bytes, l1_resident_budget_bytes, row_block,
                           _dram_oom, fused_sdpa, _sdpa_program_config, is_wormhole)

# This is a SNAPSHOT, and deliberately left as one. `_configure_active_compute_grid` widens
# tenstorrent.CORE_GRID_MAIN to 13x10 when a Blackhole device opens, after this import has
# run, so every `core_grid=CORE_GRID_MAIN` below pins 11x10 -- 110 of the 130 available
# cores. That looks like a bug; it is measurably not one. Both grids were timed on all seven
# pinned call sites: 13x10 is bit-exact
# against 11x10 everywhere (maxabs 0.0, 14/14 cells at D=1 and D=8), and it is 14.28 ms/step
# SLOWER at D=8 -- DiffusionTokenEncoder.process_z alone goes 1.967 -> 6.353 ms. So grid
# width here carries no numerics, only performance, and the narrower grid wins. Re-resolving
# this lazily would be a ~0.7% regression at D=8.

# ttnn derives a `core_grid=` linear's matmul program config from M = batch * rows:
# a larger per_core_M leaves less L1 for the in0 block, so in0_block_w -- the
# K-blocking of the fp32 accumulation -- shrinks, and the same arithmetic is grouped
# into a different number of partial sums and rounds differently in bf16. That made a
# batched design forward diverge from the standalone one, and it is why a 200-step
# batched trajectory used to drift. The linears that measurably lose batch invariance
# pass `core_grid=BATCH_INVARIANT_GRID` instead: ttnn's default heuristic blocks by K
# alone, so a batched forward stays bit-identical to the standalone one at any batch
# size. Dropping the hint from every linear would cost 1.32x/1.59x at D=1/D=8;
# confining it to the affected ones keeps the batching win.
# scripts/rfd3_port/verify_batch_invariance.py is the gate;
# scripts/rfd3_port/probe_callsites.py re-derives which linears need it.
#
# RFD3_FAST_GRID=1 pins the grid on those linears anyway. It is a MEASUREMENT LEVER and it
# is NOT a shipping option, now that both halves of its tradeoff are measured:
#   * worth -5.09% per step at D=1 and -5.19% at D=8 (3359 atoms, interleaved one-tree A/B);
#     a real 200-timestep design goes 54.07 s -> 50.13 s.
#   * costs 6.525 A RMSD / 6.380 A after Kabsch / PCC 0.920 against the shipped path at the
#     SAME seed over a full 200-step trajectory, where a seed change is 25.305 A. It returns
#     a different design, silently, still reporting finite.
# The decisive number is that RFD3_TUNE_MATMUL=1 -- which searches explicit program configs
# and keeps only bitwise-identical ones -- measures -4.96% at 3359/D=8 with maxabs exactly
# 0.0 over a D=8 trajectory. Breaking bit-exactness here buys 0.2 percentage points. So there
# is no accuracy-for-speed trade to make: take the exact path. Default OFF.
FAST_GRID = env_flag("RFD3_FAST_GRID", False)
BATCH_INVARIANT_GRID = CORE_GRID_MAIN if FAST_GRID else None


def _grid_if_single_k_tile(a):
    """`core_grid=` for matmuls whose K is one tile, `BATCH_INVARIANT_GRID` otherwise.

    The blocking that breaks batch invariance above is purely the K-blocking: matmul
    output rows are independent, so how M is spread over cores cannot change a row's
    value, while a different `in0_block_w` regroups the fp32 accumulation. With K
    inside a single tile there is only one possible K-blocking, so the hint is
    bit-exact at any batch size and costs nothing to take.

    This matters most in the decoder's [D*I, n_head, n_query, head_dim] attention
    pair, which p13 profiled running on 1 of 130 cores (177.6 ms of 2537 ms total
    device time at D=8). Hinted, it is 58.9x faster and bit-identical.
    """
    return CORE_GRID_MAIN if a.shape[-1] <= 32 else BATCH_INVARIANT_GRID


# --- batch-exact program-config tuning (p15) --------------------------------
# ttnn runs `linear([B, .., M, K], [K, N])` as B independent M-row matmuls and its config
# heuristic often leaves them on 16-64 of 130 cores (p13: matmul on 52.6 cores at 3.35% FPU
# util). Folding the batch into M -- `fuse_batch=True` in the program config -- puts the same
# work on the whole grid: 1.66-5.63x on this model's real per-step shapes.
#
# The catch is the one p14 hit with `core_grid=`: a different core distribution can regroup the
# fp32 accumulation and change the answer. Reading tt-metal's config builders says exactly which
# field does that -- `in0_block_w`, the K-blocking, and only it. `per_core_M/N`, `out_block_h/w`,
# `out_subblock_h/w`, the grid and `fuse_batch` cannot: matmul output rows are independent and
# the k-sum for one output tile is grouped purely by `in0_block_w`. The three builder branches
# each derive it differently ((Kt%2)?1:2 / largest d<=4 dividing Kt / 2 demoted to 1) and the
# branch predicate depends on M, batch, N and the grid -- which is why a hint is bit-exact when
# Kt==1 (all branches give 1: `_grid_if_single_k_tile` above) and unsafe otherwise.
#
# So the right value is not a function of K -- the same K=128 needs 2 at N=512 and 1 at N=256 --
# and a static table would be the trap p14 documented. Instead each shape is calibrated once
# against ttnn's own default output: candidates that are not BITWISE identical are discarded, and
# the fastest survivor is cached. Bit-exactness is a precondition of the choice, not an argument
# about it, so an unseen input cannot silently pick a config that rounds differently. A grouping
# mismatch is data-independent, so agreeing on one real activation tensor across thousands of
# tiles means the groupings are the same; scripts/rfd3_port/probe_pinned_pair_linears.py has the
# per-shape sweep and the cross-(I, D) evidence.
_TUNED_MM_CACHE = {}
# Stays OPT-IN, and p17 measured why rather than assuming it. The per-step win is real and
# reproduces in the shipped (trace-OFF) configuration -- +10.0% at 419 atoms and +1.4% to +4.0%
# from 979 to 3359, favouring the tuned path in 9 of 10 fixture-rounds (sign test p=0.02). Two
# things cancel it end to end:
#   * Calibration costs a fixed ~5.9s per process (measured as whole `tt-bio design` wall clock,
#     identical at 5 and at 20 timesteps, so it is one-time and not per-step). At 419 atoms one
#     batch of 8 designs at 200 timesteps only saves ~5.4s, so a single-batch run is net NEGATIVE.
#     It turns positive from the second batch on, i.e. --num_designs > --batch_size.
#   * `_tunable` needs xs[0] > 1, so the lever only engages on a batched forward. p25 raised
#     the atom-count clamp (`_BATCH_ATOM_PAIR_BUDGET`, see design.py) to the measured
#     memory bound, so that is now every design up to 3359 atoms rather than only <=838 --
#     but the sizes it newly reaches are also the slow ones, where a single batch takes long
#     enough that the 5.9s is a smaller share. Whether that flips the end-to-end sign at
#     large sizes is unmeasured; p17's numbers are per-step, and the decision below rests on
#     the whole-invocation wall clock.
# So default-on would trade ~-1% on the common single-batch invocation for ~+2% on a rarer one,
# inside a noise floor the D=1 null control puts at +-7%. Flip it once calibration is cheap: the
# dominant cost is compiling candidates that then fail L1 validation, and `_mm_candidates` can
# reject those arithmetically instead of via try/except.
# Whether to calibrate at all is a property of the TARGET, not of the build, so it is resolved
# per run from the atom count (`set_tune_matmul_for_atoms`, called by rfd3.design.run_design).
# Calibration is bit-exact -- `_calibrate_linear` keeps only a program config whose output is
# bitwise equal to the default's -- so this switch carries no numerics, only wall clock.
#
# End to end, seconds per design at the shipped 200 timesteps, qb2, ttnn 0.68.0, under a run
# lock, warm median of 2 after a discarded cold forward, and the whole three-forward
# invocation wall alongside it (perf/dsfix/results/rfd3_batch_e2e.jsonl):
#
#     atoms  batch     warm off -> on            whole invocation off -> on
#      2299      8   21.885 -> 21.066  1.039x     553.5 -> 564.0   WORSE by 1.9 %
#      3844      1   59.967 -> 57.863  1.036x     222.9 -> 207.6   better by 6.9 %
#      6051      1  143.646 -> 123.185  1.166x    452.7 -> 418.3   better by 7.6 %
#
# The per-design win is real at every size, but at 2299 atoms the wall time calibration itself
# spends does not come back inside a 24-design invocation, which is what the old default-off
# note predicted for small fixtures. So it defaults on above 2952 atoms and off at or below,
# which regresses nothing measured. The threshold is bounded by a measured loss at 2299 and a
# measured win at 3844 and is NOT itself measured; it is the same breakpoint the batch speed cap
# uses, for the same reason -- that is where this model stops behaving like a small one.
#
# RFD3_TUNE_MATMUL=1 forces it on and =0 forces it off, at any size.
# ttnn.concat runs 15-20x below its own floor if ANY input piece is narrower than a tile, and
# neither the output width nor the piece offset matters (perf/p52/concat_width_v2.json: 128+65+65
# is 25.35 ms where 128+96+96 is 1.68 and a clone of the result is 1.40). The token encoder
# concatenates z with two 65-wide one-hots, so both are on the slow side of that cliff.
#
# The way out is bit-exact rather than approximate: gather ONE combined one-hot, 130 real columns
# padded to 160, concat it with z into 320 -- both pieces tile multiples, so the fast path -- and
# slice back to 258. The padding is contiguous at the END, so the slice recovers exactly the
# tensor the old concat produced, and the rms_norm downstream still reduces over 258.
# Measured (perf/p53/combined_onehot.json): 31.335 -> 6.168 ms/call, 5.08x, 50.3 ms/step.
_CONCAT_ALIGNED = env_flag("RFD3_CONCAT_ALIGNED", True)
# p89: process_z normalises over 258 columns of which 130 are a one-hot, and the 128 that are
# not are Z_init_II -- fixed for the whole design. So the rms scale and the z half of the linear
# are both loop invariants, and the one-hot half is a table lookup. Three device ops per call
# instead of six, 5.03 ms against 15.14 at [1,685,685,*] (perf/p89/process_z.json). Splitting one
# fp32 accumulation into two bf16-rounded halves is not bit-exact (~1 bf16 ulp, rel median
# 4.8e-3). Graded on the normal bar (WH, 8 paired seeds 62-69: breaks 4 -> 3, clashes -5.6 se 10,
# drift 6.1-14.1 A vs a 13.3-20.7 A seed floor, 65.7 -> 63.9 s/design), so on in both modes.
_PROCESS_Z_COLLAPSE = env_flag("RFD3_PROCESS_Z_COLLAPSE", True)
# The atom neighbour index (_create_attention_indices: cdist + top-128 over every atom) is pure
# host torch and used to run first in every step, with the device idle: 91.5 ms/step of the
# page fold's 547 (p150a census, scripts/rfd3_port/gap_census.py). The recycle-1 token encoder
# does not read it (only S_I, Z_II and the CA coordinates), so with RFD3_ATTN_OVERLAP on that
# encoder is dispatched first and the host builds the index while the card runs it. Same ops on
# the same inputs, reordered: exact by construction, checked by the fold digest.
# The index stays on the main thread on purpose. Built on a worker thread it gets its own
# OpenMP team, which fights the main thread's for the cores: on qb1 (16 physical cores) the
# build went from 72-88 to 115-140 ms and the page fold from 107.1 to 124.0 s/design.
_ATTN_OVERLAP = env_flag("RFD3_ATTN_OVERLAP", True)
# The k-d tree index (_ATTN_KDTREE) is scipy's own threads plus numpy, so it no longer has the
# OpenMP conflict above, and on a worker thread it can start as soon as the step's coordinates
# exist instead of after the token encoder's dispatch. Same function on the same inputs: exact
# (digests equal, WH seeds 60-63 fast, 60-61 normal). WH --fast @1000 on one chip: 103.6 -> 100.3 s/design.
_ATTN_THREAD = env_flag("RFD3_ATTN_THREAD", True)
# A denoise step crossed to the host ~15 times: process_a/r/c and both time embeddings came back
# for host adds, the encoder, downcast_q, the DiT and every recycle's s were read back only to be
# uploaded again by the next module. Each crossing drains the card, and Tracy on p150a put the
# device busy for 296 of the step's 481 ms. With RESIDENT_STEP the step stays on the card from
# R_noisy_L in to R_upd out; the only reads left are R_upd and the sequence logits per recycle.
# Not bit-exact: the adds round in bf16 on the card instead of fp32 on the host, and process_a
# pools the coordinates before its (bias-free) linear instead of after, equal in exact math.
# Graded on Blackhole and Wormhole (4 seeds at 685 and 418 tokens, normal and --fast): drift
# inside the seed floor, design quality equal to exact. On by default; RFD3_RESIDENT_STEP=0 restores
# the host round trips.
_RESIDENT_STEP = env_flag("RFD3_RESIDENT_STEP", True)
# The per-step neighbour graph is the largest single host cost in a design. At the page fixture
# (6051 atoms) the ledger puts it at 51.9 + 19.8 ms of an 84.8 ms host body, and P3.7 measured
# 54.3 ms/step of that reaching the wall. It is one chain -- cdist -> masked_fill_ -> topk -- over
# an [L, L] fp32 matrix, 146 MB here, written once and read twice through DRAM. Computing it in
# row blocks keeps each slab in cache instead. An output row reads only its own row of distances,
# of the mask and of seq_idx, so blocking cannot change a value; it is checked torch.equal on the
# full index tensor at the production shape and by the fold CIF digest.
# RFD3_ATTN_ROWBLOCK=0 restores the unblocked chain so the fold A/B has an arm; a positive value
# overrides the block size. R=256 measured on the production mask and coordinates by
# scripts/rfd3_port/p61_attn_indices_prod.py (perf/p61/attn_indices_prod.json, L=6051, k=128,
# mask density 0.0078, n=5 medians): it is the fastest block size at both 8 and 16 threads,
# 53.82 -> 20.59 ms at 8 and 48.81 -> 18.60 at 16, and R=512/1024/2048 are all slower.
_ATTN_ROW_BLOCK = int(os.environ.get("RFD3_ATTN_ROWBLOCK", "256"))
# The same index from a k-d tree instead of the dense distance matrix. The fill is the
# (k - n_seq) nearest atoms outside the row's sequence mask, and the k nearest atoms overall hold
# at most n_seq masked ones, so a k-nearest query always contains the whole fill: no [L, L]
# matrix and no top-k over it. The tree measures distances in float64 where torch.cdist uses the
# fp32 |x|^2 + |y|^2 - 2xy form, so the two can only part on a near-tie at the k-th neighbour.
# Its worker threads are scipy's own, capped at torch's thread count, so they do not start a
# second OpenMP team (the reason the index stays off a Python worker thread, see _ATTN_OVERLAP).
# Synthetic 6051 atoms, k=128, 2 threads: 207.6 -> 60 ms on a loaded Galaxy host, 143.6 -> 52
# on qb2, every row equal to the cdist build and to a float64 reference. On by default: the tree is
# the more accurate index (the fp32 cdist misses 0-2 rows per case against float64), and the fold
# moves only by those near-ties. Wormhole @1000 MHz, page fixture: fast 129.4 -> 117.2 s/design,
# normal 135.8 -> 131.5. Graded against the cdist arm, 16 fast seeds: clash delta +4.9 +- 4.5,
# breaks 7 -> 7, design drift 2.6-11.2 A against a 12-20 A seed floor; 6 normal seeds +1.8 +- 4.0.
# RFD3_ATTN_KDTREE=0 restores the cdist chain.
_ATTN_KDTREE = env_flag("RFD3_ATTN_KDTREE", True)
# The atom neighbour index on the card (tt_bio.dknn): exact uint64 keys, launched ahead of the token encoder instead
# of a host k-d tree the step waits on (hostfreewh: 12-15 s/design of WH fast). Same rows as the tree (0 of 6051
# differ at six noise levels). p150a, fast seeds 60,61 x2 @1312-1350 MHz: 31.3-31.8 -> 29.7-30.0 s/design, digests
# equal. On by default except on Wormhole, where it has not run on a card yet; RFD3_DKNN=0/1 forces it.
_DKNN = env_flag("RFD3_DKNN", False) if os.environ.get("RFD3_DKNN") else None
# The tree query with the design's static atoms (a binder's target, a fixed motif) answered once per design and only
# the rows a moving atom can reach queried each step (tt_bio.rfd3.staticknn). Every row and its order equal the full
# query, so the index is unchanged. WH page cell @1000 MHz, CIFs equal: fast 61.4 -> 56.7, normal 62.7 -> 57.3 s/design.
_STATIC_KNN = env_flag("RFD3_STATIC_KNN", True)
_STATIC_KNN_CACHE: dict = {}
_NN_ROWS_CACHE: dict = {}
# With the host tree (Wormhole) and static atoms: the moving atoms' own rows, the bulk of what the host still queries
# each step (1400 of the page fixture's 6051 rows, k 128), go to the card (tt_bio.dknn over that row range) while the
# host answers the static rows. Needs the movers to be one contiguous id range, as a binder is. Same rows, so the
# CIFs are unchanged; WH page cell @1000 MHz, seeds 60,61 x2: fast 43.98 -> 42.70, normal 44.70 -> 43.65 s/design.
_DKNN_MOVERS = env_flag("RFD3_DKNN_MOVERS", True)
# With the card index: the static rows scored against their nearest static atoms (found once per design) and the
# movers only, under a bound that proves the row (tt_bio.dknn.Static); a row it cannot prove runs in full. Same rows.
_DKNN_STATIC = env_flag("RFD3_DKNN_STATIC", True)
_DKNN_STATIC_CACHE: dict = {}
_DKNN_MOVERS_MIN_L = 2048   # the atom index; the DiT's token index is too small to pay for a launch
# The token initializer's pair mean-pool as two index_adds instead of one-hot einsums over every atom pair (~800 GFLOP
# at 6051 atoms): 3.74 -> 0.14 s a call at 2 threads. Same error against float64 (7.71e-9 vs 7.73e-9 mean), but the
# order of additions differs, so 38 of 7.5M pooled values round to a different bf16 on upload. The page cell's CIFs
# stay equal in both modes (WH, .103 @1000 MHz); run_design wall fast 106.5 -> 102.2, normal 105.5 -> 104.0 s/design.
_POOL_SCATTER = env_flag("RFD3_POOL_SCATTER", True)


def _dknn_on():
    return _DKNN if _DKNN is not None else not is_wormhole()


def _pf_dpb_on():
    return _PF_DPB if _PF_DPB is not None else is_wormhole()


def _pz_addcmul_on():
    return _PZ_ADDCMUL if _PZ_ADDCMUL is not None else is_wormhole()


def _pf_rmsb_on():
    return _PF_RMSB if _PF_RMSB is not None else is_wormhole()


# The 18 DiT blocks project the SAME pair tensor with their own [c_pair, n_head] weight, and that
# projection is the largest single op in the model: 36 calls/step (18 blocks x 2 recycles),
# 26.742 ms/step, 42.6 % of this card's measured 390.0 GB/s read roof (perf/p56/linear_census.json).
# `LocalTokenTransformer.run_device` loops one `z` over the blocks, so the 123 MB input is read 18
# times to write 18 disjoint 16-wide slabs. Fused, it is one matmul that reads `z` once and each
# block slices its own columns back out.
# Each block gets a 32-wide slot rather than a 16-wide one so every slice starts tile-aligned:
# p52 measured a sub-tile piece putting a whole op 15-20x below its bandwidth floor. The write
# volume is unchanged, because the 18 shipped outputs are logically 16 wide and already tile-padded
# to 32. Screened in scripts/rfd3_port/p60_pairbias_fusion.py.
# RFD3_PAIRBIAS_FUSED=0 restores the per-block projection so the fold A/B has an arm.
_PAIRBIAS_FUSED = env_flag("RFD3_PAIRBIAS_FUSED", True)
_PAIRBIAS_SLOT = 32
# The same fusion for the atom encoder's and decoder's three blocks: each projected the shared
# neighbour pair features [1, L, 128, 16] to its 4 heads with its own matmul, 1.41 ms a call on
# Wormhole for a tile-padded 4-wide output. One matmul to all 12 heads and one permute, each block
# slices its 4 heads back out (_prime_atom_pair_bias). RFD3_ATOM_PB_FUSED=0 restores the per-block form.
_ATOM_PB_FUSED = env_flag("RFD3_ATOM_PB_FUSED", True)

# The pair `Transition` is 37 % of every step's DRAM traffic (23.70 of 63.43 GB,
# perf/p63/traffic_census.json) against an irreducible 1.98 GB, because all of its intermediates
# round-trip DRAM and are dead the instant `fc3` consumes them. Row-chunking on dim 1 -- which is
# NOT a tiled dimension, so a slice has no sub-tile cliff and no padding -- makes `fc2`'s output
# and the gated product small enough to keep in L1, so `b` and `m` are never written out and never
# read back: 1975 of the 3951 MB an H=512 call moves.
#
# Only those two. `x_norm` and `fc1`'s output stay in DRAM, and that is what makes this bit-exact
# rather than nearly so -- see `Transition._swiglu`. Chasing them as well measured 16.10 ms/call at
# H=512 instead of 18.55, and diverged (perf/p66/audit_perop.json).
#
# MEASURED on the built code path at the page fixture's pair shape [1, 685, 704, 128], warm, synced
# both sides, n=6, under benchlock on card 2 (perf/p64/pinned_l1_heights.json). ms/call:
#
#   H=512  shipped 21.91 | h=32 18.68  h=64 18.55  h=96 19.04  h=128 does not fit (185 MB)
#   H=256  shipped 13.30 | h=64 11.83  h=128 12.04  h=192 12.27  h=256 does not fit
#
# h=64 at BOTH widths, which is not the constant-`h x hidden` rule the wider variant followed: with
# only two residents the footprint is comfortable at either width (92 MB at H=512, 46 at H=256) and
# the optimum is set by the chunk count, not by the fit. The byte cap below is a safety net for an
# unmeasured shape, bracketed by measurement -- 138 MB of live L1 fits, 185 MB throws.
#
# 140.8 -> 121.5 ms/step, -19.3 ms/step, measured end to end. SIXTEEN calls per step, not the eight
# this comment used to claim: the `fc1` census counts 1596 calls per 200-timestep fold at EACH
# hidden width, and 199 diffusion-module calls (`len(sched) - 1`) x 8 + 4 from the one-time
# initializer closes that exactly. The end-to-end figure was never wrong; the call count it was
# attributed to was, and it mis-scaled region 2's prize by 2x before a census caught it
# (state/rfd3-fusion-programme.md §14.4).
#
# The chunking is NOT where the win is. With the intermediates left in DRAM, chunking alone is a
# LOSS of 1.3-1.5 ms/call (perf/p64/pair_transition_l1.json arm B): the slice, the closing concat
# and the extra op count cost more than they return. All of the result is the L1 residency.
_PAIR_TRANSITION_L1 = env_flag("RFD3_PAIR_TRANSITION_L1", True)


# `fc1` is the one matmul in a pair Transition that never got calibrated, because its silu is
# fused onto the call and ttnn wants a fused activation on the program config, which cannot be
# pinned bit-exactly. Measured at all eight of `fc1`'s live keys
# (scripts/rfd3_port/p128_fc1_config_census.py, perf/p128/fc1_config_census.json):
#
#   torch.equal( ttnn.silu(ttnn.linear(x, w)),  ttnn.linear(x, w, activation="silu") )  ->  True
#
# The shipped call is a matmul followed by a separate eltwise silu; it rounds twice. So taking the
# silu out of the call is NOT a precision change, and once it is out, `fc1` can go through
# `_tuned_linear` like its two siblings and its output can join `b` and `m` in L1.
#
# Isolated, per chunk: 2.02-3.86x at the four body/tail keys of the census fixture, every leg
# bit-exact against the shipped activated output. 63.39 -> 24.30 ms/step, -39.09 ms/step. That is
# an isolated per-op figure and `tt-bio-isolated-op-timing-oversync-inflates-cost` applies to both
# arms of it, so the fold A/B is the number that counts. That fold has run: with L5b on in the same
# fold, -6.206 s/design and 1.0671x at 685 tokens, bit-exact over seven reps
# (`perf/p132/both_ab_R4_warmfix.json`). Default ON since.
#
# Almost all of it is the L1 residency, not the pinned config. With the output left in DRAM the
# split is 1.11x at 704 tokens and 0.91x -- a LOSS -- at 1024, where calibration finds nothing to
# pin either and the split's second dispatch is all that is left
# (perf/p130/fc1_above_free_window.json). So the lever is all-or-nothing: `Transition.__call__`
# takes the split only where `fc1`'s output can join `b` and `m` in L1 without costing an extra
# chunk call, which is every size at hidden=256 and up to 693 tokens at hidden=512.
_FC1_SPLIT_SILU = env_flag("RFD3_FC1_SPLIT_SILU", True)
FC1STATS = [0, 0]                  # [served, declined], the shape `lever_census.LEVERS`
                                   # reads, so an A/B arm cannot silently decline
FC1SERVED = {}                     # shape -> count, for the calls that got both halves
FC1DECLINES = {}                   # shape+reason -> count, the other half of the same census
# Same census, for the L1 residency of the gated product itself. [granted, declined] plus the
# per-shape reasons, because a size whose verdict is "the residency was declined" has to SAY so:
# an unchanged fold digest is exactly what a working decline and a dead gate both produce
# (`negative-control-must-break-what-check-reads`).
PTL1STATS = [0, 0]
PTL1DECLINES = {}
# Shapes whose residency the allocator refused at runtime even though the budget granted it.
# Remembered the way `_dram_narrow`'s cap is: the first block of a tensor pays for the refusal
# and the rest of its blocks start where that one ended.
PTL1REFUSED = {}


def set_fc1_split_enabled(on):
    """Toggle the split `fc1` from a screen without going through the environment."""
    global _FC1_SPLIT_SILU
    prev, _FC1_SPLIT_SILU = _FC1_SPLIT_SILU, bool(on)
    return prev


# The atom attention's score tensor is [1, 4, 6051, 6080] fp32 = 588.6 MB at the page fixture, and
# 128 columns of each row carry a real value: every other column is the -1e4 mask, whose exp
# underflows to exactly 0.0. So 47.5x of the softmax's traffic contributes nothing to any row sum.
# Under this flag the row reduction runs on the gathered [1, 4, 6051, 128] form (12.4 MB) and the
# result is scattered back into a zero template.
#
# NOT bit-exact, and that is the whole point of the flag. The scores are bit-identical under
# gathering -- the QK dot is one tile deep, so the dot-product tree does not depend on the N tiling
# -- and the set of contributing terms is identical, because the masked columns are exact zeros
# after exp. What changes is the ORDER: 128 contiguous terms reduce in a different tree from 128
# terms scattered through 6080, and fp32 addition is not associative. Downstream that is a sub-ULP
# perturbation of a bf16 attention weight, which 200 diffusion steps turn into a different
# structure. Default OFF until the accuracy envelope in scripts/rfd3_port/p78_envelope_spec.json
# reads out; the bar was committed (8ae442c5) before the first number existed.
#
# Upstream RFD3 computes this attention in exactly the gathered form: `sparse_pairbias_attention` is
# the only pair-bias path in the released rc-foundry 0.2.0, and it still serves 28656 of 35820
# pair-bias calls on the H200 arm that produces our own 12.974 s/design denominator. The dense
# formulation is a tt-bio tiling choice, not the reference's.
_GATHERED_SOFTMAX = env_flag("RFD3_GATHERED_SOFTMAX", False)
# Atom attention as one fused SDPA: the dense [1,H,L,L] bias (mask plus neighbour pair bias) is
# built once per block in bf16 and the kernel never writes the scores. The shipped chain writes
# the bf16 scores, rereads them into 2x as many fp32 bytes and rereads those for the softmax,
# about 1.2 GB a call at 6051 atoms, 9 calls a step. Online softmax changes the reduction order,
# so the digest moves; the result is closer to float64 than the shipped chain (one call at the
# page shape: maxabs 0.047 against 0.087, PCC 0.99981 against 0.99975). On by default: over 4
# seeds of the 685-token page design the same-seed drift is 5.5-9.3 A, the size of the 7.3 A by
# which the shipped chain's own Blackhole and Wormhole designs differ, with design quality unchanged
# (docs/rfd3-design.md). RFD3_ATOM_SDPA=0 restores the chain.
_ATOM_SDPA = env_flag("RFD3_ATOM_SDPA", True)
ATOM_SDPA_CALLS = [0]
CSS_CALLS = [0]
# The SDPA mask is cached per atom block for the step, so three to six [1, H, N, N] bf16 masks live
# at once. Above this size per mask the shipped chain runs instead: 6080 atoms (the 685-token page
# fixture) is 296 MB, while the 12085-atom top of the size ladder would hold ~1.2 GB per block on a
# 12 GB Wormhole chip, a size whose memory headroom has not been measured with the cache.
_ATOM_SDPA_MAX_MASK_BYTES = int(float(os.environ.get("RFD3_ATOM_SDPA_MAX_MASK_BYTES", "6e8")))
# The pair Transitions' hidden activations (fc1 and fc2 outputs, the gated product) in bfp8:
# 1088 instead of 2048 bytes a tile on the region that is ~65 % of the token encoder and is bound
# by per-tile pack/unpack. The input norm and the output stay bf16. Off by default while graded.
_TRANSITION_B8 = env_flag("RFD3_TRANSITION_B8", False)
# The token pair Transition as one weight-stationary program (`tt_bio.pair_swiglu_ws`): x read once,
# [w1 | w2] and w3 resident in L1, fc1/fc2/the product never leave the core. bf16 only for now, so
# it takes the token pair tensor ahead of RFD3_TRANSITION_B8's bfp8 hidden (faster and more accurate
# than either shipped path on WH). In --fast: p150a @1350 61.30 -> 56.25 s/design
# (seeds 44-47), Wormhole @1000 114.7 -> 100.1 (seeds 50-53). Graded over 10 paired seeds on both
# arches: clash delta -1.4 (se 5.4), chain breaks identical seed for seed, same-seed drift
# 3.9-12.7 A against a 13.5-20.0 A seed floor. In normal mode, Wormhole @1000 124.0-127.4 ->
# 89.0-93.0 s/design; 20 paired WH seeds: clash delta +4.9 (se 4.3), breaks 11 -> 12, in-band equal,
# drift 2.9-14.1 A against a 13.8-21.7 A seed floor. On in both modes.
_PAIR_WS = env_flag("RFD3_PAIR_WS", True)
# RFD3_PAIR_WS with its weights in bfp8, run at HiFi2 (covers every bfp8 weight bit), the form the shared
# `tenstorrent.ws_swiglu` takes in fast mode for the other models. x and the RMSNorm prologue stay bf16. Graded off.
_PAIR_WS_B8 = env_flag("RFD3_PAIR_WS_B8", False)
# The same bfp8 weights at LoFi (WH matmul 56 TF bf16 HiFi4 vs 120 TF bfp8 LoFi). Fast mode: WH @1000 fast
# 56.4 -> 54.4 s/design, p150a 29.8 -> 28.5; 8 paired fast seeds per arch: drift 3.7-12.7 A against a 12.5-19.6 A
# seed floor, breaks 4 -> 5 (WH) and 4 -> 4 (p150a), clashes 433 -> 428 and 426 -> 393.
_PAIR_WS_LOFI = env_flag("RFD3_PAIR_WS_LOFI", False)
# The token encoder's `z = z + transition(z)` (transition_2 and the two pairformer blocks, 8 calls a step) as one
# RFD3_PAIR_WS program that seeds fc3's fp32 dest with z and writes z + t back over z (`pair_swiglu_ws` resid):
# no 1x685x704x128 add pass (WH 1.56 ms each, at its DRAM roof) and one bf16 rounding of z + t instead of two,
# closer to float64 (hidden 256 max_abs 2.0 vs 8.0). Wormhole @1000 normal 80.8 -> 79.6 s/design; 8 paired WH
# normal seeds: breaks 3 -> 4, clash delta +11 (se 9.8, median -3), drift 5.7-14.5 A against an 11.5-19.8 A
# seed floor. On in both modes.
_PAIR_WS_RESID = env_flag("RFD3_PAIR_WS_RESID", True)
# RFD3_PAIR_WS with the silu on the packer thread, fc1 of the next hidden group issued before fc2, so the matmuls
# overlap the SFPU (`pair_swiglu_ws` pack_silu, the shared `ws_psilu`). Bit-identical: CIF digests equal on every
# fold. Wormhole @1000 A/B/A/B (seeds 60,61): normal 50.15 -> 49.15 s/design, fast flat (49.52 vs 49.55); p150a
# @1350 normal 29.31 -> 28.87.
_PAIR_WS_PSILU = env_flag("RFD3_PAIR_WS_PSILU", True)
# The token encoder's pair path (process_z, transition_2 x2, each Pairformer block's z_transition) is pointwise per
# token pair, a function of Z_init (fixed for a design) and that pair's distance bin (and self-conditioning bin).
# Bit-exact memo: per recycle slot the outputs after block 0 and block 1 stay on the card, and a step recomputes only
# the 32-pair row tiles holding a pair whose bin changed (pair_swiglu_ws `rows`). On the page trajectory 58 % of row
# tiles change per step on average, ~0 % over the last 40 steps. Holds 4 copies of z; off above RFD3_PAIR_MEMO_MB.
# Wormhole @1000 (seed 60, digests equal every fold): fast 42.32 -> 40.54 s/design, normal 43.14 -> 40.64.
_PAIR_MEMO = env_flag("RFD3_PAIR_MEMO", True)
_PAIR_MEMO_MB = int(os.environ.get("RFD3_PAIR_MEMO_MB", "1200"))
# Every RFD3 matmul at HiFi2 instead of HiFi4: two multiplier passes per bf16 product instead of
# four. Off, and not in --fast: on top of the fast set it saved nothing on a p300c (4 seeds, 685
# tokens) and doubled the clashes of one 418-token design.
_HIFI2 = env_flag("RFD3_HIFI2", False)
# The dense attention (the DiT's 36 blocks a step, and atom blocks below the sparse cutoff) as one
# fused SDPA, the way Protenix's and Boltz-2's token DiTs run it (`dit_sdpa`): the [1,16,685,704]
# fp32 scores, their fused bias add and the softmax+PV pass become one kernel that never writes a
# score. The mask is (pair bias + neighbour mask) with the bias pre-scaled by sqrt(head_dim), same
# as the atom SDPA. Online softmax changes the reduction order, so it is graded on accuracy: with the
# heads padded to 64 in the weights it saves 15 % of a 685-token design (p300c, 72.7 -> 61.4 s) and
# grades inside the seed floor with design quality equal to exact on Blackhole and Wormhole (4 seeds
# at 685 and 418 tokens each). On by default; RFD3_DIT_SDPA=0 restores the materialised scores. The
# mask-size cap is the atom SDPA's.
_DIT_SDPA = env_flag("RFD3_DIT_SDPA", True)
DIT_SDPA_CALLS = [0]
# The DiT's stacked SDPA masks as one program (`tt_bio.dit_pair_bias`): projection, head-first layout, scale, mask
# add and padding in a single pass over z, the packer doing the re-tiling the permute did. WH @1000, 685 tokens:
# 24.6 -> 4.0 ms a call, closer to a float64 reference than the ttnn chain (rel err 0.0024 vs 0.0042, no element
# off by more than 0.25). Normal mode, 4 paired WH seeds: 100.1 -> 91.4 s/design, drift 3.8-10.1 A against a
# 14.2-18.0 A seed floor, breaks 2 -> 2, clashes 61.5 -> 52.8. p150a @1350 (FPU K-sum, n_bad 0): 15.9 -> 1.9 ms a
# call; 12 paired normal seeds read clashes 59.3 -> 57.2, breaks 6 -> 7, drift 2.9-13.0 A against a 14.8-19.6 A floor.
# On in both modes on both archs; RFD3_DPB=0 restores the chain.
_DIT_DPB = env_flag("RFD3_DPB", True)


# The token encoder's pairformer pair bias through the same program: 16 heads (W^T zero-padded to one 32-head tile,
# the second half written to a scratch output), scale 1, the key pad at -1e4 from the mask. Replaces the linear
# [I*I, 128] @ [128, 16] (2.03 ms a call on WH @1000 against a 0.62 ms roof), its permute and the fp32 pad.
# WH page cell @1000 MHz: fast 51.3 -> 50.7, normal 52.4 -> 51.8 s/design. Eight paired normal seeds on the WH defaults:
# design drift 4.0-13.7 A against a 13.0-20.7 A seed floor, breaks 4 -> 3, clashes 433 -> 345. On by default on Wormhole
# only (a p150a read +0.2 s); RFD3_PF_DPB=0/1 forces it.
_PF_DPB = env_flag("RFD3_PF_DPB", False) if os.environ.get("RFD3_PF_DPB") else None
PF_DPB_CALLS = [0, 0]   # served, declined
# The pairformer's single attention through the fused SDPA, the DiT's way: 24-wide heads zero-padded to 32 in the
# q/k/v/g weights (zero columns add 0 to every q.k and come back as zero output columns), one nlp_create_qkv_heads,
# and the pair bias as the SDPA mask pre-scaled by sqrt(head_dim). Replaces q.k^T, the fp32 typecasts, scale, bias add,
# softmax and attn@v, and the reshape/permute/pad per tensor. WH page cell @1000 MHz: fast 50.9 -> 49.7, normal
# 51.8 -> 50.9 s/design; eight paired normal seeds: design drift 3.5-11.4 A against a 12.6-20.3 A seed floor, breaks
# 3 -> 2, clashes 345 -> 337. p150a @1350 (linear bias, no dpb): normal 29.46 -> 28.25, fast 28.08 -> 27.11; eight
# paired normal seeds: drift 5.2-11.7 A against a 12.5-19.6 A floor, breaks 3 -> 3, clashes 426 -> 453.
_PF_SDPA = env_flag("RFD3_PF_SDPA", True)
PF_SDPA_CALLS = [0]
# The SDPA path's mask from rms0(z) @ (ln_0 * W_b * sqrt(head_dim)) in one `rms_pair_bias` program (the norm weight and
# the scale folded into the projection in fp32), plus one add of the key/row pad. Replaces the standalone rms_norm over
# [I, I, 128] and the dit_pair_bias pass that re-read its output. WH page cell @1000 MHz: fast 40.1 -> 38.9, normal
# 40.5 -> 39.7 s/design; eight paired normal seeds: design drift 3.4-12.9 A against a 13.9-18.4 A seed floor, breaks
# 3 -> 2, clashes 390 -> 370. On by default on Wormhole; RFD3_PF_RMSB=0/1 forces it.
_PF_RMSB = env_flag("RFD3_PF_RMSB", False) if os.environ.get("RFD3_PF_RMSB") else None
PF_RMSB_CALLS = [0]


# The DiT builds all 18 blocks' masks in one pass while the stack fits under this many bytes
# (bf16, 18 x 16 heads x n_key^2: 1317 tokens); above it each block builds its own.
_DIT_SDPA_STACK_BYTES = int(float(os.environ.get("RFD3_DIT_SDPA_STACK_BYTES", "1e9")))
# (q, k) chunk sizes. q only splits independent query rows, so it moves the work, never the values. WH @1000,
# 704 keys (scripts/rfd3_port/gap_ditsdpa.py): q352 0.247 ms/call against 0.294 for q64, bit-equal; on p150a
# q64 stays best (0.122 vs 0.185).
_DIT_SDPA_CHUNKS_ENV = os.environ.get("RFD3_DIT_SDPA_CHUNKS")


def _dit_sdpa_chunks():
    spec = _DIT_SDPA_CHUNKS_ENV or ("352,352" if is_wormhole() else "64,352")
    return tuple(int(x) for x in spec.split(","))


def _dense_sdpa_config(n):
    """SDPA chunks that divide the padded axis `n` exactly (q near the asked size, k the widest up
    to it), so the kernel never pads the mask grid past `n`."""
    div = [c for c in range(32, n + 1, 32) if n % c == 0]
    q_ask, k_ask = _dit_sdpa_chunks()
    q = min(div, key=lambda c: abs(c - q_ask))
    k = max(c for c in div if c <= max(k_ask, 32))
    return _sdpa_program_config(q, k)
# The atom SDPA visiting only the key chunks each query chunk's neighbours fall in (`sdpa_generic`
# css), with the mask writer skipping the rest. On the real index in sequence order 24 % of the
# 64 x 64 score tiles hold a neighbour (p108, mean over a 200-step design); every other tile is the
# -1e4 fill, whose weight in the online softmax is exactly 0, so skipping it changes only the work.
# One design per call (the per-q-chunk counts are runtime args); an axis 64 does not divide keeps the
# chunked SDPA. Bit-equal to the dense chunk-64 program, float64 maxabs 0.048 vs 0.050 for the shipped
# q160/k320 one. p150a @1350, 10 paired normal seeds: breaks 5 -> 4, clashes 53.0 -> 53.4 (se 5.7),
# drift 5.3-9.6 A against a 14.4-19.9 A seed floor; WH @1000 -4 s/design in both modes. RFD3_CSS=0 restores it.
_CSS = env_flag("RFD3_CSS", True)
_CSS_STEP = {}
# Split the atom attention's q, k and v into heads with one `nlp_create_qkv_heads` over their concat instead
# of a reshape + permute each: the 4D reshape tile-pads the 4-head axis to 32 rows, 8x the bytes (WH census:
# 0.40 + 0.10 ms per tensor at 6051 atoms, 27 tensors a step). Data movement only, same values. Only where
# head_dim is a tile multiple (the DiT's 48 has its own padded path). WH @1000, seeds 60,61 x2 in both modes:
# digests equal to the reshape path. RFD3_QKV_HEADS=0 restores it.
_QKV_HEADS = env_flag("RFD3_QKV_HEADS", True)
# The same for the gated cross attention (downcast/upcast, ~13 calls a step): q and the k|v concat go through
# one `nlp_create_qkv_heads` that also hands back k transposed, instead of a 5D reshape (heads tile-padded to
# 32 rows) + permute + reshape per tensor and a transpose of k. ~15 ms/step on WH fast; digests equal (WH, p150a).
# RFD3_GCA_HEADS=0 restores it.
_GCA_HEADS = env_flag("RFD3_GCA_HEADS", True)
# A downcast GCA has one query per token. Passed as q.unsqueeze(2), each token's row is tile-padded to 32, so
# ln_q, to_q, to_g and to_out ran on 32x the rows; an upcast's 3 keys a token pad the same way 3 -> 32 for ln_kv,
# to_k and to_v (WH fast census ~11 ms/step together). Flat [B,rows,C] inputs run them on the real rows.
# RFD3_GCA_FLAT_Q=0 restores the padded form.
_GCA_FLAT_Q = env_flag("RFD3_GCA_FLAT_Q", True)
# The atom side of a GCA (an upcast's queries, a downcast's keys) is packed into a [tokens, 32] grid where only the
# real atoms are rows (6051 of 21920 on the page cell). Its row-wise ops commute with the pack gather, and a pad row
# stays zero through them (no bias before attention), so they run on the flat atom stream and the pack moves after
# them; an upcast's output is unpacked before its gate and to_out.
# Bit-identical (CIF digests equal on every fold). Wormhole @1000 A/B/A/B (seeds 60,61): fast 45.73 -> 45.32 s/design,
# normal 46.74 -> 45.77. RFD3_GCA_ATOM_ROWS=0 restores the packed form.
_GCA_ATOM_ROWS = env_flag("RFD3_GCA_ATOM_ROWS", True)
# The collapsed process_z's table lookup as one gather that writes TILE layout (tt_bio/gather_tiled.py) instead of
# ttnn.embedding + to_layout: WH fast census 5.07 + 4.77 ms/step of embedding against a 0.5 ms roof each, plus
# 2.7 ms/step of tilize. Same values (data movement only): gather 4.76 -> 1.28 ms/call on WH, digests equal to
# the embedding path on WH and p150a (seeds 60,61 x2). RFD3_PZ_GATHER=0 restores it.
_PZ_GATHER = env_flag("RFD3_PZ_GATHER", True)
# The gather's table as gather_tiled.spread replicas: the distance bins name the last row for every far pair, so a
# single table queues every read on one DRAM bank (WH fast census cen5: 4.1-4.3 ms/call vs 1.28 ms for a random
# index; bench at 685 x 4225 rel: 5.22 -> 1.28 ms/call). Same bits, different bank: digests equal on WH and p150a.
# WH @1000 fast 57.9 -> 57.0, normal 57.6 -> 56.4 s/design; p150a fast 30.0 -> 29.1. RFD3_PZ_SPREAD=0 restores it.
_PZ_SPREAD = env_flag("RFD3_PZ_SPREAD", True)
# Under the pair memo, process_z as one gather_tiled program that also applies `ainv + em * inv`, on the memo's changed
# row tiles only: the rest of its output is never read (the transitions read those rows only).
_PZ_ROWS = env_flag("RFD3_PZ_ROWS", False)
# The distogram bins of the token encoder (2 host calls per step while the card waits) as one numba pass with the
# near-edge pairs recomputed by torch: equal to the torch chain, 4-5.7 -> 0.7-1.7 ms/call at 685 tokens on 2 cores
# (rfd3/distbins.py). RFD3_BINS_NB=0 restores the torch chain.
_BINS_NB = env_flag("RFD3_BINS_NB", True)
# The collapsed process_z's em * inv + ainv as one ttnn.addcmul (eltwise_fusion.mask_add): WH fast census multiply
# 2.84 + add 1.52 ms/call, two calls a step, and the [B,I,I,128] product is written and read back between them.
# One bf16 rounding instead of two, so the digest moves. WH page cell @1000 MHz: fast 48.06 -> 47.36, normal flat
# (47.92 vs 48.20) s/design. Eight paired normal seeds on the WH defaults: design drift 4.7-11.5 A against a 13.4-21.3 A
# seed floor, breaks 2 -> 3, clashes 337 -> 355. On by default on Wormhole (p150a not read yet); RFD3_PZ_ADDCMUL=0/1
# forces it.
_PZ_ADDCMUL = env_flag("RFD3_PZ_ADDCMUL", False) if os.environ.get("RFD3_PZ_ADDCMUL") else None
_ATOM_SDPA_CHUNKS = tuple(int(x) for x in os.environ.get("RFD3_ATOM_SDPA_CHUNKS", "160,320").split(","))
# ttnn.gather (0.68.0) on dim 3 silently returns wrong data for every tile-row after the first
# once the indexed axis exceeds this many elements: 99.87 % of elements wrong at the production
# [1,4,6051,6080], and ~100x slower at the same threshold. The trigger is element count, not
# bytes -- bf16 breaks at 2048 while fp32 at 1920 (7.5 KB/row) is exact. Measured in
# scripts/rfd3_port/p81{,b,c}_*.py, perf/p81/*.json. ttnn.scatter at the same shape is clean.
#
# Upstream fixed this one day after our tag and the fix is backported in
# scripts/patches/tt-metal-0.68.0-gather, where p97 finds no threshold at any key axis up to 6080. The
# ceiling stays here because it guards the wheel we SHIP, not the private patched runtime that
# backport builds. Raising it is not worth doing anyway: with the gather correct, the gathered
# chain is still 6.2-48.9x slower than the dense one in every atom band (p99), so this flag is
# now off on a measured loss rather than on a limitation.
_TTNN_GATHER_MAX_KEY_AXIS = 1920


def set_gathered_softmax(on):
    """Toggle the gathered atom softmax from a screen without going through the environment."""
    global _GATHERED_SOFTMAX
    _GATHERED_SOFTMAX = bool(on)

# [collapsed calls, shipped calls]. A silent arm is what made p86's batching result wrong, so both
# process_z branches count themselves: the on arm must show shipped == 0 and the off arm
# collapsed == 0, at the same total.
PZSTATS = [0, 0]
PAIRMEMO = [0, 0]   # row tiles the pair memo recomputed, row tiles a full pair path would have


# `--fast` for RFD3: the levers that pass the fast accuracy bar but not (yet) the normal one. Same
# steps, recycles and designs; only kernels change. Each flag keeps its own env switch for A/B work,
# and set_fast(False) restores exactly what the environment asked for.
_FAST_SET = {"_TRANSITION_B8": "RFD3_TRANSITION_B8", "_PAIR_WS_LOFI": "RFD3_PAIR_WS_LOFI"}


def set_fast(on):
    """Turn RFD3's fast set on, or back to the environment's defaults. Call before building the
    modules: the fidelity is read when they take their compute kernel config."""
    g = globals()
    for name, env in _FAST_SET.items():
        g[name] = True if on else env_flag(env, False)


def set_process_z_collapse(on):
    """Toggle the collapsed process_z from a screen without going through the environment."""
    global _PROCESS_Z_COLLAPSE
    _PROCESS_Z_COLLAPSE = bool(on)


_PAIR_TRANSITION_H_CHUNK = 64                   # measured optimum at both hidden widths
_PAIR_TRANSITION_L1_BYTES = 138_000_000         # fits at 138 MB live, throws at 185
_PAIR_TRANSITION_MIN_W = 512                    # token pair is 704 wide; atom pair is 128
_TUNE_MATMUL_MIN_ATOMS = 2952
_TUNE_MATMUL_ENV = os.environ.get("RFD3_TUNE_MATMUL")
_TUNE_MATMUL = _TUNE_MATMUL_ENV == "1"


def set_tune_matmul_for_atoms(n_atoms):
    """Resolve matmul calibration for a target of `n_atoms`. The env var wins if it is set."""
    global _TUNE_MATMUL
    if _TUNE_MATMUL_ENV in ("0", "1"):
        _TUNE_MATMUL = _TUNE_MATMUL_ENV == "1"
    else:
        _TUNE_MATMUL = n_atoms > _TUNE_MATMUL_MIN_ATOMS
    return _TUNE_MATMUL
# Debug: re-check every cached config against ttnn's default on every call and print any
# divergence. Doubles the matmul work, so it is for locating a break, not for production.
_TUNE_AUDIT = env_flag("RFD3_TUNE_AUDIT", False)
# Debug: print what calibration decided per shape, and why.
_TUNE_LOG = env_flag("RFD3_TUNE_LOG", False)
_TUNE_MIN_GAIN = 1.05  # ignore a candidate that is not at least 5% faster than the default
# Don't calibrate a matmul whose default call is already this fast: an explicit program config
# costs a fixed amount per call that timing the matmul alone does not see, and under this floor
# that cost is the whole result. Replaces the old `D > 1` gate in `_tunable`, which was the same
# intent expressed on the wrong variable. 0.25 ms sits above the 0.02-0.13 ms matmuls of the
# 419-atom fixture and below the 0.53-2.74 ms ones of the 3359-atom fixture at D=1.
_TUNE_MIN_MS = 0.25
_TUNE_REPS = 3


def _mm_maxabs(a, b):
    """Device-side max|a-b| (the outputs are up to 0.5 GB; do not download them).

    `ttnn.max` with no `dim` does reduce globally, to a 0-d tensor
    (scripts/rfd3_port/probe_mm_maxabs_guard.py plants one outlier at the far corner of each real
    shape and finds it). The exactness hole was never here -- see `_calibrate_linear`.
    """
    d = ttnn.abs(ttnn.subtract(a, b))
    return float(ttnn.to_torch(ttnn.max(d)).float().abs().max())


def _mm_time(fn):
    import time
    dev = get_device()
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(_TUNE_REPS):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / _TUNE_REPS


def _mm_candidates(x, w, grid):
    """Batch-folded 1D configs, parameterised by the K-blocking `in0_block_w`.

    `out_block_h=1` because every measured winner had it and a larger value only eats the L1
    headroom `out_block_w` needs; `out_block_w` is swept because that is where the win lives
    (1x1 is a 0.45x *regression* on one of these shapes where 1xNt is 2.04x).
    """
    xs, ws = list(x.padded_shape), list(w.padded_shape)
    kt = xs[-1] // 32
    nt = ws[-1] // 32
    mt = 1
    for d in xs[:-1]:
        mt *= d
    mt //= 32                                   # all leading dims folded into M, in tiles
    per_core_m = -(-mt // (grid.x * grid.y))
    for bw in (d for d in (1, 2, 3, 4, 6, 8) if kt % d == 0):
        for obw in sorted({1, 2, nt}):
            if nt % obw:
                continue
            osw = max(d for d in (1, 2, 4) if obw % d == 0 and d <= obw)
            yield ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=bw,
                out_subblock_h=1, out_subblock_w=osw, out_block_h=1, out_block_w=obw,
                per_core_M=per_core_m, per_core_N=nt, fuse_batch=True, mcast_in0=False)


def _tunable(x, w):
    """Any `[.., M, K] @ [K, N]` with a batch-1 in1 can fold its leading dims into M.

    The gate used to be `xs[0] > 1`, i.e. only a genuine design batch, and that was set from a
    D=1/I=40 measurement where the matmuls are 0.02-0.13 ms, calibration reads 1.2-2.6x on them,
    and the end-to-end result was 6% SLOWER. That conclusion was right and the gate was on the
    wrong variable: what makes a tiny matmul not worth an explicit program config is its SIZE,
    not the design count. At I=250 the same four pair linears are 0.53-2.74 ms at D=1 and the
    calibrated fuse_batch config is 1.91-5.90x on them, every arm bitwise equal to the default
    (scripts/rfd3_port/p33_padding_verdict.py, perf/p33/padding_verdict_c3.json).

    So the size predicate lives in `_calibrate_linear` instead, as a floor on the DEFAULT call's
    own measured time (`_TUNE_MIN_MS`), which costs nothing extra because calibration has to time
    the default anyway. A shape under the floor bails before it builds a single candidate, so
    I=40 still takes the untuned path and I=250 at D=1 now gets its win.
    """
    xs, ws = list(x.padded_shape), list(w.padded_shape)
    if len(xs) < 3 or len(ws) < 2:
        return False
    return all(d == 1 for d in ws[:-2])


_CKC_FIELDS = ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode",
               "dst_full_sync_en", "throttle_level")


def _ckc_key(ckc):
    """A compute-kernel config as a hashable VALUE, for `_tuned_key`.

    The ttnn config object hashes and reprs by identity, so it cannot be a cache key itself:
    two configs with identical fields would key differently, and one config reused under a
    different fidelity would key the same. Read the fields.
    """
    return None if ckc is None else tuple(str(getattr(ckc, f, None)) for f in _CKC_FIELDS)


def _tuned_key(x, w, dtype, bias, core_grid, ckc):
    """`_TUNED_MM_CACHE`'s key. Shared, so a caller can ask what calibration decided for a shape.

    The compute-kernel config is part of the key because `_calibrate_linear` proves exactness
    *against the default call under that config*: a pinned `in0_block_w` that is bit-exact at
    HiFi4 with `fp32_dest_acc_en` has been checked at nothing else. Every live RFD3 caller passes
    the one `_default_compute_kernel_config()`, so this was latent rather than live -- but
    `build_token_initializer` takes a `compute_kernel_config` from outside, and a caller that
    passed a second fidelity would have silently reused configs pinned under the first.
    """
    return (tuple(list(x.padded_shape)), tuple(list(w.padded_shape)), x.dtype, dtype,
            bias is not None, core_grid, _ckc_key(ckc))


def _tuned_pinned(x, w, dtype, ckc, core_grid=BATCH_INVARIANT_GRID):
    """Did calibration pin a bit-exact program config for this shape? Read AFTER the call.

    `_tuned_linear` declines two ways -- no bit-exact candidate, or a default call under
    `_TUNE_MIN_MS` -- and both land as a `None` in the cache. A caller that wants to report
    which of its calls the pinned path actually served reads this rather than inferring it from
    the wall clock.
    """
    return _TUNED_MM_CACHE.get(_tuned_key(x, w, dtype, None, core_grid, ckc)) is not None


def _tuned_declined(x, w, dtype, ckc, core_grid=BATCH_INVARIANT_GRID):
    """Has calibration already tried this shape and found nothing to pin?

    Not the negation of `_tuned_pinned`, which cannot tell a shape calibration declined from one
    it has not seen yet. A caller whose lever is only worth taking WITH a pinned config uses this
    to stop taking it once the first call has populated the cache.
    """
    key = _tuned_key(x, w, dtype, None, core_grid, ckc)
    return key in _TUNED_MM_CACHE and _TUNED_MM_CACHE[key] is None


def _tuned_linear(x, w, *, bias=None, ckc=None, dtype=None, core_grid=BATCH_INVARIANT_GRID,
                  mem=None):
    """`ttnn.linear` with a calibrated, bit-exact program config where one helps.

    No `activation=`: ttnn wants a fused activation on the program config instead, and a fused
    activation cannot be pinned bit-exactly -- it rounds once, on the fp32 accumulator before the
    pack, where the shipped `activation="silu"` call rounds twice. Zero of nine
    `fused_activation=SILU` candidates match the shipped output at any of `fc1`'s eight live keys,
    against three of nine plain ones at every key (perf/p128/fc1_config_census.json). So the way
    a silu-gated linear reaches this function is to take the silu OUT of the call, which is what
    `Transition._swiglu`'s `fc1_split` does; it is not a precision change, because the shipped
    call was already a matmul followed by a separate eltwise silu.

    `mem` asks for the output in L1. It is deliberately NOT part of the cache key and it is
    honoured ONLY once an explicit program config has been chosen, because that is the whole
    reason it is safe: an L1 buffer competes for the same cores' L1 that ttnn's matmul heuristic
    budgets `in0_block_w` from, so a heuristic-picked matmul re-blocks K -- and re-rounds its bf16
    accumulation -- when its operands or its output move to L1. Measured, at this model's pair
    shape: 0.03125 at hidden=512 and 0.0 at hidden=256 (perf/p66/audit_perop.json). An explicit
    program config fixes the blocking, so the same config with an L1 output does the identical
    arithmetic and only the destination changes. If calibration declined this shape there is no
    config to pin, so the request is dropped and the output goes to DRAM rather than silently
    taking the heuristic's word for it.
    """
    kw = dict(compute_kernel_config=ckc, dtype=dtype)
    if bias is not None:
        kw["bias"] = bias
    if not _TUNE_MATMUL or not _tunable(x, w):
        return ttnn.linear(x, w, core_grid=core_grid, **kw)
    key = _tuned_key(x, w, dtype, bias, core_grid, ckc)
    if key not in _TUNED_MM_CACHE:
        _TUNED_MM_CACHE[key] = _calibrate_linear(x, w, kw, core_grid)
    pc = _TUNED_MM_CACHE[key]
    if pc is None:
        return ttnn.linear(x, w, core_grid=core_grid, **kw)
    if mem is not None:
        kw["memory_config"] = mem
    out = ttnn.linear(x, w, program_config=pc, **kw)
    if _TUNE_AUDIT:
        ref = ttnn.linear(x, w, core_grid=core_grid,
                          **{k: v for k, v in kw.items() if k != "memory_config"})
        m = _mm_maxabs(out, ref)
        ttnn.deallocate(ref)
        if m != 0.0:
            print(f"[tune-audit] DIVERGES {m:.6e}  x={key[0]} w={key[1]} bw={pc.in0_block_w} "
                  f"obw={pc.out_block_w} pcM={pc.per_core_M} pcN={pc.per_core_N}", flush=True)
    return out


def _mm_random_like(t, seed):
    """A random tensor with t's exact logical shape, dtype and layout."""
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(torch.randn(*list(t.shape), generator=g), dtype=t.dtype,
                           layout=ttnn.TILE_LAYOUT, device=get_device())


def _mm_drop(*ts):
    """Deallocate the calibration tensors that got built, tolerating the ones that did not."""
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _calibrate_linear(x, w, kw, core_grid):
    """Pick the fastest program config whose output is BITWISE equal to ttnn's default.

    Exactness is checked on RANDOM operands as well as the live ones, and the random check comes
    first. One cache entry is keyed on shapes, so it serves every weight of that shape -- all 24
    DiT blocks' `gain_w` and `bias_w` share one entry, for instance. A live activation/weight
    pair can be degenerate enough to hide a K-regrouping that other weights of the same shape do
    expose: p16 measured `in0_block_w=4` on `[8,160,384] @ [384,768]` reading bit-exact on the
    calibrating weight and then diverging by up to 3e-2 on its siblings, which is what failed
    batch invariance at L=1959/D=8. Random operands exercise every grouping across the whole
    output, so surviving them makes exactness a property of the shape rather than of one tensor.

    The two screens run as two PHASES, and calibration DECLINES rather than throws when its
    scratch does not fit. Both of those are about DRAM. Everything this function allocates is
    scratch -- one reference output per screen, plus a full random copy of each operand -- and it
    is live alongside the model's own working set. Running the screens interleaved held both
    references at once, i.e. 2|out| + |x| + |w| on top of the model, and that is what ended the
    first 1536-residue rfd3 design: the throw was at `rref`, asking for a 2717908992 B DRAM
    buffer against 2.45 GB free, 12405 target atoms, j10glx02 card 30, 2026-09-23
    (mgx-design-ceiling). Phase-splitting them drops the peak to |out| + |x| + |w|, since `ref`
    and `rref` never coexist, and a shape whose scratch still does not fit is simply not
    calibratable on this card, so it takes the default call.

    Declining is safe at any size and cannot change a result: the only configs this function ever
    returns are bitwise equal to the default it falls back to. A fuller card therefore runs the
    same arithmetic more slowly, never differently.

    The timed default below is deliberately NOT guarded. It holds one output at a time, which is
    exactly what the model's own call will hold, so an allocator refusal there is the model's
    ceiling rather than calibration's and masking it would hide a real wall behind a silent skip.
    """
    def skip(why):
        if _TUNE_LOG:
            print(f"[tune] x={tuple(x.padded_shape)} w={tuple(w.padded_shape)} {why}, SKIP",
                  flush=True)
        return None

    # Time the default BEFORE building anything else. Two reasons, both measured
    # (scripts/rfd3_port/p34_calib_cost.py, perf/p34/calib_cost.json): it is the size gate
    # `_tunable` no longer applies, and the setup below is 64.9% of calibration's whole cost,
    # because `_mm_random_like` draws and uploads a full-size operand -- 256 M elements for
    # `[8,250,250,512]`, 1.69 s of the 2.01 s that shape spends. A shape under the floor pays
    # one timed default and holds nothing.
    default_t = _mm_time(lambda: ttnn.linear(x, w, core_grid=core_grid, **kw))
    if default_t * 1e3 < _TUNE_MIN_MS:
        return skip(f"default={default_t * 1e3:8.3f} ms under {_TUNE_MIN_MS} ms floor")
    grid = get_device().compute_with_storage_grid_size()

    # Screen 1, random operands: every candidate that is bitwise equal on a full random draw.
    rx = rw = rref = None
    try:
        rx, rw = _mm_random_like(x, 0), _mm_random_like(w, 1)
        rref = ttnn.linear(rx, rw, core_grid=core_grid, **kw)
    except Exception as e:
        _mm_drop(rx, rw, rref)
        return skip(f"random screen does not fit ({e})")
    exact = []
    for pc in _mm_candidates(x, w, grid):
        try:
            if _mm_maxabs(ttnn.linear(rx, rw, program_config=pc, **kw), rref) == 0.0:
                exact.append(pc)
        except Exception:
            continue  # illegal L1 / subblock combinations are expected and simply skipped
    _mm_drop(rx, rw, rref)
    if not exact:
        # Nothing survived screen 1, so screen 2 has nothing to confirm and the answer is already
        # DEFAULT. Building `ref` anyway costs a full output for an empty loop, and this is not a
        # corner case at the sizes that matter: at 1536 residues every candidate for
        # `x=(1,1536,1536,128) w=(128,576)` refused, nine of them, because a second 2717908992 B
        # buffer does not fit beside the first (j10glx02 card 15, 2026-09-23, mgx-design-ceiling).
        # An empty survivor list is exactly the signal that DRAM is tight, which is the worst
        # moment to allocate one more output.
        return skip(f"default={default_t * 1e3:8.3f} ms, no candidate survived the random screen")

    # Screen 2, the live operands, on the survivors only: confirm exactness, then time.
    try:
        ref = ttnn.linear(x, w, core_grid=core_grid, **kw)
    except Exception as e:
        return skip(f"live screen does not fit ({e})")
    budget = default_t / _TUNE_MIN_GAIN
    best = None
    for pc in exact:
        try:
            if _mm_maxabs(ttnn.linear(x, w, program_config=pc, **kw), ref) != 0.0:
                continue
            t = _mm_time(lambda: ttnn.linear(x, w, program_config=pc, **kw))
        except Exception:
            continue
        if t < budget:
            best, budget = pc, t
    _mm_drop(ref)
    if _TUNE_LOG:
        chosen = (f"bw={best.in0_block_w} obw={best.out_block_w} pcM={best.per_core_M}"
                  if best is not None else "DEFAULT")
        gain = default_t / budget if best is not None else 1.0
        print(f"[tune] x={tuple(x.padded_shape)} w={tuple(w.padded_shape)} "
              f"default={default_t * 1e3:8.3f} ms  gain={gain:5.2f}x  {chosen}", flush=True)
    return best


# --- host-side feature helpers (pure torch; mirror upstream, deps stubbed) ----
def _collapse(x, L):
    return x.reshape((L, x.numel() // L))


def _build_relpos_onehot(f, r_max, s_max):
    """Host: build the [I,I, 2*(2*r_max+3)+(2*s_max+2)+1] one-hot cat for
    RelativePositionEncodingWithIndexRemoval. Returns float32 [I,I,C_in] for the
    device linear."""
    b_samechain = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
    b_same_entity = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
    num_tok_pos_bins = (2 * r_max + 2) + 1
    d_residue = torch.where(
        b_samechain,
        torch.clip(f["residue_index"].unsqueeze(-1) - f["residue_index"].unsqueeze(-2) + r_max, 0, 2 * r_max),
        2 * r_max + 1)
    b_sameresidue = f["residue_index"].unsqueeze(-1) == f["residue_index"].unsqueeze(-2)
    tok_distance = f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + r_max
    d_token = torch.where(
        b_samechain & b_sameresidue,
        torch.clip(tok_distance, 0, 2 * r_max),
        2 * r_max + 1)
    d_chain = torch.where(
        b_same_entity,
        torch.clip(f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + s_max, 0, 2 * s_max),
        2 * s_max + 1)
    A_relchain = F.one_hot(d_chain.long(), 2 * s_max + 2)
    unindexing = f["unindexing_pair_mask"]
    d_token[unindexing] = num_tok_pos_bins - 1
    d_residue[unindexing] = num_tok_pos_bins - 1
    A_relpos = F.one_hot(d_residue.long(), num_tok_pos_bins)
    A_reltoken = F.one_hot(d_token, num_tok_pos_bins)
    return torch.cat([A_relpos, A_reltoken, b_same_entity.unsqueeze(-1), A_relchain], dim=-1).to(torch.float32)


def _sinusoidal_embed(pos_rows, pos, valid_mask, n_freqs=32):
    """Host: SinusoidalDistEmbed inputs -> (sincos [R,L,2*n_freqs], V_LL [R,L,1]).

    `pos_rows` is a row block of `pos`; pass `pos` for both to get the whole thing. The
    row block matters on host as much as on device -- `sincos` is fp32 and 64 wide, so at
    RFD3's 4558-atom Wormhole cap the unblocked form is a 5.3 GB host allocation.
    """
    D = pos_rows.unsqueeze(-2) - pos.unsqueeze(-3)
    dist = torch.linalg.norm(D, dim=-1)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs)
    angles = dist.unsqueeze(-1) * freq
    sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(torch.float32)
    return sincos, valid_mask.to(torch.float32)


# Atom-pair tensors alive at once at the widest point of one row block of the token
# initializer's P_LL section: the sincos embedding counts twice (64 channels, so two
# 32-wide tile rows) and the rest are 1 or 16 channels, each of which the tile grid
# rounds up to 32. Counted off the code below with its deallocates in place, then rounded
# up by one so a future add does not silently overrun the budget.
_ATOM_PAIR_LIVE_ROWS = 6

#: [rows, blocks] of the last atom-pair section that ran, so a run can report whether the
#: budget actually cut anything instead of asserting it. One block means the unblocked path.
ATOM_PAIR_BLOCK_STATS = [0, 0]


def _pair_add(a, b, free_b=True):
    """``a + b`` for two atom-pair tensors, freeing the operands.

    An atom-pair add whose operands outlive it holds three copies of a tensor that is
    1.33 GB at RFD3's old Wormhole cap, and the section does eight of them.
    """
    out = ttnn.add(a, b)
    ttnn.deallocate(a)
    if free_b:
        ttnn.deallocate(b)
    return out


def _build_valid_mask(tok_idx):
    tokens, counts = torch.unique(tok_idx, return_counts=True)
    A = int(counts.max())
    return torch.arange(A, device=tok_idx.device)[None, :] < counts[:, None]


def _scatter_mean_pool(pairwise_atom, tok_idx, I):
    """Host: mean-pool [L,L,c] -> [I,I,c] (pairwise_mean_pool)."""
    if _POOL_SCATTER:
        t = tok_idx.long()
        rows = pairwise_atom.new_zeros(I, *pairwise_atom.shape[1:]).index_add_(0, t, pairwise_atom)
        sums = pairwise_atom.new_zeros(I, I, pairwise_atom.shape[2]).index_add_(1, t, rows)
        counts = torch.bincount(t, minlength=I).to(sums.dtype)
        return sums / torch.outer(counts, counts).clamp(min=1).unsqueeze(-1)
    onehot = F.one_hot(tok_idx.long(), num_classes=I).to(torch.float32)
    temp = torch.einsum("ia,bacd->bicd", onehot.T, pairwise_atom.unsqueeze(0))
    sums = torch.einsum("cj,bicd->bijd", onehot, temp)
    counts = onehot.sum(0)
    pc = torch.outer(counts, counts).clamp(min=1).unsqueeze(0)
    return (sums / pc.unsqueeze(-1)).squeeze(0)


# --- ttnn helpers ----------------------------------------------------------
# ttnn.from_torch(layout=TILE_LAYOUT) tilizes on the HOST, single-threaded, and that cost
# dominates every large upload in this model: one 6.9M-element atom-pair tensor costs 45 ms
# at 250 residues and a whole diffusion step is only ~600 ms. Uploading row-major (already
# cast to the target dtype, so the PCIe transfer is half the bytes for bf16) and tilizing on
# device is 2.8-8.5x faster and bit-exact -- verified elementwise-equal on every large shape
# this model uploads. Small tensors keep the host path; the extra device op is not worth it
# below about a tile-grid of data.
_DEVICE_TILIZE_MIN_ELEMENTS = 1 << 16

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
}


def _tt(x, dev, dtype=ttnn.bfloat16):
    torch_dtype = _TORCH_DTYPE.get(dtype)
    if torch_dtype is not None and x.numel() >= _DEVICE_TILIZE_MIN_ELEMENTS:
        row_major = ttnn.from_torch(x.to(torch_dtype), layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=dev, dtype=dtype)
        return ttnn.to_layout(row_major, ttnn.TILE_LAYOUT)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def _dev(x, dev, dtype=ttnn.bfloat16):
    """`_tt` for a host tensor; a tensor already on the card passes through (cast if needed)."""
    if isinstance(x, ttnn.Tensor):
        return x if x.dtype == dtype else ttnn.typecast(x, dtype)
    return _tt(x, dev, dtype)


# The diffusion loop re-uploads several tensors that cannot have changed. p41 measured 32 _tt
# calls per step costing 14.6 ms at 3359 atoms, of which 7.5 ms is byte-identical on every one
# of the 199 steps: the trunk pair stream Z_init_II (5.14 ms/step, 2 x [1,250,250,128]), the
# downcast additive mask (1.80), and the atom/token conditioning the decoder's downcast reads
# (0.59). Uploading them once is bit-exact by construction -- the same bytes through the same
# deterministic cast produce the same device tensor -- so the gate is the design's CIF sha.
_UPLOAD_CACHE: dict = {}
# Bounded so a callsite whose data DOES change per step degrades to today's behaviour instead
# of growing an entry per step. 24 covers the annotated sites several times over.
_UPLOAD_CACHE_MAX = 24


def _tt_cached(x, dev, dtype=ttnn.bfloat16):
    """`_tt`, keeping the device tensor when the host bytes provably have not changed.

    Keyed on the storage address plus shape/stride/dtype and torch's own version counter, which
    any in-place write bumps. The entry holds a reference to ``x``, so its storage cannot be
    freed and the address handed to a different tensor underneath a live key; an evicted entry
    drops that reference and its key with it, so a recycled address cannot false-hit either.
    A consumer that deallocated the cached tensor is detected rather than trusted.
    """
    key = (dev.id(), x.data_ptr(), tuple(x.shape), tuple(x.stride()),
           str(x.dtype), str(dtype), x._version)
    ent = _UPLOAD_CACHE.get(key)
    if ent is not None and ent[1].is_allocated():
        return ent[1]
    out = _tt(x, dev, dtype)
    if len(_UPLOAD_CACHE) >= _UPLOAD_CACHE_MAX:
        _UPLOAD_CACHE.pop(next(iter(_UPLOAD_CACHE)))
    _UPLOAD_CACHE[key] = (x, out)
    return out


def _pair_transition_chunk_h(w_pad, hidden, height, residents=2):
    """Rows of the pair tensor one L1-resident SwiGLU chunk may cover.

    Live per chunk is `b` + `m`, both [1, h, w_pad, hidden] bf16. The batch is not in the
    formula because `Transition.__call__` slices it: one `_swiglu` call covers one batch
    element, so the footprint per call does not depend on the batch and neither does h.

    The cap used to divide by the batch instead. That held the measured-safe 138 MB at any
    batch -- it is what fixed the b=2 OOM, request 92 274 688 B against 68.4 MB free -- but it
    made h a function of the batch, and h selects which of several bit-different answers the
    chunked path produces (state doc §15.2/§15.3: four heights, four fold digests). So a b=2
    design did not reproduce the b=1 structure at 62.7 % of (size, hidden) pairs, in a port
    that carries `BATCH_INVARIANT_GRID` because batch invariance is a requirement here.
    Slicing the batch keeps the bound and the structure: h is now a pure function of
    (w_pad, hidden, height) and every call runs the b=1 chunk plan.

    `residents=3` prices the same budget with `fc1`'s output in L1 as well (`_FC1_SPLIT_SILU`).
    """
    cap = _PAIR_TRANSITION_L1_BYTES // (2 * residents * w_pad * hidden)
    return max(1, min(height, _PAIR_TRANSITION_H_CHUNK, cap))


def _pair_transition_l1_fits(w_pad, hidden, h, residents=2):
    """Whether `residents` [1, h, w_pad, hidden] bf16 chunks fit this part's L1 at once.

    The chunk height `h` is fixed before this is asked and this cannot move it, which is the
    whole design: `h` selects which of several bit-different answers the chunked path produces
    (`_pair_transition_chunk_h`), so a capacity gate that shrank it would change the structure
    the model designs. What it decides instead is only the DESTINATION -- L1 or DRAM -- and a
    destination cannot change the arithmetic, because `_tuned_linear` honours `mem` only on a
    pinned program config and `ttnn.multiply` is elementwise (`_swiglu`).

    `_pair_transition_chunk_h`'s own budget cannot do this job. It is a constant, 138000000 B,
    and a Wormhole part has 100470528 B of L1, so at every width up to 1053 tokens the 64-row
    cap binds first and the budget declines nothing. That is how 768 residues came to ask for
    two 700416 B-per-bank buffers out of a 1395424 B bank and miss by 5408 B: 0.39 %, which made
    the boundary a coin flip rather than a wall (`state/ceiling-rfd3.md`). Priced against the
    part's real L1 the same shape declines the residency and runs in DRAM instead of throwing.

    `row_block` rather than a second copy of the same division: it is the shared helper the
    atom-pair section already uses, `per_row_bytes` is what one row costs summed over the
    tensors live at the same moment, and its 32-alignment is exact at the 64-row cap this site
    asks for.
    """
    per_row = 2 * residents * int(w_pad) * int(hidden)
    return row_block(per_row, l1_resident_budget_bytes(), cap=h) >= h


def _pair_transition_slices(batch, height, h):
    """Every (batch index, row start, row stop) chunk the L1-resident path covers.

    The batch extent is always 1, which is the whole point: it is what takes the batch out of
    `_pair_transition_chunk_h` and therefore out of the model's arithmetic. At b=1 this is the
    row split the unbatched path always did.
    """
    return [(b, s, min(s + h, height))
            for b in range(max(1, batch)) for s in range(0, height, h)]


def _pair_transition_join(parts, batch):
    """Reassemble `_pair_transition_slices`' chunk outputs: rows on dim 1, batch on dim 0.

    At b=1 this is the single dim-1 concat the unbatched path always did, in the same order,
    which is why the b=1 digest cannot move.
    """
    n = len(parts) // max(1, batch)
    rows = [ttnn.concat(parts[i:i + n], dim=1) for i in range(0, len(parts), n)]
    for p in parts:
        ttnn.deallocate(p)
    if len(rows) == 1:
        return rows[0]
    out = ttnn.concat(rows, dim=0)
    for r in rows:
        ttnn.deallocate(r)
    return out


def _tt_idx(indices, dev):
    """Upload a flat gather/scatter index tensor once (uint32, row-major)."""
    return ttnn.from_torch(indices.to(torch.int32).reshape(1, -1),
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)


# --- ttnn Transition (RFD3: RMSNorm + silu-gated SwiGLU, keys layer_norm_1/linear_1-3) --
class Transition(Module):
    def __init__(self, state_dict, ckc, c, n, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.norm_w = self.torch_to_tt("layer_norm_1.weight", dtype=self.dtype)
        self.fc1_w = self.torch_to_tt("linear_1.weight", dtype=self.dtype)
        self.fc2_w = self.torch_to_tt("linear_2.weight", dtype=self.dtype)
        self.fc3_w = self.torch_to_tt("linear_3.weight", dtype=self.dtype)

    def __call__(self, x):
        """Whole-tensor by default; row blocks on the token pair tensor, L1-resident where they fit.

        The row block's height comes from `_pair_transition_chunk_h` and its destination from
        `_pair_transition_l1_fits`, and the split matters: the height is part of the arithmetic
        and the destination is not, so only the destination may answer to a capacity budget.

        See `_PAIR_TRANSITION_L1` for the measurements. `RFD3_PAIR_TRANSITION_L1=0` restores
        the whole-tensor path op for op at 3-to-63-row tails; at 1- and 2-row tails (22 of 689
        production sizes, e.g. H=513/514) it differs by 2.44e-4, one bf16 ULP at this magnitude
        (`perf/p131/tail_sweep.json`). Not chased further: the gap is sub-ULP rounding, not a
        wrong result.
        """
        if _PAIR_WS and len(x.shape) == 4:
            out = self._weight_stationary(x)
            if out is not None:
                return out
        if not (_PAIR_TRANSITION_L1 and len(x.shape) == 4
                and x.shape[2] >= _PAIR_TRANSITION_MIN_W):
            if _FC1_SPLIT_SILU and len(x.shape) == 4:
                # Decline route 3 of 3: no chunked site at all, so the split is never reached.
                # Censused rather than inferred from an unchanged digest -- a rung whose verdict
                # is "nothing happened" needs the guard to SAY it declined
                # (`negative-control-must-break-what-check-reads`).
                k = "w=%d no-chunked-site" % int(x.padded_shape[2])
                FC1DECLINES[k] = FC1DECLINES.get(k, 0) + 1
                FC1STATS[1] += 1
            if _TRANSITION_B8 and len(x.shape) == 4 and x.shape[0] > 1:
                # With a bfp8 hidden, ttnn's heuristic fc3 program for a [B, I, I, hidden] input
                # sizes its circular buffers from the batch: at B=8 (the CLI default) and I~150 it
                # asked 1564704 B of a Wormhole core's 1499136 B of L1 and threw. One element at a
                # time runs the batch-1 program, so a design comes out the same at any batch.
                B, I, J, C = (int(d) for d in x.shape)
                parts = []
                for b in range(B):
                    xb = ttnn.slice(x, [b, 0, 0, 0], [b + 1, I, J, C])
                    parts.append(self._swiglu(xb, None))
                    ttnn.deallocate(xb)
                out = ttnn.concat(parts, dim=0)
                for p in parts:
                    ttnn.deallocate(p)
                return out
            return self._swiglu(x, None)
        H, hidden = x.shape[1], int(self.fc1_w.shape[-1])
        w_pad = int(x.padded_shape[2])
        h = _pair_transition_chunk_h(w_pad, hidden, H)
        # Admit `fc1`'s output to L1 only where the third resident leaves the chunk height
        # ALONE. One L1 budget divided by three instead of two shrinks h, and h is not a
        # footprint detail: at 514 tokens h=64 is the only height of 64, 63, 59 and 53 that does
        # not reproduce the whole-tensor path (2.44e-4, one bf16 ULP per call, compounding over
        # 200 diffusion steps into a different structure -- state doc §15.3). So a lever that
        # moves h moves the answer, whatever it does to the clock. An equal chunk COUNT is not
        # enough for that: 49 sizes at hidden=512 keep the count and move the height, and the
        # first fold at one of them moved the CIF digest (§15.2).
        # The lever is all-or-nothing: `fc1`'s output goes to L1 or its silu stays fused on the
        # call. The split without the residency is 1.11x at 704 tokens and 0.91x at 1024, a LOSS,
        # so taking it only where the third resident is free keeps every measured gain and cannot
        # regress. At an equal height that is every size at hidden=256, and 161 of the 689
        # chunked sizes at hidden=512.
        h3 = _pair_transition_chunk_h(w_pad, hidden, H, residents=3)
        # Does this part's L1 hold the two residents `_swiglu` keeps live, `b` and `m`? Priced
        # against the part rather than against a constant, and it decides the destination only:
        # `h` is already fixed above and this cannot move it, so the chunk plan and therefore
        # the arithmetic are the same either way. Declining costs the DRAM round trip the
        # residency exists to avoid; not declining costs the fold, which is what 768 residues
        # did on a Wormhole part.
        shape = "tensor_rows=%d w=%d hidden=%d" % (H, w_pad, hidden)
        # The budget decides, and a refusal this shape has already paid for overrides it: a
        # granted residency the allocator refused once is not granted again, so the census and
        # the split read the same verdict `_swiglu_resident` will act on.
        budget_ok = _pair_transition_l1_fits(w_pad, hidden, h)
        resident = budget_ok and shape not in PTL1REFUSED
        mem = ttnn.L1_MEMORY_CONFIG if resident else None
        PTL1STATS[0 if resident else 1] += 1
        if not resident:
            key = "%s h=%d %s" % (shape, h, "l1-holds-fewer-than-two" if not budget_ok
                                  else "l1-refused-at-runtime")
            PTL1DECLINES[key] = PTL1DECLINES.get(key, 0) + 1
        # `fc1`'s output is a third resident in the same L1, so the split cannot outlive the
        # residency it shares. Nothing else about the split moves: the `h3 == h` test still
        # decides it wherever L1 holds two, so no size that takes the split today loses it.
        split = _FC1_SPLIT_SILU and h3 == h and resident
        if _FC1_SPLIT_SILU and not split:
            # Decline route 2 of 3, and the only one that was invisible. Two ways in, kept apart
            # in the census because they call for different levers: the third resident would move
            # the chunk height (528 of the 689 chunked sizes at hidden=512), or L1 does not hold
            # even the two the gated product needs, in which case there is no residency to join.
            k = ("%s chunk-height-would-move %d->%d" % (shape, h, h3) if resident
                 else "%s no-residency-to-join" % shape)
            FC1DECLINES[k] = FC1DECLINES.get(k, 0) + 1
            FC1STATS[1] += 1
        # Slice lazily rather than `ttnn.chunk`, which materialises a second full copy of the
        # pair tensor up front. The 685-row tail is ragged at every h and gets its own shape,
        # hence its own program-config cache entry, which is what keeps it exact.
        parts = []
        for b, s, e in _pair_transition_slices(int(x.shape[0]), H, h):
            c = x[b:b + 1, s:e]
            # The blocks AFTER a refusal in the same tensor read the memo too, not just the next
            # tensor. Measured on a Galaxy chip at 1024 residues: the first hidden=256 tensor had
            # all 16 of its blocks ask for an L1 residency the allocator had already refused
            # once, so 15 doomed allocations and 15 throws bought the same DRAM path the memo
            # would have chosen. It is also what kept the census honest -- `granted` is
            # incremented once per tensor and was being decremented once per refused block, and
            # came back as -13.
            blk = mem if shape not in PTL1REFUSED else None
            parts.append(self._swiglu_resident(
                c, blk, ttnn.L1_MEMORY_CONFIG if split and blk is not None else None, shape))
            ttnn.deallocate(c)
        return _pair_transition_join(parts, int(x.shape[0]))

    def add_into(self, x):
        """`x + self(x)`. Under RFD3_PAIR_WS_RESID the weight-stationary program adds into `x` in place and
        returns it, so the caller must hold no other reference to `x` it still needs."""
        if _PAIR_WS and _PAIR_WS_RESID and len(x.shape) == 4:
            out = self._weight_stationary(x, resid=True)
            if out is not None:
                return out
        return ttnn.add(x, self(x))

    def add_rows(self, x, out, rows):
        """`out = x + self(x)` on the row tiles of `rows` (pair_swiglu_ws.row_list) only; `out` may be `x`. None where
        the weight-stationary program with its RMS prologue declines, and then nothing was written."""
        if not (_PAIR_WS and len(x.shape) == 4):
            return None
        return self._weight_stationary(x, resid=True, out=out, rows=rows)

    def _weight_stationary(self, x, resid=False, out=None, rows=None):
        """`_swiglu` through `pair_swiglu_ws`, every batch element in one program (rows are tile-aligned per
        element, so the flattened tensor splits into whole row tiles); None where it declines."""
        if getattr(self, "_ws12", None) is None:
            self._ws12 = ttnn.concat([self.fc1_w, self.fc2_w], dim=1)
            self._ws3 = self.fc3_w
            if _PAIR_WS_B8 or _PAIR_WS_LOFI:
                self._ws12 = ttnn.typecast(self._ws12, ttnn.bfloat8_b)
                self._ws3 = ttnn.typecast(self.fc3_w, ttnn.bfloat8_b)
        r = (x if out is None else out) if resid else None
        if not _WS.eligible(x, self._ws12, self._ws3, resid=r):
            return None
        from ..tenstorrent import lofi_allowed   # Wormhole: the bfp8 weights at HiFi2 instead
        fid = (ttnn.MathFidelity.LoFi if _PAIR_WS_LOFI and lofi_allowed()
               else ttnn.MathFidelity.HiFi2 if _PAIR_WS_B8 or _PAIR_WS_LOFI
               else self.compute_kernel_config.math_fidelity)
        # The input RMSNorm as the program's prologue: bit-identical to ttnn.rms_norm first, and the normalised
        # tensor never goes to DRAM. p150a @1350 2.65 -> 2.39 ms (hidden 256), 4.38 -> 4.09 (512); flat on WH.
        if _WS.eligible(x, self._ws12, self._ws3, ln=(self.norm_w, None), resid=r):
            return _WS.pair_swiglu(x, self._ws12, self._ws3, fid, ln=(self.norm_w, None, 1e-6), resid=r,
                                  pack_silu=_PAIR_WS_PSILU, rows=rows,
                                  resid_src=None if r is None or r is x else x)
        if rows is not None:
            return None
        xn = ttnn.rms_norm(x, weight=self.norm_w, epsilon=1e-6, compute_kernel_config=self.compute_kernel_config)
        out = _WS.pair_swiglu(xn, self._ws12, self._ws3, fid, resid=r, pack_silu=_PAIR_WS_PSILU)
        ttnn.deallocate(xn)
        return out

    def _swiglu_resident(self, x, mem, fc1_mem, key):
        """`_swiglu`, dropping the L1 residency rather than the fold if the allocator refuses.

        `_pair_transition_l1_fits` is a pure function of the part's L1 and the block's shape, and
        that is deliberate -- a residency decision that read live occupancy would make the chunk
        plan depend on allocator history. The cost of being pure is that it can grant a residency
        an allocator holding something it did not price then refuses.

        `_with_dram_narrowing` is the same guard for DRAM row blocks and its answer is to halve
        the block, which is not available here: the block height is part of this op's arithmetic
        (`_pair_transition_chunk_h`). Dropping the residency is not. It is the same chunk plan
        written to DRAM, measured byte-identical at 768 and 1024 total residues, so the retry
        returns what the first attempt would have returned. That also makes the mixed case safe:
        when a refusal lands mid-tensor, the blocks before it ran in L1 and the blocks after it
        run in DRAM, and the concatenation is the same either way.

        A refusal is remembered per shape, and `__call__` reads that memo before it grants the
        next tensor at the same shape, so only the first block of the first such tensor pays for
        it -- and the census says `l1-refused-at-runtime` rather than reporting a grant that then
        ran in DRAM.

        The retry runs OUTSIDE the handler on purpose. The failed frame's own `xn`, `a` and `b`
        are still referenced by the exception's traceback, so retrying inside `except` would
        re-run with those buffers still holding the L1 that was just refused. Letting the handler
        exit drops the traceback and frees them first.
        """
        refused = False
        try:
            return self._swiglu(x, mem, fc1_mem=fc1_mem)
        except RuntimeError as exc:
            # `_dram_oom` matches the allocator's own "Out of Memory" wording, which it uses for
            # both spaces -- it is named for its first caller, not for the space it can read.
            # Anything else (a compile error, a bad shape) re-raises rather than being retried
            # and reported as a residency problem.
            if mem is None or not _dram_oom(exc):
                raise
            refused = True
        if refused:
            PTL1REFUSED[key] = True
            PTL1STATS[0] -= 1
            PTL1STATS[1] += 1
            k = key + " l1-refused-at-runtime"
            PTL1DECLINES[k] = PTL1DECLINES.get(k, 0) + 1
        return self._swiglu(x, None, fc1_mem=None)

    def _swiglu(self, x, mem, fc1_mem=None):
        """RMSNorm + silu-gated SwiGLU. `mem=None` keeps every intermediate in DRAM.

        With `mem` set, `fc2`'s output and the gated product stay in L1, so `b` and `m` are never
        written to DRAM and never read back: 1975 of the 3951 MB an H=512 call moves.

        `x_norm` stays in DRAM on purpose, and that is part of why this is bit-exact: a
        heuristic-blocked matmul re-blocks K -- and re-rounds its bf16 accumulation -- when its
        operands or its output move to L1 (measured 0.03125 at hidden=512, perf/p66/audit_perop.json).
        `fc2` and `fc3` are immune because `_tuned_linear` pins their blocking. `fc1` used to be
        the exception: its silu was fused onto the call, ttnn wants a fused activation on the
        program config, and no bit-exact config for a fused activation exists, so it stayed
        heuristic-blocked with a DRAM output.

        `fc1_mem` retires that exception rather than working around it. The shipped
        `activation="silu"` call is a matmul followed by a separate eltwise silu -- it rounds
        twice, and `ttnn.silu(ttnn.linear(x, w))` is bit-identical to it at all eight live keys
        (`_FC1_SPLIT_SILU`). With the silu out of the call `fc1` is an ordinary linear, takes a
        pinned config like its siblings, and its output goes to L1 with `b` and `m`.

        Setting `fc1_mem` is the WHOLE lever, both halves at once, and that is deliberate. The
        split on its own -- pinned config, output still in DRAM -- is 1.11x at 704 tokens and
        0.91x at 1024, a LOSS, because the extra DRAM round trip and the second dispatch eat the
        config's gain (perf/p130/fc1_above_free_window.json). So `__call__` asks for the split
        only where the L1 residency comes with it, and the one shape that slips through that
        guard, a shape calibration declines, unsplits itself on its second call.
        """
        xn = ttnn.rms_norm(x, weight=self.norm_w, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        hid = ttnn.bfloat8_b if _TRANSITION_B8 and len(x.shape) == 4 else self.dtype
        if fc1_mem is not None:
            # Only the chunked pair path asks for the split, so only it has a rank-4 `xn` to key
            # a census row on -- `Transition` also serves 2D and 3D sites.
            rows = "rows=%d w=%d hidden=%d" % (int(xn.padded_shape[1]), int(xn.padded_shape[2]),
                                               int(self.fc1_w.shape[-1]))
            if _tuned_declined(xn, self.fc1_w, self.dtype, self.compute_kernel_config):
                # Calibration has already tried this shape and has nothing to pin, so all the
                # split would buy here is its own overhead. The first call pays to find out, no
                # later one does. Both paths are bit-identical, so this never moves a digest. The
                # R3 ragged tail is the live example: 0.036-0.073 ms, under `_TUNE_MIN_MS`.
                k = rows + " no-pinned-config"
                FC1DECLINES[k] = FC1DECLINES.get(k, 0) + 1
                FC1STATS[1] += 1
                fc1_mem = None
        if fc1_mem is None:
            a = ttnn.linear(xn, self.fc1_w, activation="silu",
                             compute_kernel_config=silu_ckc(self.compute_kernel_config),
                             dtype=hid, core_grid=BATCH_INVARIANT_GRID)
        else:
            a = _tuned_linear(xn, self.fc1_w, ckc=self.compute_kernel_config, dtype=hid,
                              core_grid=BATCH_INVARIANT_GRID, mem=fc1_mem)
            a, pre = ttnn.silu(a, memory_config=fc1_mem), a
            ttnn.deallocate(pre)
            # Provenance. An `on` arm that served nothing is an A/A wearing an A/B's label, so the
            # lever counts its own calls rather than leaving it to be inferred from the clock.
            # A shape's FIRST call lands here even if calibration then declines it, which is what
            # the `no-pinned-config` census row is: the one call that paid to find out.
            FC1STATS[0] += 1
            served = _tuned_pinned(xn, self.fc1_w, self.dtype, self.compute_kernel_config)
            c, k = ((FC1SERVED, rows) if served
                    else (FC1DECLINES, rows + " no-pinned-config"))
            c[k] = c.get(k, 0) + 1
        b = _tuned_linear(xn, self.fc2_w, ckc=self.compute_kernel_config,
                          dtype=hid, core_grid=BATCH_INVARIANT_GRID, mem=mem)
        ttnn.deallocate(xn)
        kw = {} if mem is None else {"memory_config": mem}
        if hid != self.dtype:
            kw["dtype"] = hid
        m = ttnn.multiply(a, b, **kw)
        ttnn.deallocate(b)
        refused = False
        try:
            out = _tuned_linear(m, self.fc3_w, ckc=self.compute_kernel_config,
                                dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        except RuntimeError as exc:
            # ttnn's heuristic program for a bfp8 `m` can ask more circular-buffer L1 than a
            # Wormhole core has (1630624 of 1499136 B whole-tensor at 685 tokens, perf
            # gap_pt_bench on .114). The throw comes at program build, before anything runs, so
            # fc3 is redone on a bf16 copy of `m`, the dtype the exact path feeds it.
            if hid == self.dtype or "beyond max L1 size" not in str(exc):
                raise
            refused = True
        if refused:
            m16 = ttnn.typecast(m, self.dtype)
            out = _tuned_linear(m16, self.fc3_w, ckc=self.compute_kernel_config,
                                dtype=self.dtype, core_grid=CORE_GRID_MAIN)
            ttnn.deallocate(m16)
        ttnn.deallocate(m)
        return out


# --- Pairformer attention (AttentionPairBiasPairformerDeepspeed): unconditioned MHA,
# per-head kq_norm, pair bias from RMSNorm(Z)+0, gate, output linear. NO mask (full I×I). -
class PairformerAttention(Module):
    def __init__(self, state_dict, ckc, c_a=384, c_z=128, n_head=16, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.n_head = n_head
        self.head_dim = c_a // n_head  # 24
        self.ln_1_w = self.torch_to_tt("ln_1.weight", dtype=self.dtype)
        self.to_q_w = self.torch_to_tt("to_q.weight", dtype=self.dtype)
        self.to_q_b = self.torch_to_tt("to_q.bias", dtype=self.dtype)
        self.to_k_w = self.torch_to_tt("to_k.weight", dtype=self.dtype)
        self.to_k_ln = self.torch_to_tt("to_k.ln.weight", dtype=self.dtype)
        self.to_v_w = self.torch_to_tt("to_v.weight", dtype=self.dtype)
        self.to_v_ln = self.torch_to_tt("to_v.ln.weight", dtype=self.dtype)
        self.ln_0_w = self.torch_to_tt("ln_0.weight", dtype=self.dtype)
        self.to_b_w = self.torch_to_tt("to_b.weight", dtype=self.dtype)
        self.to_g_w = self.torch_to_tt("to_g.0.weight", dtype=self.dtype)
        self.to_a_w = self.torch_to_tt("to_a.weight", dtype=self.dtype)

    def __call__(self, s, z):
        # s: [1,I,384], z: [1,I,I,128]
        ckc = self.compute_kernel_config
        a = ttnn.rms_norm(s, weight=self.ln_1_w, epsilon=1e-6,
                          compute_kernel_config=ckc)
        B, I = s.shape[0], s.shape[1]
        if _PF_SDPA:
            return self._sdpa(a, z, B, I)
        q = ttnn.linear(a, self.to_q_w, bias=self.to_q_b,
                          compute_kernel_config=self.compute_kernel_config,
                          dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        k = ttnn.linear(a, self.to_k_w, compute_kernel_config=self.compute_kernel_config,
                          dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        k = ttnn.rms_norm(k, weight=self.to_k_ln, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        v = ttnn.linear(a, self.to_v_w, compute_kernel_config=self.compute_kernel_config,
                           dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        v = ttnn.rms_norm(v, weight=self.to_v_ln, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        # split heads: [1,I,384] -> [1,I,16,24] -> [1,16,I,24]
        q = ttnn.reshape(q, (B, I, self.n_head, self.head_dim))
        k = ttnn.reshape(k, (B, I, self.n_head, self.head_dim))
        v = ttnn.reshape(v, (B, I, self.n_head, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        # pair bias: [1,I,I,128] -> rms_norm -> linear -> [1,I,I,16] -> [1,16,I,I]
        z = ttnn.rms_norm(z, weight=self.ln_0_w, epsilon=1e-6,
                           compute_kernel_config=self.compute_kernel_config)
        n_key = align_tile(I)
        bias_f = self._pair_bias_dpb(z, I, n_key) if _pf_dpb_on() else None
        if bias_f is None:
            bias = _tuned_linear(z, self.to_b_w, ckc=self.compute_kernel_config,
                                 dtype=self.dtype, core_grid=CORE_GRID_MAIN)
            bias = ttnn.permute(bias, (0, 3, 1, 2))  # [1,16,I,I]
            bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
            bias_f = pad_axis(bias_f, n_key, 3, -1e4)
        # Manual attention (SDPA forbids head_dim=24 padding); bf16 softmax matches the
        # reference (autocast bf16). softmax over keys (dim=-1).
        #
        # Extend the KEY axis out to a tile multiple, the same way the DiT's attention does at
        # :1600 -- zero keys and values scored at -1e4, so their weight is exactly 0 after exp.
        # A ragged key axis here is not a wrong answer (softmax_generic declines it and
        # ttnn.softmax masks its own tail) but it is the fused softmax going dark on every call
        # at every design length that is not a multiple of 32: censused 6 of 6. See
        # tt_bio/token_axis.py and PLAYBOOKS.md §MODEL 2b.
        k = pad_axis(k, n_key, 2, 0.0)
        v = pad_axis(v, n_key, 2, 0.0)
        kt = ttnn.permute(k, (0, 1, 3, 2))                 # [1,16,24,n_key]
        sc = ttnn.matmul(q, kt, compute_kernel_config=ckc)  # [1,16,I,n_key]
        ttnn.deallocate(kt)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
        sc = ttnn.multiply(sc, self.head_dim ** -0.5)
        sc = ttnn.add(sc, bias_f)
        ttnn.deallocate(bias_f)
        # fp32 softmax reduction, packing bf16 straight out of DST -- one kernel, and the
        # typecast it replaces read the whole score tensor back out of DRAM.
        attn_bf = softmax_generic.softmax_bf16(sc, self.dtype)
        o = attn_value_matmul(attn_bf, v, ckc, self.dtype)  # [1,16,I,24]
        ttnn.deallocate(attn_bf)
        # merge heads: [1,16,I,24] -> [1,I,384], then gate. The gate is elementwise, so
        # applying it after the merge is bit-identical and saves splitting `g` into heads.
        o = _merge_heads(o, (B, I, self.n_head * self.head_dim))
        g = ttnn.linear(a, self.to_g_w, compute_kernel_config=self.compute_kernel_config,
                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        o = ttnn.multiply(o, ttnn.sigmoid(g))
        out = ttnn.linear(o, self.to_a_w, compute_kernel_config=self.compute_kernel_config,
                            dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        return out


    def _padded(self):
        """q/k/v/g (and q's bias) with zero output columns after each 24-wide head, to_a with zero input rows, so every
        head is a tile wide. k/v's rms_norm over the padded width sees the same sum of squares over more columns, so
        its weight is scaled by sqrt(c/padded) and its eps by c/padded (as RFD3AtomBlock._padded_head_weights)."""
        w = getattr(self, "_pw", None)
        if w is not None:
            return w
        hd, dp, H = self.head_dim, align_tile(self.head_dim), self.n_head
        s = hd / dp
        W = self.weights

        def cols(x):                                                     # [..., H*hd] -> [..., H*dp]
            return F.pad(x.reshape(*x.shape[:-1], H, hd), (0, dp - hd)).reshape(*x.shape[:-1], H * dp)

        def up(x):
            return ttnn.from_torch(x.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=self.device,
                                   dtype=self.dtype)
        w = {k: up(cols(W[n].t())) for k, n in (("q", "to_q.weight"), ("k", "to_k.weight"), ("v", "to_v.weight"),
                                                  ("g", "to_g.0.weight"))}
        w["q_b"] = up(cols(W["to_q.bias"].reshape(1, -1)))
        w["k_ln"] = up(cols(W["to_k.ln.weight"].reshape(1, -1) * s ** 0.5))
        w["v_ln"] = up(cols(W["to_v.ln.weight"].reshape(1, -1) * s ** 0.5))
        w["a"] = up(cols(W["to_a.weight"]).t())
        w["eps"] = 1e-6 * s
        self._pw = w
        return w

    def _sdpa(self, a, z, B, I):
        """The attention through the fused SDPA on tile-wide heads (_PF_SDPA)."""
        ckc, dt, w = self.compute_kernel_config, self.dtype, self._padded()
        lin = dict(compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        n_key, hp = align_tile(I), self.n_head * align_tile(self.head_dim)
        q = ttnn.linear(a, w["q"], bias=w["q_b"], **lin)
        k = ttnn.rms_norm(ttnn.linear(a, w["k"], **lin), weight=w["k_ln"], epsilon=w["eps"],
                          compute_kernel_config=ckc)
        v = ttnn.rms_norm(ttnn.linear(a, w["v"], **lin), weight=w["v_ln"], epsilon=w["eps"],
                          compute_kernel_config=ckc)
        qkv = ttnn.reshape(ttnn.concat([q, k, v], dim=-1), (B, 1, I, 3 * hp))
        for x in (q, k, v):
            ttnn.deallocate(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.n_head, num_kv_heads=self.n_head, transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        mask = self._sdpa_mask_rms(z, I, n_key) if _pf_rmsb_on() else None
        if mask is None:
            z = ttnn.rms_norm(z, weight=self.ln_0_w, epsilon=1e-6, compute_kernel_config=ckc)
            mask = self._sdpa_mask(z, I, n_key)
        o = fused_sdpa(pad_axis(q, n_key, 2, 0.0), pad_axis(k, n_key, 2, 0.0), pad_axis(v, n_key, 2, 0.0),
                       attn_mask=mask, scale=self.head_dim ** -0.5,
                       program_config=_dense_sdpa_config(n_key), compute_kernel_config=ckc)
        ttnn.deallocate(mask)
        PF_SDPA_CALLS[0] += 1
        o = ttnn.experimental.nlp_concat_heads(o)                        # [B, 1, n_key, hp]
        o = ttnn.reshape(ttnn.slice(o, [0, 0, 0, 0], [B, 1, I, hp]), (B, I, hp))
        g = ttnn.linear(a, w["g"], **lin)
        o = ttnn.multiply(o, ttnn.sigmoid(g))
        return ttnn.linear(o, w["a"], **lin)

    def _sdpa_mask_rms(self, z, I, n_key):
        """`_sdpa_mask` straight from the un-normalised z through `rms_pair_bias` (_PF_RMSB); None when z does not
        fit it."""
        if getattr(self, "_b_wt_rms", None) is None:
            w = self.weights["ln_0.weight"].double()[:, None] * self.weights["to_b.weight"].double().t()
            self._b_wt_rms = _DPB.rms_weight([(w * self.head_dim ** 0.5).float()], self.device)
            self._pf_pads = {}
        if int(z.padded_shape[2]) != n_key or not _DPB.rms_eligible(z, self._b_wt_rms, rows=True):
            return None
        pad = self._pf_pads.get((I, n_key))
        if pad is None:
            m = torch.full((1, 1, n_key, n_key), -1e4)
            m[..., :I, :I] = 0.0
            pad = self._pf_pads[(I, n_key)] = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=self.device,
                                                              dtype=self.dtype)
        PF_RMSB_CALLS[0] += 1
        bias, = _DPB.rms_pair_bias(z, self._b_wt_rms, 1e-6, self.n_head,
                                   fidelity=self.compute_kernel_config.math_fidelity)
        mask = ttnn.add(bias, pad)
        ttnn.deallocate(bias)
        return mask

    def _sdpa_mask(self, z, I, n_key):
        """[B, 16, n_key, n_key] bf16 SDPA mask: the pair bias times sqrt(head_dim) (the kernel adds the mask before it
        scales), keys and rows >= I at -1e4."""
        r = self.head_dim ** 0.5
        if _pf_dpb_on():
            m = self._pair_bias_dpb(z, I, n_key, scale=r, raw=True)
            if m is not None:
                return m
        # sqrt(head_dim) folded into the weight in fp32: a bf16 multiply after the linear rounds twice, and its scalar
        # goes in as bf16 (4.906 for 4.899), a 0.15 % scale on every bias.
        w = getattr(self, "_b_w_sdpa", None)
        if w is None:
            w = self._b_w_sdpa = ttnn.from_torch((self.weights["to_b.weight"].float().t() * r).contiguous(),
                                                 layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype)
        bias = _tuned_linear(z, w, ckc=self.compute_kernel_config, dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        pb = ttnn.permute(bias, (0, 3, 1, 2))                            # [B,16,I,I]
        ttnn.deallocate(bias)
        # not deallocated: a pad inside the tile padding hands back a tensor on pb's own buffer
        return pad_axis(pad_axis(pb, n_key, 3, -1e4), n_key, 2, -1e4)

    def _pair_bias_dpb(self, z, I, n_key, scale=1.0, raw=False):
        """[B, 16, I, n_key] fp32 pair bias, keys >= I at -1e4, from one dit_pair_bias program; None when z
        does not fit it. The bias is bf16 out of the program as out of the linear, so the fp32 cast is exact."""
        B = int(z.shape[0])
        if getattr(self, "_b_wt32", None) is None:
            w = torch.zeros(1, 1, 32, int(z.shape[3]))
            w[0, 0, :self.n_head] = self.weights["to_b.weight"].float()
            self._b_wt32 = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype)
            self._pf_masks = {}
        mask = self._pf_masks.get((I, n_key))
        if mask is None:
            m = torch.full((1, 1, n_key, n_key), -1e4)
            m[..., :I, :I] = 0.0
            mask = self._pf_masks[(I, n_key)] = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=self.device,
                                                                dtype=self.dtype)
        if int(z.padded_shape[2]) != n_key or not _DPB.eligible(z, self._b_wt32, mask):
            PF_DPB_CALLS[1] += 1
            return None
        PF_DPB_CALLS[0] += 1
        bias, scratch = _DPB.dit_pair_bias(z, self._b_wt32, mask, scale,
                                           self.compute_kernel_config.math_fidelity, n_out=2)
        ttnn.deallocate(scratch)
        if raw:                                                          # the padded bf16 [B,16,n_key,n_key]
            return bias
        if I != n_key:
            bias = ttnn.reshape(bias, ttnn.Shape([B, self.n_head, I, n_key]),
                                ttnn.Shape([B, self.n_head, n_key, n_key]))
        out = ttnn.typecast(bias, ttnn.float32)
        ttnn.deallocate(bias)
        return out


class PairformerBlock(Module):
    def __init__(self, state_dict, ckc, c_s=384, c_z=128, n_head=16, dtype=None,
                 fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.fp32_residual = fp32_residual
        self.z_transition = Transition(self.scope("z_transition"), ckc, c_z, n=4, dtype=self.dtype)
        self.s_transition = Transition(self.scope("s_transition"), ckc, c_s, n=4, dtype=self.dtype)
        self.attn = PairformerAttention(self.scope("attention_pair_bias"), ckc, c_s, c_z, n_head, dtype=self.dtype)

    def __call__(self, s, z, z_update=None):
        """`z_update(z_transition)`, if given, replaces `z_transition.add_into(z)` (the token encoder's pair memo)."""
        if not self.fp32_residual:
            z = self.z_transition.add_into(z) if z_update is None else z_update(self.z_transition)
            s = ttnn.add(s, self.attn(s, z))
            s = ttnn.add(s, self.s_transition(s))
            return s, z
        # fp32 residual stream (both s and z): matmuls/linears/norms run bf16 (self.dtype),
        # only the residual accumulation is fp32 — no fp32 matmul is issued (Blackhole
        # fp32 matmul is a host-fallback dead-end, per p7 §2g.2). Mirrors RFD3AtomBlock.
        dt = self.dtype
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
            z = ttnn.typecast(z, ttnn.float32, memory_config=z.memory_config())
        zc = ttnn.typecast(z, dt, memory_config=z.memory_config())
        z_upd = self.z_transition(zc)
        z = ttnn.add(z, ttnn.typecast(z_upd, ttnn.float32, memory_config=z_upd.memory_config()))
        sc = ttnn.typecast(s, dt, memory_config=s.memory_config())
        zc = ttnn.typecast(z, dt, memory_config=z.memory_config())
        s_upd = self.attn(sc, zc)
        s = ttnn.add(s, ttnn.typecast(s_upd, ttnn.float32, memory_config=s_upd.memory_config()))
        sc = ttnn.typecast(s, dt, memory_config=s.memory_config())
        s_upd = self.s_transition(sc)
        s = ttnn.add(s, ttnn.typecast(s_upd, ttnn.float32, memory_config=s_upd.memory_config()))
        return s, z


class TokenInitializer(Module):
    """ttnn on-device port of RFD3 TokenInitializer. forward(f) takes the host `f`
    dict (43 keys, as captured) and returns {Q_L_init, C_L, P_LL, S_I, Z_II} on host."""

    C_S, C_Z, C_ATOM, C_ATOMPAIR = 384, 128, 128, 16
    N_PAIRFORMER, N_HEAD = 2, 16
    R_MAX, S_MAX = 32, 2

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        dev = self.device

        # OneD embedder weights (each feature -> linear to its channel). nn.Linear (out,in);
        # torch_to_tt transposes to (in,out) for ttnn.linear.
        def _embedder_weights(prefix):
            return {feat: self.torch_to_tt(f"{prefix}.embedders.{feat}.weight", dtype=self.dtype)
                    for feat in self._feat_keys(prefix)}
        self.w_tok1d = _embedder_weights("token_1d_embedder")
        self.w_atom1d_1 = _embedder_weights("atom_1d_embedder_1")
        self.w_atom1d_2 = _embedder_weights("atom_1d_embedder_2")

        self.downcast_gca = self.scope("downcast_atom.gca")
        # GatedCrossAttention weights (device port; c_query=c_kv=c_s=384, c_model=128, n_head=4, hd=32)
        g = "downcast_atom.gca."
        self.gca_ln_q = self.torch_to_tt(g + "ln_q.weight", dtype=self.dtype)
        self.gca_ln_kv = self.torch_to_tt(g + "ln_kv.weight", dtype=self.dtype)
        self.gca_to_q = self.torch_to_tt(g + "to_q.weight", dtype=self.dtype)
        self.gca_to_k = self.torch_to_tt(g + "to_k.weight", dtype=self.dtype)
        self.gca_to_v = self.torch_to_tt(g + "to_v.weight", dtype=self.dtype)
        self.gca_to_g = self.torch_to_tt(g + "to_g.0.weight", dtype=self.dtype)
        self.gca_k_norm = self.torch_to_tt(g + "k_norm.weight", dtype=self.dtype)
        self.gca_q_norm = self.torch_to_tt(g + "q_norm.weight", dtype=self.dtype)
        self.gca_to_out_w = self.torch_to_tt(g + "to_out.0.weight", dtype=self.dtype)
        self.gca_to_out_b = self.torch_to_tt(g + "to_out.0.bias", dtype=self.dtype)
        self.tr_post_tok = Transition(self.scope("transition_post_token"), ckc, self.C_S, n=2, dtype=self.dtype)
        self.tr_post_atom = Transition(self.scope("transition_post_atom"), ckc, self.C_S, n=2, dtype=self.dtype)
        self.process_s_init_n = self.torch_to_tt("process_s_init.0.weight", dtype=self.dtype)
        self.process_s_init_w = self.torch_to_tt("process_s_init.1.weight", dtype=self.dtype)
        self.to_z_init_i = self.torch_to_tt("to_z_init_i.weight", dtype=self.dtype)
        self.to_z_init_j = self.torch_to_tt("to_z_init_j.weight", dtype=self.dtype)
        self.relpos_lin = self.torch_to_tt("relative_position_encoding.linear.weight", dtype=self.dtype)
        self.relpos2_lin = self.torch_to_tt("relative_position_encoding2.linear.weight", dtype=self.dtype)
        self.proc_token_bonds = self.torch_to_tt("process_token_bonds.weight", dtype=self.dtype)
        self.refpos_tok_invd = self.torch_to_tt("ref_pos_embedder_tok.process_inverse_dist.weight", dtype=self.dtype)
        self.refpos_tok_vm = self.torch_to_tt("ref_pos_embedder_tok.process_valid_mask.weight", dtype=self.dtype)
        self.proc_z_init_n = self.torch_to_tt("process_z_init.0.weight", dtype=self.dtype)
        self.proc_z_init_w = self.torch_to_tt("process_z_init.1.weight", dtype=self.dtype)
        self.tr1_0 = Transition(self.scope("transition_1.0"), ckc, self.C_Z, n=2, dtype=self.dtype)
        self.tr1_1 = Transition(self.scope("transition_1.1"), ckc, self.C_Z, n=2, dtype=self.dtype)
        self.blocks = [PairformerBlock(self.scope(f"transformer_stack.{i}"), ckc,
                                       self.C_S, self.C_Z, self.N_HEAD, dtype=self.dtype)
                        for i in range(self.N_PAIRFORMER)]
        self.proc_s_trunk_n = self.torch_to_tt("process_s_trunk.0.weight", dtype=self.dtype)
        self.proc_s_trunk_w = self.torch_to_tt("process_s_trunk.1.weight", dtype=self.dtype)
        self.proc_single_l_w = self.torch_to_tt("process_single_l.1.weight", dtype=self.dtype)
        self.proc_single_m_w = self.torch_to_tt("process_single_m.1.weight", dtype=self.dtype)
        self.proc_z_n = self.torch_to_tt("process_z.0.weight", dtype=self.dtype)
        self.proc_z_w = self.torch_to_tt("process_z.1.weight", dtype=self.dtype)
        self.motif_pos_proj = self.torch_to_tt("motif_pos_embedder.output_proj.weight", dtype=self.dtype)
        self.motif_pos_vm = self.torch_to_tt("motif_pos_embedder.process_valid_mask.weight", dtype=self.dtype)
        self.refpos_invd = self.torch_to_tt("ref_pos_embedder.process_inverse_dist.weight", dtype=self.dtype)
        self.refpos_vm = self.torch_to_tt("ref_pos_embedder.process_valid_mask.weight", dtype=self.dtype)
        self.pair_mlp_w = [self.torch_to_tt(f"pair_mlp.{i}.weight", dtype=self.dtype) for i in (1, 3, 5)]
        self.proc_pll_w = self.torch_to_tt("process_pll.weight", dtype=self.dtype)
        self.project_pll_w = self.torch_to_tt("project_pll.weight", dtype=self.dtype)

    @staticmethod
    def _feat_keys(prefix):
        if prefix == "token_1d_embedder":
            return ["ref_motif_token_type", "restype", "ref_plddt", "is_non_loopy"]
        return ["ref_atom_name_chars", "ref_element", "ref_charge", "ref_mask",
                "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
                "has_zero_occupancy", "ref_pos", "ref_atomwise_rasa", "active_donor",
                "active_acceptor", "is_atom_level_hotspot"]

    def _embed1d(self, f, weights, collapse_len, keys):
        """Sum of per-feature device linears on collapsed features -> [collapse_len, C]."""
        acc = None
        for feat in keys:
            x = _collapse(f[feat].float(), collapse_len)
            xt = _tt(x, self.device, self.dtype)
            y = ttnn.linear(xt, weights[feat], compute_kernel_config=self.compute_kernel_config,
                              dtype=self.dtype, core_grid=CORE_GRID_MAIN)
            acc = y if acc is None else ttnn.add(acc, y)
        return acc

    # --- host-side GatedCrossAttention reference (kept for parity isolation) ---
    def _host_gca(self, s_h, ql_h, vm):
        """Mirror Downcast(cross_attention) + GatedCrossAttention(kq_norm=True) on host.
        s_h [I, C_S], ql_h [L, C_S], vm [I, A]. Returns the per-token update [I, C_S]."""
        W = self.downcast_gca
        c_s, c_model, n_head = self.C_S, 128, 4
        I, A = s_h.shape[0], vm.shape[1]
        hd = c_model // n_head
        # ungroup atoms: Q_L [L,384] -> Q_IA [1, I, A, 384]
        flat_idx = vm.flatten().nonzero(as_tuple=False).squeeze(1)
        idx = flat_idx.view(1, -1, 1).expand(1, -1, c_s)
        Q_IA = torch.zeros(1, I * A, c_s, dtype=ql_h.dtype)
        Q_IA = Q_IA.scatter(1, idx, ql_h.unsqueeze(0)).reshape(1, I, A, c_s)
        q = s_h.unsqueeze(0).unsqueeze(2)          # [1, I, 1, C_S]
        kv = Q_IA                                   # [1, I, A, C_S]
        attn_mask = vm.unsqueeze(1)                 # [I, 1, A]
        q = F.rms_norm(q, (c_s,), W["ln_q.weight"], 1e-6)
        kv = F.rms_norm(kv, (c_s,), W["ln_kv.weight"], 1e-6)
        qq = F.linear(q, W["to_q.weight"]); kk = F.linear(kv, W["to_k.weight"]); vv = F.linear(kv, W["to_v.weight"])
        gg = torch.sigmoid(F.linear(q, W["to_g.0.weight"]))
        kk = F.rms_norm(kk, (c_model,), W["k_norm.weight"], 1e-6)
        qq = F.rms_norm(qq, (c_model,), W["q_norm.weight"], 1e-6)

        def heads(t):
            b, t_, n, _ = t.shape
            return t.reshape(b, t_, n, n_head, hd).permute(0, 3, 1, 2, 4)  # [b,h,t,n,c]

        qh, kh, vh, gh = heads(qq), heads(kk), heads(vv), heads(gg)
        scale = 1.0 / math.sqrt(hd)
        attn = torch.einsum("bhtqc,bhtkc->bhtqk", qh, kh) * scale   # [1,4,I,1,A]
        attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))
        invalid = ~torch.any(attn_mask, dim=-1)                    # [I]
        if invalid.any():
            attn[:, :, invalid, :, :] = 0.0
        attn = F.softmax(attn, dim=-1)
        o = torch.einsum("bhtqk,bhtkd->bhtqd", attn, vh) * gh       # [1,4,I,1,hd]
        o = o.permute(0, 2, 3, 1, 4).reshape(1, I, 1, c_model)       # [1,I,1,128]
        o = F.linear(o, W["to_out.0.weight"], W["to_out.0.bias"])  # [1,I,1,C_S]
        return o.squeeze(0).squeeze(1)                            # [I, C_S]

    def _device_gca(self, s_h, ql_h, vm):
        """On-device GatedCrossAttention (Downcast). s_h [I, C_S], ql_h [L, C_S],
        vm [I, A]. Returns the per-token update [I, C_S] on host. head_dim=32 (tile-aligned)
        so manual matmul-softmax attention is clean (same recipe as PairformerAttention)."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        c_s, c_model, n_head = self.C_S, 128, 4
        hd = c_model // n_head  # 32
        I, A = s_h.shape[0], vm.shape[1]
        # ungroup atoms on host: Q_L [L,384] -> Q_IA [1, I, A, 384]
        flat_idx = vm.flatten().nonzero(as_tuple=False).squeeze(1)
        idx = flat_idx.view(1, -1, 1).expand(1, -1, c_s)
        Q_IA = torch.zeros(1, I * A, c_s, dtype=ql_h.dtype).scatter(1, idx, ql_h.unsqueeze(0))
        Q_IA = Q_IA.reshape(1, I, A, c_s)
        q = _tt(s_h.unsqueeze(0), dev, dt)
        kv = _tt(Q_IA, dev, dt)
        q = ttnn.rms_norm(q, weight=self.gca_ln_q, epsilon=1e-6, compute_kernel_config=ckc)
        kv = ttnn.rms_norm(kv, weight=self.gca_ln_kv, epsilon=1e-6, compute_kernel_config=ckc)
        qq = ttnn.linear(q, self.gca_to_q, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(kv, self.gca_to_k, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(kv, self.gca_to_v, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(q, self.gca_to_g, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.gca_q_norm, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.gca_k_norm, epsilon=1e-6, compute_kernel_config=ckc)
        # Batch over tokens. Keep token before head until after flattening the token
        # batch; moving head before token here scrambles both axes.
        qq = ttnn.permute(ttnn.reshape(qq, (1, I, 1, n_head, hd)), (0, 1, 3, 2, 4))
        qq = ttnn.reshape(qq, (I, n_head, 1, hd))                                    # [I,4,1,32]
        gg = ttnn.permute(ttnn.reshape(gg, (1, I, 1, n_head, hd)), (0, 1, 3, 2, 4))
        gg = ttnn.reshape(gg, (I, n_head, 1, hd))
        kk = ttnn.permute(ttnn.reshape(kk, (1, I, A, n_head, hd)), (0, 1, 3, 2, 4))
        vv = ttnn.permute(ttnn.reshape(vv, (1, I, A, n_head, hd)), (0, 1, 3, 2, 4))
        kk = ttnn.reshape(kk, (I, n_head, A, hd))                                    # [I,4,A,32]
        vv = ttnn.reshape(vv, (I, n_head, A, hd))
        kt = ttnn.permute(kk, (0, 1, 3, 2))                                        # [I,4,32,A]
        sc = ttnn.matmul(qq, kt, compute_kernel_config=ckc,
                         core_grid=_grid_if_single_k_tile(qq))                       # [I,4,1,A]
        ttnn.deallocate(qq); ttnn.deallocate(kt)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
        sc = ttnn.multiply(sc, hd ** -0.5)
        mask = torch.where(vm, 0.0, -1e4).to(torch.float32).unsqueeze(1).unsqueeze(1)  # [I,1,1,A]
        mask = _tt(mask, dev, ttnn.float32)
        sc = ttnn.add(sc, mask)
        ttnn.deallocate(mask)
        attn = softmax_generic.softmax_bf16(sc, dt)
        o = ttnn.matmul(attn, vv, compute_kernel_config=ckc, dtype=dt,
                        core_grid=_grid_if_single_k_tile(attn))                      # [I,4,1,32]
        ttnn.deallocate(attn); ttnn.deallocate(vv)
        o = ttnn.multiply(o, ttnn.sigmoid(gg))
        ttnn.deallocate(gg)
        o = ttnn.reshape(o, (1, I, c_model))
        o = ttnn.linear(o, self.gca_to_out_w, bias=self.gca_to_out_b,
                          compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        return ttnn.to_torch(o).float().squeeze(0)                                  # [I, C_S]

    def __call__(self, f):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        tok_idx = f["atom_to_token_map"].long()
        L = len(tok_idx)
        f = dict(f)  # shallow copy (we mutate ref_atom_name_chars)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        # ===== init_tokens =====
        # token_1d embedder (device linears, summed)
        s = self._embed1d(f, self.w_tok1d, I, list(self.w_tok1d.keys()))
        s = ttnn.add(s, self.tr_post_tok(s))
        # atom_1d embedder_1 (device) -> Q_L [L, C_S]
        ql = self._embed1d(f, self.w_atom1d_1, L, list(self.w_atom1d_1.keys()))
        # downcast_atom (host GCA this pass): S_I += gca(S_I, Q_L, tok_idx)
        s_h = ttnn.to_torch(s).float().squeeze(0)            # [I, C_S]
        ql_h = ttnn.to_torch(ql).float().squeeze(0)         # [L, C_S]
        vm = _build_valid_mask(tok_idx)                     # [I, A]
        s_h = s_h + self._device_gca(s_h, ql_h, vm)          # [I, C_S]
        s = _tt(s_h.unsqueeze(0), dev, dt)                 # back to device [1,I,C_S]
        s = ttnn.add(s, self.tr_post_atom(s))
        # process_s_init: RMSNorm + linear
        s = ttnn.rms_norm(s, weight=self.process_s_init_n, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.linear(s, self.process_s_init_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        s_h = ttnn.to_torch(s).float().squeeze(0)           # [I, C_S] host (for outer-sum + later gathers)
        # Z_init = outer(to_z_init_i(S), to_z_init_j(S)) [1,I,I,C_Z]
        zi = ttnn.linear(s, self.to_z_init_i, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        zj = ttnn.linear(s, self.to_z_init_j, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        zi = ttnn.reshape(zi, (1, I, 1, self.C_Z))
        zj = ttnn.reshape(zj, (1, 1, I, self.C_Z))
        z = ttnn.add(zi, zj)                              # [1,I,I,128]
        # + relative_position_encoding (host one-hot -> device linear)
        rph = _tt(_build_relpos_onehot(f, self.R_MAX, self.S_MAX).unsqueeze(0), dev, dt)
        z = ttnn.add(z, ttnn.linear(rph, self.relpos_lin, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN))
        # + process_token_bonds
        tb = _tt(f["token_bonds"].unsqueeze(-1).float().unsqueeze(0), dev, dt)  # [1,I,I,1]
        z = ttnn.add(z, ttnn.linear(tb, self.proc_token_bonds, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN))
        # + ref_pos_embedder_tok (no-frame; token-level, I×I)
        is_ca = f["is_ca"]
        rpos_ca = f["ref_pos"][is_ca].float()              # [I, 3]
        tid = f["ref_space_uid"][is_ca].long()            # [I]
        vm_tok = (tid.unsqueeze(-1) == tid.unsqueeze(-2)).unsqueeze(-1).float()  # [I,I,1]
        invd = 1.0 / (1.0 + (rpos_ca.unsqueeze(-2) - rpos_ca.unsqueeze(-3)).pow(2).sum(-1, keepdim=True))
        invd = _tt(invd.unsqueeze(0), dev, dt); vm_tok = _tt(vm_tok.unsqueeze(0), dev, dt)
        rp = ttnn.multiply(ttnn.linear(invd, self.refpos_tok_invd, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_tok)
        rp = ttnn.add(rp, ttnn.multiply(ttnn.linear(vm_tok, self.refpos_tok_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_tok))
        z = ttnn.add(z, rp)
        # 2 Pairformer blocks
        for blk in self.blocks:
            s_dev, z = blk(_tt(s_h.unsqueeze(0), dev, dt), z)
            s_h = ttnn.to_torch(s_dev).float().squeeze(0)
        # cat([Z, relpos2]) -> process_z_init (RMSNorm(2*C_Z) + linear) -> 2x transition_1
        rph2 = _tt(_build_relpos_onehot(f, self.R_MAX, self.S_MAX).unsqueeze(0), dev, dt)
        z2 = ttnn.linear(rph2, self.relpos2_lin, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        z = ttnn.concat([z, z2], dim=-1)               # [1,I,I,256]
        z = ttnn.rms_norm(z, weight=self.proc_z_init_n, epsilon=1e-6, compute_kernel_config=ckc)
        z = ttnn.linear(z, self.proc_z_init_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        z = ttnn.add(z, self.tr1_0(z))
        z = ttnn.add(z, self.tr1_1(z))
        S_init_I = s_h                                          # [I, C_S] host
        Z_init_II = ttnn.to_torch(z).float().squeeze(0)        # [I, I, C_Z] host
        return self._init_atoms(f, S_init_I, Z_init_II, tok_idx, L, I)

    def _init_atoms(self, f, S_init_I, Z_init_II, tok_idx, L, I):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        # Q_L_init = atom_1d_embedder_2 (device linears) [L, C_ATOM]
        ql_init = self._embed1d(f, self.w_atom1d_2, L, list(self.w_atom1d_2.keys()))
        # process_s_trunk(S_init_I): RMSNorm + linear -> [I, C_ATOM]; gather to atoms via tok_idx
        s_tr = _tt(S_init_I.unsqueeze(0), dev, dt)
        s_tr = ttnn.rms_norm(s_tr, weight=self.proc_s_trunk_n, epsilon=1e-6, compute_kernel_config=ckc)
        s_tr = ttnn.linear(s_tr, self.proc_s_trunk_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        s_tr_h = ttnn.to_torch(s_tr).float().squeeze(0)        # [I, C_ATOM]
        c_l_h = s_tr_h[tok_idx]                                # [L, C_ATOM] (gather)
        c_l = ttnn.add(ql_init, _tt(c_l_h.unsqueeze(0), dev, dt))  # C_L [1,L,C_ATOM]

        # process_single_l/m (ReLU + linear on C_L) -- c_l_h uploaded once,
        # reused for both sl/sm (was 2x, p23 perf; identical bf16 cast either way).
        # Both are [1,L,16] and the pair loop below needs a row slice of one and all of
        # the other, so they are built once, before it.
        c_l_h = ttnn.to_torch(c_l).float().squeeze(0)        # [L, C_ATOM]
        c_l_dev = _tt(c_l_h.unsqueeze(0), dev, dt)
        sl = ttnn.relu(c_l_dev)
        sl = ttnn.linear(sl, self.proc_single_l_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,L,16]
        sm = ttnn.relu(c_l_dev)
        sm = ttnn.linear(sm, self.proc_single_m_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        sm_cols = ttnn.unsqueeze(sm, -3)                     # [1,1,L,16], broadcast over rows
        # process_z(Z_init_II): RMSNorm + linear -> [I,I,16]; gathered to atom rows in the loop.
        # (Z_init_II_dev kept around unmodified -- ttnn ops return new tensors, not in-place --
        # and reused below for the zupd add instead of a 2nd upload of the same host tensor;
        # p23 perf, bit-identical.)
        Z_init_II_dev = _tt(Z_init_II.unsqueeze(0), dev, dt)
        z_dev = ttnn.rms_norm(Z_init_II_dev, weight=self.proc_z_n, epsilon=1e-6, compute_kernel_config=ckc)
        pz = ttnn.linear(z_dev, self.proc_z_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,I,I,16]
        pz_h = ttnn.to_torch(pz).float().squeeze(0)          # [I,I,16]
        ttnn.deallocate(pz); ttnn.deallocate(z_dev)

        # ---- P_LL [L, L, C_ATOMPAIR=16], one row block of atoms at a time ----
        # Every tensor here is [L, L, <=32 after tile padding] and L is ATOMS, ~11 per
        # residue: one of them is 1.33 GB at the 4558 atoms RFD3 used to stop at, and the
        # unblocked form held eight of them to the end of the function. The atom row axis
        # is independent in all of it -- elementwise products, broadcasts, and linears that
        # contract the channel axis -- so a row block computes exactly the rows it covers
        # and `rows == L` is the unblocked path, byte for byte.
        mp = f["motif_pos"].float()
        rp = f["ref_pos"].float()
        coord_ok = f["is_motif_atom_with_fixed_coord"]
        seq_ok = f["is_motif_atom_with_fixed_seq"]
        uid = f["ref_space_uid"]
        rows = row_block(_ATOM_PAIR_LIVE_ROWS * align_tile(L) * TILE * 2,
                         atom_pair_budget_bytes())
        ATOM_PAIR_BLOCK_STATS[:] = [rows, -(-L // rows)]
        p_blocks, pll_blocks = [], []
        for a0 in range(0, L, rows):
            a1 = min(a0 + rows, L)
            # motif_pos_embedder (SinusoidalDistEmbed): host sincos -> device output_proj + valid_mask linears
            vm_mp = (coord_ok[a0:a1].unsqueeze(-1) & coord_ok.unsqueeze(-2)).unsqueeze(-1).float()
            sc, vsc = _sinusoidal_embed(mp[a0:a1], mp, vm_mp)     # [R,L,64], [R,L,1]
            sc = _tt(sc.unsqueeze(0), dev, dt); vsc = _tt(vsc.unsqueeze(0), dev, dt)
            proj = ttnn.linear(sc, self.motif_pos_proj, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
            ttnn.deallocate(sc)
            p = ttnn.multiply(proj, vsc)
            ttnn.deallocate(proj)
            p = _pair_add(p, ttnn.multiply(
                ttnn.linear(vsc, self.motif_pos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vsc))
            ttnn.deallocate(vsc)
            # ref_pos_embedder (no-frame): host inv_dist -> device linears
            same_tok = (uid[a0:a1].unsqueeze(-1) == uid.unsqueeze(-2)).unsqueeze(-1).float()
            has_seq = (seq_ok[a0:a1].unsqueeze(-1) & seq_ok.unsqueeze(-2)).unsqueeze(-1).float()
            vm_rp = same_tok * has_seq
            D = rp[a0:a1].unsqueeze(-2) - rp.unsqueeze(-3)
            invd = 1.0 / (1.0 + D.pow(2).sum(-1, keepdim=True).clamp(min=1e-6))
            invd = _tt(invd.unsqueeze(0), dev, dt); vm_rp = _tt(vm_rp.unsqueeze(0), dev, dt)
            p = _pair_add(p, ttnn.multiply(
                ttnn.linear(invd, self.refpos_invd, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
            ttnn.deallocate(invd)
            p = _pair_add(p, ttnn.multiply(
                ttnn.linear(vm_rp, self.refpos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
            ttnn.deallocate(vm_rp)
            # A single block slices nothing: `sl` itself is the block, so the unblocked
            # path stays the op sequence it always was.
            sl_rows = sl if rows >= L else ttnn.slice(sl, (0, a0, 0), (1, a1, self.C_ATOMPAIR))
            p = _pair_add(p, ttnn.unsqueeze(sl_rows, -2), free_b=False)  # [1,R,1,16] over cols
            p = _pair_add(p, sm_cols, free_b=False)                      # [1,1,L,16] over rows
            # process_z gathered to this block's atom rows, both axes
            p = _pair_add(p, _tt(pz_h[tok_idx[a0:a1]][:, tok_idx, :].unsqueeze(0), dev, dt))
            # pair_mlp (ReLU + linear x3) residual
            m = p
            for w in self.pair_mlp_w:
                r = ttnn.relu(m)
                if m is not p:
                    ttnn.deallocate(m)
                m = ttnn.linear(r, w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
                ttnn.deallocate(r)
            p = _pair_add(p, m)
            p_blocks.append(ttnn.to_torch(p).float().squeeze(0))            # [R,L,16]
            pll = ttnn.linear(p, self.proc_pll_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
            ttnn.deallocate(p)
            pll_blocks.append(ttnn.to_torch(pll).float().squeeze(0))        # [R,L,16]
            ttnn.deallocate(pll)
        p_h = p_blocks[0] if len(p_blocks) == 1 else torch.cat(p_blocks, dim=0)
        pll_h = pll_blocks[0] if len(pll_blocks) == 1 else torch.cat(pll_blocks, dim=0)
        del p_blocks, pll_blocks
        # pooled = scatter_mean_pool(process_pll(P_LL)) -> project_pll -> add to Z.
        # Not blocked: the pool reduces over the atom row axis, and summing per block and
        # adding the parts is a different order of additions from summing once.
        pooled = _scatter_mean_pool(pll_h, tok_idx, I)        # [I,I,16]
        pooled = _tt(pooled.unsqueeze(0), dev, dt)
        zupd = ttnn.linear(pooled, self.project_pll_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,I,I,128]
        z_dev = ttnn.add(Z_init_II_dev, zupd)
        Z_II = ttnn.to_torch(z_dev).float().squeeze(0)       # [I,I,128]
        Q_L_init = ttnn.to_torch(ql_init).float().squeeze(0)  # [L,128]
        C_L = ttnn.to_torch(c_l).float().squeeze(0)          # [L,128]
        P_LL = p_h                                          # [L,L,16]
        return {"Q_L_init": Q_L_init, "C_L": C_L, "P_LL": P_LL, "S_I": S_init_I, "Z_II": Z_II}


def _merge_heads(x, shape):
    """[..., n_head, rows, head_dim] -> `shape` (the heads folded back into channels).

    The obvious `permute(0,2,1,3) + reshape` retiles twice through a 4D intermediate whose
    head axis is tile-padded (16 heads occupy a 32-row tile), and measures 4-9 GB/s.
    `nlp_concat_heads` does the same movement in one kernel at 55-63 GB/s -- but it assumes
    head_dim is a tile multiple and silently reads 64-wide heads out of a 48-wide tensor when
    it is not (p31: bit-exact at head_dim=32, maxabs 6.2 at 48 and 24). So it is used only
    where head_dim is aligned, and the two-op form stays for the DiT (48) and the pairformer
    (24). Both branches are pure data movement, and both are bit-exact.
    """
    if x.shape[-1] % TILE == 0:
        return ttnn.reshape(ttnn.experimental.nlp_concat_heads(x), shape)
    return ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), shape)


def _dense_attention_mask(indices):
    """Convert [B,L,K] neighbour indices to the equivalent dense additive mask."""
    indices = indices.long()
    batch, length, _ = indices.shape
    keep = torch.zeros(batch, length, length, dtype=torch.bool)
    keep.scatter_(2, indices.cpu(), True)
    return torch.where(keep, 0.0, -1e4).unsqueeze(1)


def _sparse_qk_host(p_host, indices, n_heads=4):
    indices = indices.long().cpu()
    batch, length, n_keys = indices.shape
    p_host = p_host.unsqueeze(0) if p_host.ndim == 3 else p_host
    if p_host.shape[0] == 1 and batch != 1:
        p_host = p_host.expand(batch, -1, -1, -1)
    batch_idx = torch.arange(batch)[:, None, None]
    row_idx = torch.arange(length)[None, :, None]
    p_sparse = p_host[batch_idx, row_idx, indices]
    attn_idx = indices.unsqueeze(1).expand(batch, n_heads, length, n_keys)
    return p_sparse, attn_idx.to(torch.int32), n_keys


def _mask_template(cache, device, dtype, batch, n_heads, length):
    """The -1e4 dense attention-mask template the pair bias is scattered into.

    It is a pure constant of (batch, n_heads, length). Re-creating it every step was
    9% of a diffusion step, so one template is kept alive per shape; that is
    bit-exact because `ttnn.scatter` is out-of-place (verify_scatter_aliasing.py
    replays a captured scatter against a single persistent template with two
    different index sets and gets results identical to a fresh template each time).

    The key axis is padded out to a tile multiple (`pad_axis`): the scatter that
    writes the pair bias into this template does not write the output's tile padding, so
    a non-tile-multiple key axis would leave the softmax reducing over undefined DRAM.
    Making the axis a tile multiple costs no device memory -- the buffer was tile-padded
    to the same size either way -- and the extra keys stay at -1e4, whose weight after
    exp is exactly 0.

    One slot on purpose -- a run folds one design shape at a time, and holding the
    90 MB template is not extra peak memory (the old code allocated it anyway).
    """
    key = (batch, n_heads, length, dtype)
    entry = cache.get("mask")
    if entry is None or entry[0] != key:
        entry = (key, ttnn.full(
            (batch, n_heads, length, align_tile(length)), -1e4,
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
        cache["mask"] = entry
    return entry[1]


def _zero_template(cache, device, dtype, batch, n_heads, length):
    """The zero dense template the gathered attention weights are scattered into.

    The mask template's counterpart for the other side of the softmax. `_mask_template` holds
    -1e4 because it is scattered into BEFORE the exp; this one holds 0.0 because it is scattered
    into after, and the dense arm's non-neighbour weights are post-softmax exact zeros. Same
    single-slot, same out-of-place-scatter argument (verify_scatter_aliasing.py).
    """
    key = (batch, n_heads, length, dtype)
    entry = cache.get("zeros") if cache is not None else None
    if entry is None or entry[0] != key:
        entry = (key, ttnn.zeros(
            (batch, n_heads, length, align_tile(length)),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
        if cache is not None:
            cache["zeros"] = entry
    return entry[1]


_PAIR_TABLE_SLOTS = 2


def _pair_gather_table(cache, p_host, device, dtype):
    """Resident (L*L, C) gather table for the step-invariant pair features.

    P_LL is built once by TokenInitializer._init_atoms and is
    the same data for all 200 diffusion steps and every recycle of each, so the table
    the per-step neighbour gather reads can live on the card: the gather becomes one
    ttnn.embedding instead of a host advanced-index plus an upload of the result
    (13.8 MB per step at 3359 atoms, 110 MB at batch 8).

    Keyed on the storage address, not id() -- the diffusion module normalizes P_LL to
    4-D on entry, so every step hands this a fresh view of the same data. The entry
    holds its own reference to that view, so no other tensor can take the address
    while the table is alive. Two slots because a classifier-free-guidance run
    alternates between the conditional and unconditional P_LL every step and one slot
    would rebuild both; a third distinct table (the next design shape) evicts the
    oldest.
    """
    key = (p_host.data_ptr(), tuple(p_host.shape), dtype)
    tables = cache.setdefault("tables", {})
    entry = tables.get(key)
    if entry is None:
        while len(tables) >= _PAIR_TABLE_SLOTS:
            ttnn.deallocate(tables.pop(next(iter(tables)))[1])
        entry = (p_host, _upload_pair_table(p_host.reshape(-1, p_host.shape[-1]), device))
        tables[key] = entry
    return entry[1]


_TABLE_ROWS_PER_PAGE = 16
_TABLE_CHUNK_ROWS = 1 << 20


def _upload_pair_table(flat, device):
    """Upload a bf16 [N, C] row-major gather table at wide-page speed without a second copy on the card.

    A row-major [N, 16] tensor moves as one 32-byte page per row (~520 MB/s on WH). Each 1M-row chunk goes up as
    [rows/16, 16*C] instead (512-byte pages), is reshaped on the card and written into the preallocated table, so
    the peak is the table plus one 32 MB chunk. The torch-side bf16 cast also beats ttnn's own fp32 conversion.
    Page fixture (36.6M x 16, WH, 2 threads): 5.35 s (fp32 narrow) -> 3.1 (bf16 narrow) -> 0.9 s, gathers equal.
    Rows past N are padding the gather never indexes.
    """
    n, c = flat.shape
    chunk = min(_TABLE_CHUNK_ROWS, -(-n // _TABLE_ROWS_PER_PAGE) * _TABLE_ROWS_PER_PAGE)
    n_pad = -(-n // chunk) * chunk
    table = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n_pad, c]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                           device)
    for r0 in range(0, n_pad, chunk):
        part = flat[r0:r0 + chunk].to(torch.bfloat16)
        if part.shape[0] < chunk:
            part = torch.cat([part, part.new_zeros(chunk - part.shape[0], c)])
        wide = ttnn.from_torch(part.reshape(chunk // _TABLE_ROWS_PER_PAGE, _TABLE_ROWS_PER_PAGE * c),
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        rows = ttnn.reshape(wide, (1, 1, chunk, c))
        ttnn.experimental.slice_write(rows, table, [0, 0, r0, 0], [1, 1, r0 + chunk, c], [1, 1, 1, 1])
        ttnn.deallocate(rows)
        ttnn.deallocate(wide)
    return ttnn.reshape(table, (n_pad, c))


def _sparse_pair_gather(cache, p_host, indices, device, dtype):
    """[B,L,K,C] neighbour pair features, gathered on device off the resident table.

    Only the flat row index crosses the host boundary (a quarter of the bytes the
    gathered features would be, and it is uint32 rather than the wider pair channel).
    """
    table = _pair_gather_table(cache, p_host, device, dtype)
    batch, length, n_keys = indices.shape
    row_offset = (torch.arange(length, dtype=torch.int64) * length).reshape(1, length, 1)
    flat = (indices.long().cpu() + row_offset).reshape(1, batch * length * n_keys)
    idx = ttnn.from_torch(flat.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=device, dtype=ttnn.uint32)
    rows = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(idx)
    out = ttnn.to_layout(
        ttnn.reshape(rows, (batch, length, n_keys, p_host.shape[-1])), ttnn.TILE_LAYOUT)
    ttnn.deallocate(rows)
    return out


def _check_gather_bound(length):
    """Refuse the gathered atom softmax where ttnn.gather is known to return wrong data.

    A default-off flag that silently computes garbage is worse than no flag: the arm was built
    with five passing invariant tests, every one of which pinned a property of the INDEX rather
    than the output of the op, and the first fold run on it would have produced a plausible
    number. Fail loudly instead.
    """
    n_key_axis = align_tile(length)
    if n_key_axis > _TTNN_GATHER_MAX_KEY_AXIS:
        raise RuntimeError(
            "RFD3_GATHERED_SOFTMAX is unusable at this shape: the key axis is %d and ttnn.gather "
            "returns wrong data above %d. The gathered atom softmax needs a fused kernel, not "
            "ttnn.gather." % (n_key_axis, _TTNN_GATHER_MAX_KEY_AXIS))


def _sparse_attn_index(indices, device, n_heads):
    """[B,H,L,K] uint32 scatter index, replicated over heads on device.

    Every head scatters to the same columns, so only a quarter of these bytes need to
    cross the host boundary.
    """
    up = _tt(indices.cpu().unsqueeze(1).to(torch.int32).contiguous(), device, ttnn.uint32)
    if n_heads == 1:
        return up
    out = ttnn.concat([up] * n_heads, dim=1)
    ttnn.deallocate(up)
    return out


def _sparse_attn_index_rm(indices, device):
    """[1,1,L,K] uint32 ROW_MAJOR neighbour index, for the fused bias kernel.

    The kernel reads the same index for every head, so unlike the scatter path nothing is
    replicated over heads and nothing is tilized. One page per row of 4*K bytes is also what
    makes the kernel's per-band index fetch a contiguous page read.
    """
    return ttnn.from_torch(
        indices.cpu().unsqueeze(1).to(torch.int32).contiguous(),
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32)


def _css_plan(indices, attn_idx_dev, device):
    """Chunk lists of one step's [1, L, K] neighbour index for the css atom SDPA, kept until the
    next step's index replaces it. Chunk 64 only: at 32 the generic SDPA is off by ~1.0 against
    float64 dense or not (p109), so an axis 64 does not divide keeps the shipped SDPA."""
    L = int(indices.shape[-2])
    n = align_tile(L)
    c = 64
    _CSS_STEP.clear()
    if n % c:
        return
    nq = n // c
    occ = torch.zeros(nq, nq, dtype=torch.bool)
    ki = indices[0].long() // c
    occ[(torch.arange(L) // c).unsqueeze(1).expand_as(ki), ki] = True
    rows, counts = _SG.css_lists(occ)
    lists = ttnn.from_torch(rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _CSS_STEP["step"] = (attn_idx_dev, (lists, counts), c)


def _css_for(attn_idx_dev):
    hit = _CSS_STEP.get("step")
    return hit if hit is not None and hit[0] is attn_idx_dev else None


def _sparse_qk_inputs(p_host, indices, device, dtype, n_heads=4, mask_cache=None):
    """Gather local pair features and build the scatter index for one step.

    A diffusion step calls this three times -- once from the encoder and once per
    recycle from the decoder -- and all three get the SAME P_LL and the SAME
    f["attn_indices"], so all three produce identical tensors. Callers that share one
    cache dict (RFD3DiffusionModule hands the encoder and the decoder the same one)
    get one build per step.

    That one build is a device gather off a resident P_LL table when the cache makes
    the table worth holding and the pair features are bf16 (ttnn.embedding takes a
    bf16 table only, the same constraint protenix.py:_window_kv documents). Otherwise
    -- an fp32 pair stream, a cacheless isolated test, or a p_host already expanded
    over the batch -- it falls back to the host gather, which is bit-identical: a
    gather is a copy, so bf16(P_LL)[idx] == bf16(P_LL[idx]).

    Keyed on the identity of p_host and indices, with the entry holding its own
    references to both so their ids cannot be recycled underneath it while it
    lives -- the same idiom _attention_index_prefix uses. The cached device
    tensors are read-only downstream: the pair features feed a linear, the index
    feeds ttnn.scatter (out-of-place, see _mask_template).
    """
    key = (id(p_host), id(indices), id(device), dtype, n_heads)
    if mask_cache is not None:
        hit = mask_cache.get("step")
        if hit is not None and hit[0] == key and hit[1] is p_host and hit[2] is indices:
            return hit[3]
        # Drop the stale entry before building the replacement so the old 20.6 MB
        # of device tensors is freed first rather than held alongside the new ones.
        mask_cache.pop("step", None)
    length = indices.shape[-2]
    batch = indices.shape[0]
    on_device = (
        mask_cache is not None
        and dtype == ttnn.bfloat16
        and p_host.ndim == 4
        and p_host.shape[:3] == (1, length, length)
    )
    # The fused kernel writes the mask constant itself, so on that path there is no dense
    # template to build or hold (90 MB at 3359 atoms) and no per-head index replica: it takes
    # one ROW_MAJOR [1,1,L,K] index and returns the fp32 bias directly. `dense_bias is None`
    # is what tells _sparse_bias_f32 which route to take, so the choice is made once per step
    # here rather than per block.
    fused_bias = rfd3_bias.eligible_shape(batch, n_heads, length, indices.shape[-1], dtype)
    if on_device:
        n_keys = indices.shape[-1]
        p_dev = _sparse_pair_gather(mask_cache, p_host, indices, device, dtype)
        attn_idx_dev = (_sparse_attn_index_rm(indices, device) if fused_bias
                        else _sparse_attn_index(indices, device, n_heads))
    else:
        p_sparse, attn_idx, n_keys = _sparse_qk_host(p_host, indices, n_heads)
        p_dev = _tt(p_sparse, device, dtype)
        attn_idx_dev = (_sparse_attn_index_rm(indices, device) if fused_bias
                        else _tt(attn_idx, device, ttnn.uint32))
    if fused_bias:
        dense_bias = None
    elif mask_cache is None:
        dense_bias = ttnn.full(
            (batch, n_heads, length, align_tile(length)), -1e4,
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        dense_bias = _mask_template(
            mask_cache, device, dtype, batch, n_heads, length
        )
    if _GATHERED_SOFTMAX:
        _check_gather_bound(length)
        # The gathered softmax needs the [B,H,L,K] TILE index that ttnn.gather/scatter take, which
        # is what the non-fused route already builds. On the fused-bias route attn_idx_dev is the
        # ROW_MAJOR [1,1,L,K] variant its kernel wants, so build the tiled replica as well rather
        # than turning the fused bias off -- the arm has to differ from the shipped default in the
        # softmax and nothing else.
        gather_idx = (attn_idx_dev if not fused_bias
                      else _sparse_attn_index(indices, device, n_heads))
        gathered = (gather_idx,
                    _zero_template(mask_cache, device, dtype, batch, n_heads, length))
    else:
        gathered = None
    # Block-sparse plan for this step, or None to run the shipped dense chain. Built here rather
    # than per call because a diffusion step's three call sites share one index and one cache
    # entry, so the host work happens once a step -- which is what makes its 2.0 ms affordable.
    # Requires the fused-bias route: the blocked chain reuses that kernel with a block-local
    # index, and the non-fused route's dense -1e4 template is the thing this arm exists to avoid.
    block = None
    if _BS.enabled() and fused_bias and not _GATHERED_SOFTMAX:
        bplan = _BS.plan(indices, align_tile(length))
        if bplan is not None:
            nb, q_block, u_width, gather, pos = bplan
            block = (nb, q_block, u_width,
                     _BS.gather_index(gather, n_heads, align_tile(length), device),
                     _sparse_attn_index_rm(pos.unsqueeze(0), device))
    if _CSS and _ATOM_SDPA and fused_bias and batch == 1 and block is None and gathered is None:
        _css_plan(indices, attn_idx_dev, device)
    out = (p_dev, n_keys, attn_idx_dev, dense_bias, gathered, block)
    if mask_cache is not None:
        mask_cache["step"] = (key, p_host, indices, out)
    return out


class GatedCrossAttention(Module):
    """RFD3 GatedCrossAttention on device; token grouping stays host-side."""

    def __init__(
        self,
        state_dict,
        ckc,
        c_query,
        c_kv,
        c_model=128,
        n_head=4,
        dtype=None,
    ):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.c_query = c_query
        self.c_kv = c_kv
        self.c_model = c_model
        self.n_head = n_head
        self.head_dim = c_model // n_head
        self.ln_q = self.torch_to_tt("ln_q.weight", dtype=self.dtype)
        self.ln_kv = self.torch_to_tt("ln_kv.weight", dtype=self.dtype)
        self.to_q = self.torch_to_tt("to_q.weight", dtype=self.dtype)
        self.to_k = self.torch_to_tt("to_k.weight", dtype=self.dtype)
        self.to_v = self.torch_to_tt("to_v.weight", dtype=self.dtype)
        self.to_g = self.torch_to_tt("to_g.0.weight", dtype=self.dtype)
        self.k_norm = self.torch_to_tt("k_norm.weight", dtype=self.dtype)
        self.q_norm = self.torch_to_tt("q_norm.weight", dtype=self.dtype)
        self.to_out_w = self.torch_to_tt("to_out.0.weight", dtype=self.dtype)
        self.to_out_b = self.torch_to_tt("to_out.0.bias", dtype=self.dtype)

    def _prepare_additive_mask(self, mask, batch, tokens, n_query, n_key):
        """Host mask -> device additive mask. Callers that reuse the SAME mask
        across multiple run_device() calls (e.g. an unchanged token-grouping
        mask across a block loop) should call this once and pass the result
        via attn_mask_dev instead of re-uploading identical data every call."""
        dev, dt = self.device, self.dtype
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        # Keyed on the SOURCE mask, not on the derived additive one: `torch.where` allocates a
        # fresh tensor per call, so keying on the result could never hit and would evict the
        # rest of the cache. `_grouping_buffers` is supposed to make this once-per-design, and
        # p41 measured it running twice on every step, so the upload is cached here as well.
        cache = self.__dict__.setdefault("_additive_mask_cache", {})
        ck = (mask.data_ptr(), tuple(mask.shape), tuple(mask.stride()), str(mask.dtype),
              mask._version, batch, tokens, n_query, n_key, dev.id(), str(dt))
        ent = cache.get(ck)
        if ent is not None and ent[1].is_allocated():
            return ent[1]
        add = torch.where(mask, 0.0, -1e4).to(torch.float32)
        add = add.expand(batch, -1, -1, -1).reshape(batch * tokens, 1, n_query, n_key)
        out = _tt(add, dev, dt)
        cache.clear()  # one mask per module; the reference in the entry pins the source
        cache[ck] = (mask, out)
        return out

    def run_device(self, q, kv, attn_mask=None, attn_mask_dev=None, n_key=None, q_pack=None, kv_pack=None):
        """q [B,T,Q,Cq], kv [B,T,K,Ckv]; return device [B,T,Q,Cq].
        A 3D q [B,T,Cq] is one query per token (the downcasts) and returns [B,T,Cq]. Its
        row-wise ops (norms, to_q, to_g, to_out) then run on T rows: the same call on
        q.unsqueeze(2) tile-pads every token's single row to 32 and does 32x the work.
        Likewise a 3D kv [B,T*K,Ckv] with `n_key`=K (the upcasts' 3 keys a token) runs
        ln_kv, to_k, to_v and k_norm on T*K rows instead of T tile-padded groups of K.
        attn_mask_dev (already-uploaded additive mask) takes priority over
        attn_mask (host bool mask, uploaded here) when both are given.
        q_pack = (pack, unpack) takes a flat atom q [B,L,Cq]: the q-side row ops run on L rows, `pack` maps
        qq to [B,T,Q,C] and `unpack` brings the attention output back to [B,L,C]; returns [B,L,Cq].
        kv_pack = pack takes a flat atom kv [B,L,Ckv]: ln_kv, to_k, to_v, k_norm run on L rows and
        `pack` maps [k|v] to [B,T,K,2C]."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        flat = len(q.shape) == 3 and _GCA_FLAT_Q and q_pack is None
        if len(q.shape) == 3 and not flat and q_pack is None:
            return ttnn.squeeze(self.run_device(ttnn.unsqueeze(q, 2), kv, attn_mask, attn_mask_dev, n_key), 2)
        if flat:
            batch, tokens, _ = q.shape
            n_query = 1
        elif q_pack is None:
            batch, tokens, n_query, _ = q.shape
        if n_key is None and kv_pack is None:
            n_key = kv.shape[2]
        q = ttnn.rms_norm(q, weight=self.ln_q, epsilon=1e-6, compute_kernel_config=ckc)
        kv = ttnn.rms_norm(kv, weight=self.ln_kv, epsilon=1e-6, compute_kernel_config=ckc)
        qq = ttnn.linear(q, self.to_q, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(kv, self.to_k, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(kv, self.to_v, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(q, self.to_g, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.q_norm, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.k_norm, epsilon=1e-6, compute_kernel_config=ckc)

        if flat:
            qq = ttnn.reshape(qq, (batch, tokens, 1, self.n_head * self.head_dim))
        if q_pack is not None:
            qq = q_pack[0](qq)
            batch, tokens, n_query, _ = qq.shape
        kvh = None
        if kv_pack is not None:
            kvh = kv_pack(ttnn.concat([kk, vv], dim=-1))
            ttnn.deallocate(kk)
            ttnn.deallocate(vv)
            n_key = kvh.shape[2]

        def split(x, count):
            x = ttnn.reshape(
                x, (batch, tokens, count, self.n_head, self.head_dim)
            )
            x = ttnn.permute(x, (0, 1, 3, 2, 4))
            return ttnn.reshape(
                x, (batch * tokens, self.n_head, count, self.head_dim)
            )

        # `gg` stays unsplit -- see _merge_heads / AttentionPairBias: an elementwise gate
        # commutes with the head merge exactly, so one split per call disappears.
        if _GCA_HEADS and self.head_dim % TILE == 0 and align_tile(n_query) == align_tile(n_key):
            # Both row counts pad to one tile, which is all the kernel checks; it labels every output with
            # q's row count, so k and v get theirs back by a view (same padded shape).
            bt, hp = batch * tokens, self.n_head * self.head_dim
            if kvh is None:
                kvh = ttnn.concat([kk, vv], dim=-1)
                ttnn.deallocate(kk)
                ttnn.deallocate(vv)
            qq, kt, vv = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.reshape(qq, (bt, 1, n_query, hp)), ttnn.reshape(kvh, (bt, 1, n_key, 2 * hp)),
                num_heads=self.n_head, num_kv_heads=self.n_head, transpose_k_heads=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(kvh)
            pad = align_tile(n_key)
            kt = ttnn.reshape(kt, ttnn.Shape([bt, self.n_head, self.head_dim, n_key]),
                              ttnn.Shape([bt, self.n_head, self.head_dim, pad]))
            vv = ttnn.reshape(vv, ttnn.Shape([bt, self.n_head, n_key, self.head_dim]),
                              ttnn.Shape([bt, self.n_head, pad, self.head_dim]))
        else:
            if kvh is not None:
                hp = self.n_head * self.head_dim
                kk, vv = kvh[:, :, :, :hp], kvh[:, :, :, hp:]
            qq = split(qq, n_query)
            kt = ttnn.permute(split(kk, n_key), (0, 1, 3, 2))
            vv = split(vv, n_key)
        scores = ttnn.matmul(
            qq, kt, compute_kernel_config=ckc,
            core_grid=_grid_if_single_k_tile(qq),
        )
        scores = ttnn.multiply(scores, self.head_dim**-0.5)
        if attn_mask_dev is not None:
            scores = ttnn.add(scores, attn_mask_dev)
        elif attn_mask is not None:
            scores = ttnn.add(scores, self._prepare_additive_mask(attn_mask, batch, tokens, n_query, n_key))
        attention = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attention, vv, compute_kernel_config=ckc, dtype=dt,
                          core_grid=_grid_if_single_k_tile(attention))
        out = _merge_heads(out, (batch, tokens, n_query, self.c_model))
        if flat:
            out = ttnn.reshape(out, (batch, tokens, self.c_model))
        if q_pack is not None:
            out = q_pack[1](out)
        out = ttnn.multiply(out, ttnn.sigmoid(gg))
        out = ttnn.linear(
            out,
            self.to_out_w,
            bias=self.to_out_b,
            compute_kernel_config=ckc,
            dtype=dt,
            core_grid=CORE_GRID_MAIN,
        )
        return out

    def __call__(self, q_host, kv_host, attn_mask=None):
        """Host-boundary wrapper used by isolated component tests."""
        q = _tt(q_host, self.device, self.dtype)
        kv = _tt(kv_host, self.device, self.dtype)
        return ttnn.to_torch(self.run_device(q, kv, attn_mask)).float()


# The atom/DiT blocks' five sigmoid gates per block (two adaLN gains, the attention gate, the two
# output gates) as the multiply's own operand activation instead of a separate sigmoid op: 5 fewer
# device programs per block, 180 a step in the DiT alone, where per-op fixed cost is the limit.
# RFD3_SIGMOID_FOLD=0 restores the separate op for the A/B.
_SIGMOID_FOLD = env_flag("RFD3_SIGMOID_FOLD", True)
# The DiT's conditioning `c` is the same for all 18 blocks of a call, and each block spends six linears and two
# rms_norms on it (adaLN gain and shift twice, two output gates): 216 launch-bound [L,384]x[384,768] linears a
# step. With this the stack runs them as three wide linears a call (the per-block ln_s folded into the weights,
# sigmoids in the linear) and each block reads its 768 columns in place (addcmul_at), as OpenFold3's dit_shoist.
# Folding ln_s and the fused sigmoid move rounding: graded on the normal bar (8 paired p150a seeds), so on in both modes.
_DIT_SHOIST = env_flag("RFD3_DIT_SHOIST", True)
_DIT_SHOIST_BYTES = 384 << 20   # the three wide bf16 tensors of one call; 685 tokens x 1 design is 114 MB


def _gated(x, gate):
    if _SIGMOID_FOLD:
        return ttnn.multiply(x, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    return ttnn.multiply(x, ttnn.sigmoid(gate))


def _sdpa_dense_mask(pair_bias, additive_mask, head_dim, n_key):
    """The dense SDPA mask from head-major pair bias [.., H, L, L]: the kernel adds the mask before
    it scales, so only the bias is pre-scaled; the -1e4 neighbour mask and the pad rows/columns
    stay where their weight is 0."""
    pb = ttnn.multiply(pair_bias, head_dim**0.5)
    mask = pad_axis(pad_axis(ttnn.add(pb, additive_mask), n_key, 3, -1e4), n_key, 2, -1e4)
    ttnn.deallocate(pb)
    return mask


class RFD3AtomBlock(Module):
    """One dense-mask RFD3 structure-local transformer block.

    Parameterized by dims so the same block serves the atom encoder/decoder
    (c_a=128, c_s=128, c_pair=16, n_head=4, head_dim=32) and the 18-block token
    DiT (c_a=768, c_s=384, c_pair=128, n_head=16, head_dim=48). Weight shapes
    encode c_a/c_s/c_pair; n_head is the only structural knob that is not."""

    def __init__(self, state_dict, ckc, c_a=128, c_s=128, c_pair=16, n_head=4, dtype=None,
                 fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        dt = self.dtype
        self.fp32_residual = fp32_residual
        self.n_head = n_head
        self.head_dim = c_a // n_head
        a = "attention_pair_bias."
        self.a_ln_s = self.torch_to_tt(a + "ada_ln_1.ln_s.weight", dtype=dt)
        self.a_gain_w = self.torch_to_tt(a + "ada_ln_1.to_gain.0.weight", dtype=dt)
        self.a_gain_b = self.torch_to_tt(a + "ada_ln_1.to_gain.0.bias", dtype=dt)
        self.a_bias_w = self.torch_to_tt(a + "ada_ln_1.to_bias.weight", dtype=dt)
        self.q_w = self.torch_to_tt(a + "to_q.weight", dtype=dt)
        self.k_w = self.torch_to_tt(a + "to_k.weight", dtype=dt)
        self.v_w = self.torch_to_tt(a + "to_v.weight", dtype=dt)
        self.b_w = self.torch_to_tt(a + "to_b.weight", dtype=dt)
        self.g_w = self.torch_to_tt(a + "to_g.0.weight", dtype=dt)
        self.q_ln = self.torch_to_tt(a + "ln_q.weight", dtype=dt)
        self.k_ln = self.torch_to_tt(a + "ln_k.weight", dtype=dt)
        self.o_w = self.torch_to_tt(a + "to_o.weight", dtype=dt)
        self.a_out_w = self.torch_to_tt(a + "linear_output_project.0.weight", dtype=dt)
        self.a_out_b = self.torch_to_tt(a + "linear_output_project.0.bias", dtype=dt)
        # The DiT's 48-wide heads, padded to a tile in the weights for the dense SDPA arm.
        self.pad_w = self._padded_head_weights(a) if _DIT_SDPA and self.head_dim % TILE else None
        self._pb_primed = None          # (p, [1, H, L, K] pair bias), set by _prime_atom_pair_bias

        t = "transition_block."
        self.t_ln_s = self.torch_to_tt(t + "ada_ln.ln_s.weight", dtype=dt)
        self.t_gain_w = self.torch_to_tt(t + "ada_ln.to_gain.0.weight", dtype=dt)
        self.t_gain_b = self.torch_to_tt(t + "ada_ln.to_gain.0.bias", dtype=dt)
        self.t_bias_w = self.torch_to_tt(t + "ada_ln.to_bias.weight", dtype=dt)
        self.t_fc1 = self.torch_to_tt(t + "linear_1.weight", dtype=dt)
        self.t_fc2 = self.torch_to_tt(t + "linear_2.weight", dtype=dt)
        self.t_fc3 = self.torch_to_tt(t + "linear_3.weight", dtype=dt)
        self.t_out_w = self.torch_to_tt(t + "linear_output_project.0.weight", dtype=dt)
        self.t_out_b = self.torch_to_tt(t + "linear_output_project.0.bias", dtype=dt)

    def _adaln(self, a, s, ln_s, gain_w, gain_b, bias_w):
        ckc, dt = self.compute_kernel_config, self.dtype
        a = ttnn.rms_norm(a, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.rms_norm(s, weight=ln_s, epsilon=1e-6, compute_kernel_config=ckc)
        gain = _tuned_linear(
            s, gain_w, bias=gain_b, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        bias = _tuned_linear(
            s, bias_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID
        )
        return ttnn.add(_gated(a, gain), bias)

    def _sparse_pair_bias(self, p, cache):
        """The compact `[1, H, L, K]` bf16 neighbour pair bias, for the fused score path.

        Same cache discipline as `_sparse_bias_f32` and for the same reason -- the decoder runs
        its three atom blocks twice per step on the same gathered pair features -- but nothing
        dense is built or held: `rfd3_bias.fused_scores_bias_fp32` reads these 3.4 MB directly and
        materialises the 180.6 MB fp32 bias one L1 tile at a time inside the op.
        """
        key = (id(p), id(self))
        if cache is not None:
            hit = cache.get((id(self), "pb"))
            if hit is not None and hit[0] == key and hit[1] is p:
                return hit[2]
        primed = self._pb_primed
        if primed is not None and primed[0] is p:
            pair_bias = primed[1]
        else:
            pair_bias = _tuned_linear(
                p, self.b_w, ckc=self.compute_kernel_config, dtype=self.dtype,
                core_grid=CORE_GRID_MAIN,
            )
            pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
        if cache is not None:
            cache[(id(self), "pb")] = (key, p, pair_bias)
        return pair_bias

    def _sdpa_bias(self, p, attn_idx_dev, cache, css=None):
        """The atom-SDPA mask: the neighbour bias times sqrt(head_dim), bf16, [1, H, N, N].

        The kernel adds the mask before it applies `scale`, so the compact [1, H, L, 128] bias is
        pre-scaled here, before it is spread. The -1e4 fill is not: after the kernel's scale it
        still sits ~1765 below every real score and its weight is exactly 0. Cached per pair
        stream like `_sparse_bias_f32`, so the decoder's second recycle reuses it.
        """
        key = (id(p), id(attn_idx_dev))
        if cache is not None:
            hit = cache.get((id(self), "sdpa"))
            if hit is not None and hit[0] == key and hit[1] is p:
                return hit[2]
        pb = ttnn.multiply(self._sparse_pair_bias(p, cache), self.head_dim**0.5)
        bias = rfd3_bias.sparse_mask_bf16(pb, attn_idx_dev,
                                          css=None if css is None else (css[1][0], css[2]))
        ttnn.deallocate(pb)
        if cache is not None:
            cache[(id(self), "sdpa")] = (key, p, bias)
        return bias

    def _sparse_bias_f32(self, p, dense_bias, attn_idx_dev, cache):
        """The dense fp32 attention bias for one atom block, built once per pair stream.

        The decoder runs its three atom blocks twice per diffusion step (n_recycle=2) and
        `_sparse_qk_inputs` hands both calls the SAME gathered pair features and the SAME
        neighbour index, so the second recycle rebuilds a bit-identical bias -- one
        `ttnn.scatter` plus one fp32 typecast per block, thrown away. That scatter cannot
        be made cheaper, only rarer: it is per-ELEMENT limited at ~9.7 G elem/s where
        `ttnn.add` over the identical tensor reaches 69.5 (p30 measured a bfloat8_b dense
        of HALF the bytes at the same 4.66 ms, and every reformulation -- gather with an
        inverted index, scatter_add, tosa_scatter, the scale_mask_softmax family -- either
        loses or rejects a per-row bias). So callers whose pair stream repeats pass a dict
        and pay for it once.

        The entry keeps its own reference to `p`, so its id cannot be recycled underneath
        the key while the entry lives -- the idiom `_sparse_qk_inputs` uses. A key miss
        overwrites the slot, which drops the previous bias; there is nothing to clear at a
        step boundary.
        """
        ckc, dt = self.compute_kernel_config, self.dtype
        key = (id(p), id(attn_idx_dev))
        if cache is not None:
            hit = cache.get(id(self))
            if hit is not None and hit[0] == key and hit[1] is p:
                return hit[2]
        pair_bias = _tuned_linear(
            p, self.b_w, ckc=ckc, dtype=dt, core_grid=CORE_GRID_MAIN,
        )
        pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
        if dense_bias is None:
            # One pass: the kernel fills the mask constant, pokes the neighbour bias and
            # writes fp32, which is the same values in the same positions as the three ops
            # below and is gated on torch.equal against them
            # (scripts/rfd3_port/p36_bias_kernel_probe.py). 0.932 ms against 5.437.
            bias_f = rfd3_bias.sparse_bias_fp32(pair_bias, attn_idx_dev)
        else:
            # dense_bias carries the -1e4 mask, so scattering the local pair bias into it
            # gives pair_bias at neighbours (as the dense path does) and leaves -1e4, whose
            # exp underflows to zero, everywhere else.
            bias = ttnn.scatter(dense_bias, 3, attn_idx_dev, pair_bias)
            bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
            ttnn.deallocate(bias)
        if cache is not None:
            cache[id(self)] = (key, p, bias_f)
        return bias_f

    def _adaln_at(self, a, hoist, i):
        """_adaln with the stack's wide sigmoid gains / shifts (LocalTokenTransformer._hoist), columns read in place."""
        sc, sb, _ = hoist
        w, off = self.head_dim * self.n_head, (2 * self.idx + i) * self.head_dim * self.n_head
        an = ttnn.rms_norm(a, epsilon=1e-6, compute_kernel_config=self.compute_kernel_config)
        out = _addcmul_at(sb, an, sc, w, x_off=off, z_off=off)
        if out is None:
            out = ttnn.add(ttnn.multiply(an, sc[..., off:off + w]), sb[..., off:off + w])
        ttnn.deallocate(an)
        return out

    def _residual_at(self, q, upd, hoist, i):
        """q + upd * sigmoid(gate) with the gate's columns read from the stack's wide sigmoid gates."""
        w, off = self.head_dim * self.n_head, (2 * self.idx + i) * self.head_dim * self.n_head
        out = _addcmul_at(q, upd, hoist[2], w, z_off=off)
        if out is None:
            out = ttnn.add(q, ttnn.multiply(upd, hoist[2][..., off:off + w]))
        return out

    def _transition(self, norm):
        ckc, dt = self.compute_kernel_config, self.dtype
        left = ttnn.linear(
            norm, self.t_fc1, activation="silu", compute_kernel_config=silu_ckc(ckc),
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        right = _tuned_linear(
            norm, self.t_fc2, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        return _tuned_linear(
            ttnn.multiply(left, right), self.t_fc3, ckc=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )

    def __call__(self, q, c, p, additive_mask=None, sparse_qk=None, bias_cache=None,
                 pair_bias=None, sdpa_mask=None, hoist=None):
        ckc, dt = self.compute_kernel_config, self.dtype
        f32 = self.fp32_residual
        if f32 and q.dtype != ttnn.float32:
            # promote the residual stream to fp32 on entry (first block); subsequent
            # blocks already receive an fp32 residual from the previous block.
            q = ttnn.typecast(q, ttnn.float32, memory_config=q.memory_config())
        batch, length = q.shape[0], q.shape[1]
        # matmuls/linears/norms run in bf16 (dt); only the residual accumulation is fp32,
        # so no fp32 matmul is ever issued (Blackhole fp32 matmul is a host-fallback dead-end).
        q_compute = ttnn.typecast(q, dt, memory_config=q.memory_config()) if f32 else q
        if hoist is not None and f32:
            hoist = None
        norm = self._adaln_at(q_compute, hoist, 0) if hoist is not None else self._adaln(
            q_compute, c, self.a_ln_s, self.a_gain_w, self.a_gain_b, self.a_bias_w
        )
        n_key = align_tile(length)
        if (self.pad_w is not None and sparse_qk is None and dt == ttnn.bfloat16
                and (sdpa_mask is not None
                     or 2 * self.n_head * n_key * n_key <= _ATOM_SDPA_MAX_MASK_BYTES)):
            out = self._dense_sdpa_padded(norm, p, additive_mask, pair_bias, sdpa_mask, batch, length)
        else:
            out = self._attention(norm, p, additive_mask, sparse_qk, bias_cache, pair_bias, sdpa_mask,
                                  batch, length)
        if hoist is not None:
            q = self._residual_at(q, out, hoist, 0)
            ttnn.deallocate(out)
            norm = self._adaln_at(q, hoist, 1)
            update = self._transition(norm)
            q2 = self._residual_at(q, update, hoist, 1)
            ttnn.deallocate(update)
            return q2
        gate = ttnn.linear(
            c, self.a_out_w, bias=self.a_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        upd = _gated(out, gate)
        if f32:
            q = ttnn.add(q, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config()))
        else:
            q = ttnn.add(q, upd)
        ttnn.deallocate(upd)

        q_compute = ttnn.typecast(q, dt, memory_config=q.memory_config()) if f32 else q
        norm = self._adaln(
            q_compute, c, self.t_ln_s, self.t_gain_w, self.t_gain_b, self.t_bias_w
        )
        update = self._transition(norm)
        gate = ttnn.linear(
            c, self.t_out_w, bias=self.t_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        upd = _gated(update, gate)
        if f32:
            q = ttnn.add(q, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config()))
        else:
            q = ttnn.add(q, upd)
        return q

    def _attention(self, norm, p, additive_mask, sparse_qk, bias_cache, pair_bias, sdpa_mask,
                   batch, length):
        """Attention with pair bias on the adaLN'd stream, through the output projection."""
        ckc, dt = self.compute_kernel_config, self.dtype
        qq = _tuned_linear(norm, self.q_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        kk = _tuned_linear(norm, self.k_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        vv = _tuned_linear(norm, self.v_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        gg = _tuned_linear(norm, self.g_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        qq = ttnn.rms_norm(qq, weight=self.q_ln, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.k_ln, epsilon=1e-6, compute_kernel_config=ckc)

        def heads(x):
            x = ttnn.reshape(
                x, (batch, length, self.n_head, self.head_dim)
            )
            return ttnn.permute(x, (0, 2, 1, 3))

        # `gg` is NOT split: the gate is elementwise, so multiplying it before or after the
        # merge touches the same pairs of values in the same order. Gating after the merge
        # deletes one head split per block -- 36 of the DiT's 144 per step -- and is
        # bit-exact by construction rather than by measurement.
        if _QKV_HEADS and self.head_dim % TILE == 0:
            hp = self.n_head * self.head_dim
            qkv = ttnn.reshape(ttnn.concat([qq, kk, vv], dim=-1), (batch, 1, length, 3 * hp))
            for x in (qq, kk, vv):
                ttnn.deallocate(x)
            qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.n_head, num_kv_heads=self.n_head, transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(qkv)
        else:
            qq, kk, vv = map(heads, (qq, kk, vv))
        # Attention reduces over the key axis, and a ttnn softmax over a last dim that is
        # not a tile multiple reads that axis' tile padding -- which no op guarantees to
        # have written. So extend the key axis logically: zero keys (contributing a score
        # of 0, masked to -1e4) instead of 18 columns of whatever DRAM held. Free at these
        # shapes -- the buffers were tile-padded to the same size already.
        n_key = align_tile(length)
        kk = pad_axis(kk, n_key, 2, 0.0)
        vv = pad_axis(vv, n_key, 2, 0.0)
        # Block-sparse arm. Replaces the whole tail -- scores, bias, softmax and the value
        # matmul -- with a batched dense chain over the block's own key union, and hands back the
        # same [1,H,length,head_dim] the dense chain would. Off by default; see block_sparse.py.
        bs_out = None
        block = sparse_qk[4] if sparse_qk is not None and len(sparse_qk) > 4 else None
        if (block is not None and sparse_qk[2] is None
                and rfd3_bias.fused_enabled() and dt == ttnn.bfloat16):
            nb, q_block, u_width, gather_dev, pos_rm = block
            pb = self._sparse_pair_bias(p, bias_cache)
            # The blocked scores have nb*q_block rows and the compact bias has `length`, so the
            # pad rows need a bias too. -1e4 makes them fully masked; they are sliced off the
            # output either way.
            pb = pad_axis(pb, nb * q_block, 2, -1e4)
            bs_out = _BS.attention(qq, kk, vv, pb, pos_rm, gather_dev, nb, q_block, u_width,
                                   self.head_dim**-0.5, dt, ckc)
        if (bs_out is None and _ATOM_SDPA and sparse_qk is not None and sparse_qk[2] is None
                and rfd3_bias.fused_enabled() and dt == ttnn.bfloat16
                and 2 * self.n_head * n_key * n_key <= _ATOM_SDPA_MAX_MASK_BYTES):
            css = _css_for(sparse_qk[1]) if batch == 1 else None
            mask = self._sdpa_bias(p, sparse_qk[1], bias_cache, css)
            if css is None:
                o = fused_sdpa(pad_axis(qq, n_key, 2, 0.0), kk, vv, attn_mask=mask,
                               scale=self.head_dim**-0.5,
                               program_config=_sdpa_program_config(*_ATOM_SDPA_CHUNKS),
                               compute_kernel_config=ckc)
            else:
                dev = qq.device()
                g = dev.compute_with_storage_grid_size()
                o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.n_head, n_key, self.head_dim]), dt,
                                                   ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
                _SG.sdpa(dev, pad_axis(qq, n_key, 2, 0.0), kk, vv, mask, o, css[2], css[2], (g.x, g.y),
                         (ckc.math_fidelity, ckc.math_approx_mode, ckc.fp32_dest_acc_en, False),
                         self.head_dim**-0.5, css=css[1])
                CSS_CALLS[0] += 1
            bs_out = ttnn.slice(o, [0, 0, 0, 0], [batch, self.n_head, length, self.head_dim])
            ATOM_SDPA_CALLS[0] += 1
        elif bs_out is not None:
            _BS.STATS[0] += 1
        elif sparse_qk is not None and len(sparse_qk) > 4:
            _BS.STATS[1 if _BS.enabled() else 2] += 1
        fused = False
        dense_fused = False
        gathered = None
        if bs_out is None and sparse_qk is None:
            # `pair_bias` arrives precomputed when the caller hoisted all of its blocks'
            # projections into one matmul; see _PAIRBIAS_FUSED. `sdpa_mask` arrives when the
            # caller also built every block's SDPA mask at once (_dit_sdpa_masks).
            if sdpa_mask is None:
                if pair_bias is None:
                    pair_bias = _tuned_linear(
                        p, self.b_w, ckc=ckc, dtype=dt, core_grid=CORE_GRID_MAIN,
                    )
                pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
            if sdpa_mask is not None or (_DIT_SDPA and dt == ttnn.bfloat16
                    and 2 * self.n_head * n_key * n_key <= _ATOM_SDPA_MAX_MASK_BYTES):
                mask = sdpa_mask
                if mask is None:
                    mask = _sdpa_dense_mask(pair_bias, additive_mask, self.head_dim, n_key)
                    ttnn.deallocate(pair_bias)
                # SDPA refuses a head_dim that is not a tile multiple (the DiT's is 48), so the
                # heads get zero columns out to the next tile: they add 0 to every q.k and come
                # back as zero columns of the output, sliced off below. The scale stays 48^-0.5.
                hd = align_tile(self.head_dim)
                qq, kk, vv = (pad_axis(x, hd, 3, 0.0) for x in (qq, kk, vv))
                o = fused_sdpa(pad_axis(qq, n_key, 2, 0.0), kk, vv, attn_mask=mask,
                               scale=self.head_dim**-0.5,
                               program_config=_dense_sdpa_config(n_key),
                               compute_kernel_config=ckc)
                ttnn.deallocate(mask)
                bs_out = ttnn.slice(o, [0, 0, 0, 0], [batch, self.n_head, length, self.head_dim])
                DIT_SDPA_CALLS[0] += 1
        if bs_out is not None:
            pass
        elif sparse_qk is None:
            bias = pad_axis(ttnn.add(pair_bias, additive_mask), n_key, 3, -1e4)
            scores = ttnn.matmul(
                qq, ttnn.permute(kk, (0, 1, 3, 2)),
                compute_kernel_config=ckc,
            )
            # L2: one kernel for the tail of this branch -- widen both bf16 operands, scale
            # and add in an fp32 DST -- so neither fp32 operand is ever written to DRAM. The three
            # ops it replaces move r 30.9 + w 92.7 MB per call at the page fixture where the kernel
            # moves r 30.9 + w 30.9, and it measures 0.5093 -> 0.2038 ms/call, 2.499x, with no rung
            # of the size ladder regressing (perf/p72/dense_kernel_probe.json). Bit-exact by
            # torch.equal at every rung, not by tolerance: ttnn own folded form
            # `add(bf16, bf16, dtype=fp32, act=scale)` is 2.55 maxabs off the three-op chain,
            # because its activation pass packs the scaled operand back at the INPUT dtype and
            # rounds it to bf16. RFD3_DENSE_BIAS_FUSED=0 restores the three ops for the A/B.
            dense_fused = rfd3_bias.dense_eligible(scores, bias)
            if not dense_fused:
                bias_f = ttnn.typecast(
                    bias, ttnn.float32, memory_config=bias.memory_config()
                )
        else:
            # Only the pair-bias projection is sparsified. QK stays dense: it
            # reduces over head_dim (a single tile deep), so its dot-product tree
            # is independent of the M/N tiling and the scores are bit-identical to
            # the gathered form -- while at L=3359 the dense matmul costs 0.375 ms
            # against 33.9 ms for a gathered [1,32]@[32,128] batch plus scatter.
            n_keys, attn_idx_dev, dense_bias, gathered, block = sparse_qk
            # L6b: one kernel for the last five ops of this path -- the mask template, the
            # scatter of the neighbour bias, both widens and the scaled add. It reads the bf16
            # scores and the compact pair bias and writes `scores*scale + bias` in fp32, so the
            # dense fp32 bias never exists in DRAM and 8.5 ms/call of traffic becomes 1.67.
            # Bit-exact against the five ops it replaces by construction and by torch.equal at
            # the production shape (scripts/rfd3_port/p42_fused_scores_probe.py), including on
            # the softmax that consumes it. `dense_bias is None` is L6a's own gate.
            fused = dense_bias is None and rfd3_bias.fused_enabled() and dt == ttnn.bfloat16
            if fused:
                pair_bias = self._sparse_pair_bias(p, bias_cache)
            else:
                bias_f = self._sparse_bias_f32(p, dense_bias, attn_idx_dev, bias_cache)
            scores = ttnn.matmul(
                qq, ttnn.permute(kk, (0, 1, 3, 2)), compute_kernel_config=ckc,
            )
        if bs_out is None:
            if dense_fused:
                scores = rfd3_bias.dense_fused_scores_bias_fp32(
                    scores, bias, self.head_dim**-0.5
                )
            elif fused:
                scores = rfd3_bias.fused_scores_bias_fp32(
                    scores, pair_bias, attn_idx_dev, self.head_dim**-0.5
                )
            else:
                scores = ttnn.typecast(
                    scores, ttnn.float32, memory_config=scores.memory_config()
                )
                # Scale and add in ONE op: the scale rides on operand a as a MUL_UNARY_SFPU activation.
                # Bit-exact rather than close, and by measurement rather than by argument
                # (scripts/rfd3_port/p35_dense_chain_price.py, perf/p35/dense_chain_qb1c0.json): 1.285 ms
                # against 2.246 for the pair at [1,4,3359,3360], torch.equal on the softmax output. It
                # holds here because both operands are already fp32, so the op's destination register is
                # fp32 and nothing is silently computed at the input dtype -- the trap that stops a
                # bf16->fp32 widen from folding into a binary op the same way. Do not copy this to a
                # bf16 site (GatedCrossAttention.run_device) on the strength of this comment: there the
                # split form rounds the scaled scores to bf16 before the add and the folded form does not.
                scores = ttnn.add(scores, bias_f, input_tensor_a_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.head_dim**-0.5)])
            if gathered is None:
                # L5b: the softmax kernel MACs the value tiles where it would otherwise pack the
                # normalised row to DRAM, so neither the 294.3 MB attention write nor the PV
                # matmul's read of it happens. It declines -- returning None -- on every shape
                # where it cannot reproduce `attn_value_matmul`'s accumulation grouping bit for
                # bit; see `softmax_generic.pv_classify`. Default ON, behind
                # `RFD3_SOFTMAX_PV_FUSED`: composed with the split `fc1` it is -6.206 s/design and
                # 1.0671x at 685 tokens (`perf/p132/both_ab_R4_warmfix.json`).
                fused_pv = softmax_generic.softmax_pv_fused(scores, vv, dt, ckc)
                if fused_pv is not None:
                    ttnn.deallocate(scores)
                    out = fused_pv
                    attention = None
                else:
                    attention = softmax_generic.softmax_bf16(scores, dt)
            else:
                # Reduce over the 128 columns that carry a value, not over all 6080. Every row has
                # exactly 128 valid indices in [0, L) -- _extend_with_neighbours fills the sequence
                # slots and tops up from the distance topk, and _create_attention_indices sorts them --
                # so there is no ragged row and no index outside the key axis.
                gather_idx, zeros = gathered
                compact = ttnn.gather(scores, 3, gather_idx)
                ttnn.deallocate(scores)
                weights = softmax_generic.softmax_bf16(compact, dt)
                ttnn.deallocate(compact)
                attention = ttnn.scatter(zeros, 3, gather_idx, weights)
                ttnn.deallocate(weights)
            if attention is not None:
                out = attn_value_matmul(attention, vv, ckc, dt)
        else:
            out = bs_out
        out = _merge_heads(out, (batch, length, self.n_head * self.head_dim))
        out = _gated(out, gg)
        out = _tuned_linear(
            out, self.o_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID
        )
        return out

    def _dense_sdpa_padded(self, norm, p, additive_mask, pair_bias, sdpa_mask, batch, length):
        """The DiT's dense SDPA with its 48-wide heads padded to 64 in the weights (`pad_w`).

        q/k/v/g come out of their projections already head-major-padded, so one
        `nlp_create_qkv_heads` splits them and one `nlp_concat_heads` merges the output, where
        the 48-wide layout needs a reshape + permute + pad per tensor and a slice + permute +
        reshape back. The zero columns add 0 to every q.k and come back as zero output columns,
        which the padded rows of `o` (zero) drop. gap_dit_heads.py: 1.27 -> 0.39 ms per call at
        685 tokens, same error against float64.
        """
        ckc, dt, w = self.compute_kernel_config, self.dtype, self.pad_w
        lin = dict(ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        n_key, hp = align_tile(length), self.n_head * align_tile(self.head_dim)
        qq = ttnn.rms_norm(_tuned_linear(norm, w["q"], **lin), weight=w["q_ln"], epsilon=w["eps"],
                           compute_kernel_config=ckc)
        kk = ttnn.rms_norm(_tuned_linear(norm, w["k"], **lin), weight=w["k_ln"], epsilon=w["eps"],
                           compute_kernel_config=ckc)
        vv = _tuned_linear(norm, w["v"], **lin)
        gg = _tuned_linear(norm, w["g"], **lin)
        qkv = ttnn.reshape(ttnn.concat([qq, kk, vv], dim=-1), (batch, 1, length, 3 * hp))
        for x in (qq, kk, vv):
            ttnn.deallocate(x)
        qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.n_head, num_kv_heads=self.n_head, transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        mask = sdpa_mask
        if mask is None:
            if pair_bias is None:
                pair_bias = _tuned_linear(p, self.b_w, ckc=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
            pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
            mask = _sdpa_dense_mask(pair_bias, additive_mask, self.head_dim, n_key)
            ttnn.deallocate(pair_bias)
        o = fused_sdpa(pad_axis(qq, n_key, 2, 0.0), pad_axis(kk, n_key, 2, 0.0), pad_axis(vv, n_key, 2, 0.0),
                       attn_mask=mask, scale=self.head_dim**-0.5,
                       program_config=_dense_sdpa_config(n_key), compute_kernel_config=ckc)
        ttnn.deallocate(mask)
        DIT_SDPA_CALLS[0] += 1
        o = ttnn.experimental.nlp_concat_heads(o)                            # [B, 1, n_key, hp]
        o = ttnn.reshape(ttnn.slice(o, [0, 0, 0, 0], [batch, 1, length, hp]), (batch, length, hp))
        return _tuned_linear(_gated(o, gg), w["o"], **lin)

    def _padded_head_weights(self, a):
        """q/k/v/g with zero output columns after each head and o with zero input rows, so every
        head is a tile wide. rms_norm over the padded width sees the same sum of squares over
        more columns, so its weight is scaled by sqrt(c/padded) and its eps by c/padded."""
        hd, dp, c = self.head_dim, align_tile(self.head_dim), self.n_head * self.head_dim
        dt, s = self.dtype, c / (self.n_head * dp)

        def cols(w):                                                     # [in, c] -> [in, H*dp]
            return F.pad(w.reshape(w.shape[0], self.n_head, hd), (0, dp - hd)).reshape(w.shape[0], -1)

        out = {k: self.torch_to_tt(a + n, transform=lambda w: cols(w.t()), dtype=dt)
               for k, n in (("q", "to_q.weight"), ("k", "to_k.weight"), ("v", "to_v.weight"),
                            ("g", "to_g.0.weight"))}
        out["o"] = self.torch_to_tt(a + "to_o.weight", transform=lambda w: cols(w).t(), dtype=dt)
        for k, n in (("q_ln", "ln_q.weight"), ("k_ln", "ln_k.weight")):
            out[k] = self.torch_to_tt(a + n, transform=lambda w: cols(w.reshape(1, -1) * s ** 0.5)[0],
                                      dtype=dt)
        out["eps"] = 1e-6 * s
        return out



def _prime_atom_pair_bias(owner, blocks, p, cache=None):
    """Hand every atom block its [1, H, L, K] neighbour pair bias from ONE projection of the shared
    pair features `p` (see _ATOM_PB_FUSED). Skipped when off, or when `cache` already holds this
    stream's masks (the decoder's second recycle builds nothing). Returns the blocks it primed, for
    _unprime_atom_pair_bias."""
    if not _ATOM_PB_FUSED:
        return ()
    if cache is not None:
        hit = cache.get((id(blocks[0]), "sdpa")) or cache.get((id(blocks[0]), "pb"))
        if hit is not None and hit[1] is p:
            return ()
    w = getattr(owner, "_atom_b_w", None)
    if w is None:
        cols = torch.cat([b.weights["attention_pair_bias.to_b.weight"].t().float() for b in blocks], dim=1)
        # Straight from the torch weights, not torch_to_tt: see the DiT's b_w_fused.
        w = owner._atom_b_w = ttnn.from_torch(cols, layout=ttnn.TILE_LAYOUT, device=owner.device,
                                              dtype=owner.dtype)
    pb = _tuned_linear(p, w, ckc=owner.compute_kernel_config, dtype=owner.dtype, core_grid=CORE_GRID_MAIN)
    pb_t = ttnn.permute(pb, (0, 3, 1, 2))
    ttnn.deallocate(pb)
    B, _, L, K = (int(d) for d in pb_t.shape)
    lo = 0
    for b in blocks:
        b._pb_primed = (p, ttnn.slice(pb_t, [0, lo, 0, 0], [B, lo + b.n_head, L, K]))
        lo += b.n_head
    ttnn.deallocate(pb_t)
    return blocks


def _unprime_atom_pair_bias(blocks):
    for b in blocks:
        b._pb_primed = None


class LocalAtomTransformer(Module):
    """Three-block RFD3 atom encoder with parity-preserving sparse QK."""

    def __init__(self, state_dict, ckc, n_blocks=3, dtype=None, fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.blocks = [
            RFD3AtomBlock(self.scope(f"blocks.{i}"), ckc, dtype=self.dtype,
                          fp32_residual=fp32_residual)
            for i in range(n_blocks)
        ]
        self._mask_cache = {}  # one -1e4 scatter template, see _mask_template

    def run_device(self, q, c, p, additive_mask):
        for block in self.blocks:
            q = block(q, c, p, additive_mask)
        return q

    def __call__(self, q_host, c_host, p_host, indices, resident=False):
        """q/c may already be on the card; `resident` returns the device tensor."""
        dt, dev = self.dtype, self.device
        p_host = p_host.unsqueeze(0) if p_host.ndim == 3 else p_host
        if env_flag("RFD3_SPARSE_QK", True):
            p, n_keys, attn_idx_dev, dense_bias, gathered, block = _sparse_qk_inputs(
                p_host, indices, dev, dt, mask_cache=self._mask_cache
            )
            q, c = _dev(q_host, dev, dt), _dev(c_host, dev, dt)
            sparse_qk = (n_keys, attn_idx_dev, dense_bias, gathered, block)
            primed = _prime_atom_pair_bias(self, self.blocks, p)
            for block in self.blocks:
                q = block(q, c, p, sparse_qk=sparse_qk)
            _unprime_atom_pair_bias(primed)
            out = q
        else:
            q = _dev(q_host, dev, dt)
            c = _dev(c_host, dev, dt)
            p = _tt(p_host, dev, dt)
            mask = _tt(_dense_attention_mask(indices), dev, dt)
            out = self.run_device(q, c, p, mask)
        return out if resident else ttnn.to_torch(out).float()


class CompactStreamingDecoder(Module):
    """RFD3 decoder: three device Upcast/atom blocks plus device Downcast.

    pack_idx/unpack_idx/valid/upcast_mask_dev depend only on tok_idx, which is the SAME
    object for an entire design's sampling loop (RFD3Sampler.sample passes one `f` dict by
    reference through every step), so they are cached by id(tok_idx) and rebuilt only when
    that identity (or shape) changes. a/q/c/p/mask are uploaded every call."""

    def __init__(self, state_dict, ckc, dtype=None, fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self._design_state = None  # {"key", "valid", "pack_idx_dev", "unpack_idx_dev", "upcast_mask_dev"}
        self._mask_cache = {}      # one -1e4 scatter template, see _mask_template
        self._bias_cache = {}      # one dense bias per atom block, see _sparse_bias_f32
        self.upcast = [
            GatedCrossAttention(
                self.scope(f"upcast.{i}.gca"), ckc,
                c_query=128, c_kv=256, dtype=self.dtype,
            )
            for i in range(3)
        ]
        # The decoder atom blocks stay bf16 even when fp32_residual is requested: the
        # decoder interleaves a GatedCrossAttention (upcast) between atom blocks, and an
        # fp32 residual would leak into that GCA's matmul -> host fp32 fallback. The
        # decoder's atom-grouping gathers also truncate the inter-block residual to bf16
        # (ttnn.embedding), so an fp32 residual here would not compound anyway. fp32-residual
        # is only useful on a pure atom-block stack whose output hits a casting boundary
        # (the encoder + the DiT), not on the decoder's interleaved layout.
        self.atom_blocks = [
            RFD3AtomBlock(
                self.scope(f"atom_transformer.{i}"), ckc, dtype=self.dtype
            )
            for i in range(3)
        ]
        self.downcast = GatedCrossAttention(
            self.scope("downcast.gca"), ckc,
            c_query=768, c_kv=128, dtype=self.dtype,
        )
        self.process_s_n = self.torch_to_tt(
            "downcast.process_s.0.weight", dtype=self.dtype
        )
        self.process_s_w = self.torch_to_tt(
            "downcast.process_s.1.weight", dtype=self.dtype
        )

    def _grouping_indices(self, tok_idx, batch):
        valid = _build_valid_mask(tok_idx)
        length = tok_idx.numel()
        padded = torch.full(valid.shape, length, dtype=torch.int64)
        padded[valid] = torch.arange(length)
        pack = torch.cat(
            [padded.reshape(-1) + b * (length + 1) for b in range(batch)]
        )
        flat_valid = valid.flatten().nonzero(as_tuple=False).squeeze(1)
        unpack = torch.cat(
            [flat_valid + b * valid.numel() for b in range(batch)]
        )
        return valid, pack, unpack

    def _pack_atoms_device(self, q, pack_idx_dev, valid):
        batch, length, channels = q.shape
        orig_dt = q.dtype
        # ttnn.embedding requires bf16; the gather is a pure reindex (exact), so round-trip
        # through bf16 only for the embedding op, then restore the compute dtype.
        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.pad(q, [[0, 0], [0, 1], [0, 0]], 0.0)
        q = ttnn.reshape(q, (batch * (length + 1), channels))
        if orig_dt != ttnn.bfloat16:
            q = ttnn.typecast(q, ttnn.bfloat16)
        packed = ttnn.embedding(
            pack_idx_dev, q, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if orig_dt != ttnn.bfloat16:
            packed = ttnn.typecast(packed, orig_dt)
        packed = ttnn.reshape(
            packed, (batch, valid.shape[0], valid.shape[1], channels)
        )
        return ttnn.to_layout(packed, ttnn.TILE_LAYOUT)

    def _unpack_atoms_device(self, q, unpack_idx_dev, length):
        batch, tokens, atoms, channels = q.shape
        orig_dt = q.dtype
        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.reshape(q, (batch * tokens * atoms, channels))
        if orig_dt != ttnn.bfloat16:
            q = ttnn.typecast(q, ttnn.bfloat16)
        unpacked = ttnn.embedding(
            unpack_idx_dev, q, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if orig_dt != ttnn.bfloat16:
            unpacked = ttnn.typecast(unpacked, orig_dt)
        unpacked = ttnn.reshape(unpacked, (batch, length, channels))
        return ttnn.to_layout(unpacked, ttnn.TILE_LAYOUT)

    def run_device(
        self, a, q, c, p, mask, upcast_mask_dev, pack_idx_dev,
        unpack_idx_dev, valid, length, sparse_qk=None, bias_cache=None,
    ):
        """Pure-device core loop (pack -> upcast -> unpack -> atom_block, x3).
        `a` is the FLAT (not pre-split) atom-pair stream."""
        # The upcasts' keys: each token's 768 channels as 3 rows of 256, flat when _GCA_FLAT_Q.
        a_split = ttnn.reshape(a, (a.shape[0], a.shape[1] * 3, 256) if _GCA_FLAT_Q else (a.shape[0], a.shape[1], 3, 256))
        primed = _prime_atom_pair_bias(self, self.atom_blocks, p, bias_cache) if sparse_qk is not None else ()
        q_pack = (lambda x: self._pack_atoms_device(x, pack_idx_dev, valid),
                  lambda x: self._unpack_atoms_device(x, unpack_idx_dev, length))
        for upcast, atom_block in zip(self.upcast, self.atom_blocks):
            if _GCA_ATOM_ROWS:
                q = ttnn.add(q, upcast.run_device(q, a_split, attn_mask_dev=upcast_mask_dev, n_key=3, q_pack=q_pack))
            else:
                q_grouped = self._pack_atoms_device(q, pack_idx_dev, valid)
                q_grouped = ttnn.add(
                    q_grouped, upcast.run_device(q_grouped, a_split, attn_mask_dev=upcast_mask_dev, n_key=3)
                )
                q = self._unpack_atoms_device(q_grouped, unpack_idx_dev, length)
            q = atom_block(q, c, p, mask, sparse_qk=sparse_qk, bias_cache=bias_cache)
        _unprime_atom_pair_bias(primed)
        return q

    def _design_buffers(self, tok_idx, batch):
        dev = self.device
        key = (id(tok_idx), tok_idx.shape, batch)
        st = self._design_state
        if st is None or st["key"] != key:
            valid, pack_indices, unpack_indices = self._grouping_indices(tok_idx, batch)
            pack_idx_dev = _tt_idx(pack_indices, dev)
            unpack_idx_dev = _tt_idx(unpack_indices, dev)
            valid_q = valid.unsqueeze(-1).expand(-1, -1, 3)
            upcast_mask_dev = self.upcast[0]._prepare_additive_mask(
                valid_q, batch, valid.shape[0], valid.shape[1], 3)
            st = dict(key=key, valid=valid, pack_idx_dev=pack_idx_dev,
                      unpack_idx_dev=unpack_idx_dev, upcast_mask_dev=upcast_mask_dev)
            self._design_state = st
        return st["valid"], st["pack_idx_dev"], st["unpack_idx_dev"], st["upcast_mask_dev"]

    def __call__(self, a_host, s_host, q_host, c_host, p_host, tok_idx, indices):
        """Host-in/host-out, for the component parity scripts. The per-step path calls
        `run_full_device` and keeps both outputs on the card."""
        a_out, q = self.run_full_device(a_host, s_host, q_host, c_host, p_host, tok_idx, indices)
        return ttnn.to_torch(a_out).float(), ttnn.to_torch(q).float()

    def run_full_device(self, a_host, s_host, q_host, c_host, p_host, tok_idx, indices):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        batch, length, _ = q_host.shape
        valid, pack_idx_dev, unpack_idx_dev, upcast_mask_dev = self._design_buffers(tok_idx, batch)
        p_host = p_host.unsqueeze(0) if p_host.ndim == 2 else p_host

        if env_flag("RFD3_SPARSE_QK", True):
            p, n_keys, attn_idx_dev, dense_bias, gathered, block = _sparse_qk_inputs(
                p_host, indices, dev, dt, mask_cache=self._mask_cache
            )
            sparse_qk = (n_keys, attn_idx_dev, dense_bias, gathered, block)
            a, q, c = (_dev(x, dev, dt) for x in (a_host, q_host, c_host))
            # The two recycle calls in a step share p and the neighbour index, so
            # each atom block's dense bias is bit-identical between them; build it
            # on the first call only (see GatedCrossAttention._sparse_bias_f32).
            q = self.run_device(
                a, q, c, p, None, upcast_mask_dev, pack_idx_dev,
                unpack_idx_dev, valid, length, sparse_qk=sparse_qk,
                bias_cache=self._bias_cache,
            )
        else:
            a, q, c = (_dev(x, dev, dt) for x in (a_host, q_host, c_host))
            p = _tt(p_host, dev, dt)
            mask = _tt(_dense_attention_mask(indices), dev, dt)
            q = self.run_device(a, q, c, p, mask, upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length)

        down_mask = valid.unsqueeze(1)
        if _GCA_ATOM_ROWS:
            a_update = self.downcast.run_device(
                a, q, down_mask, kv_pack=lambda x: self._pack_atoms_device(x, pack_idx_dev, valid))
        else:
            a_update = self.downcast.run_device(a, self._pack_atoms_device(q, pack_idx_dev, valid), down_mask)
        s = _dev(s_host, dev, dt)
        s = ttnn.rms_norm(
            s, weight=self.process_s_n, epsilon=1e-6, compute_kernel_config=ckc
        )
        s = ttnn.linear(
            s, self.process_s_w, compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        return ttnn.add(ttnn.add(a, a_update), s), q


class LinearSequenceHead(Module):
    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.weight = self.torch_to_tt("linear.weight", dtype=self.dtype)
        self.bias = self.torch_to_tt("linear.bias", dtype=self.dtype)
        self.valid_out_mask = self.weights["valid_out_mask"].bool()

    def __call__(self, a):
        """`a` may be a host tensor (the parity-script path) or an already-resident ttnn
        one handed over by the decoder."""
        if not isinstance(a, ttnn.Tensor):
            a = _tt(a, self.device, self.dtype)
        logits = ttnn.linear(
            a,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            core_grid=CORE_GRID_MAIN,
        )
        logits = ttnn.to_torch(logits).float()
        masked = logits.masked_fill(~self.valid_out_mask.view(1, 1, -1), float("-inf"))
        return logits, masked.argmax(dim=-1)


def _scaled_distogram_bins(R_L, min_dist=1.0, max_dist=30.0, sigma_data=16.0, n_bins=65):
    """Host port of block_utils.bucketize_scaled_distogram, stopping at the bin index.

    R_L: [B, N, 3] -> int32 [B, N, N]. The one-hot expansion this feeds is done on
    device (see DiffusionTokenEncoder._onehot_dev): materializing it here costs a
    [B,N,N,65] int64 tensor plus its float copy on host and then uploads 65 bytes
    per pair where 4 would do -- 51.0 ms of host work plus a 48.9 ms upload at
    B=8/N=250, against 1.6 + 0.3 + 7.2 ms for bins + upload + device one-hot.
    """
    lo, hi = min_dist / sigma_data, max_dist / sigma_data
    bins = torch.linspace(lo, hi, n_bins - 1, device=R_L.device)
    if _BINS_NB and R_L.device.type == "cpu" and R_L.dtype == torch.float32:
        return _distbins.bins(R_L, bins)
    D_LL = torch.linalg.norm(R_L.unsqueeze(-2) - R_L.unsqueeze(-3), dim=-1)  # [B, N, N]
    return torch.bucketize(D_LL, bins).to(torch.int32)


class DiffusionTokenEncoder(Module):
    """RFD3 DiffusionTokenEncoder: self-conditioning distogram + noise distogram -> 2-block
    no-triangle Pairformer. Reuses the verified PairformerBlock (c_s=384, c_z=128, n_head=16)."""

    C_S, C_Z, N_HEAD = 384, 128, 16
    N_BINS, N_PAIRFORMER = 65, 2

    def __init__(self, state_dict, ckc, sigma_data=16.0, dtype=None, fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.fp32_residual = fp32_residual
        self.sigma_data = sigma_data
        self.transition_1 = [Transition(self.scope(f"transition_1.{i}"), ckc, self.C_S, n=2, dtype=self.dtype)
                             for i in range(2)]
        cat_c_z = self.C_Z + self.N_BINS + self.N_BINS  # 128 + 65 (distogram) + 65 (self)
        self.process_z_n = self.torch_to_tt("process_z.0.weight", dtype=self.dtype)
        self.process_z_w = self.torch_to_tt("process_z.1.weight", dtype=self.dtype)
        self.transition_2 = [Transition(self.scope(f"transition_2.{i}"), ckc, self.C_Z, n=2, dtype=self.dtype)
                             for i in range(2)]
        self.pairformer_stack = [PairformerBlock(self.scope(f"pairformer_stack.{i}"), ckc,
                                 self.C_S, self.C_Z, self.N_HEAD, dtype=self.dtype,
                                 fp32_residual=fp32_residual)
                                for i in range(self.N_PAIRFORMER)]
        # pure constants of (dtype) / (batch, tokens, dtype); see _onehot_dev and _zeros_dev
        self._const = {}
        # one design's process_z invariants, released when Z_init_II changes; see
        # _process_z_invariant. Not in _const: these are O(I^2) and design-scoped.
        self._zinv = None

    def _batched(self, x_dev, batch):
        """Replicate a batch-1 device tensor over the batch dim.

        The pair stack's Z_init_II is invariant across designs, so the only reason it
        needs a batch dim is the concat below. Doing that on device costs 0.67 ms for
        the [8,250,250,128] bf16 form (384 GB/s, i.e. bandwidth-bound) against 18.5 ms
        of host expand plus a 77.3 ms upload for the same bytes -- 144x, and a pure
        copy, so bit-exact.
        """
        if batch == 1 or x_dev.shape[0] == batch:
            return x_dev
        return ttnn.concat([x_dev] * batch, dim=0)

    def _onehot_dev(self, bins, batch, I):
        """One-hot the [B,I,I] int32 bin indices on device, by gathering identity rows.

        Exact: the gathered values are 0.0 and 1.0, both representable in every dtype
        this model uses, so the result is elementwise-identical to uploading the host
        one-hot (verify_distogram_onehot_parity.py).
        """
        dev, dt = self.device, self.dtype
        eye = self._const.get(("eye", dt))
        if eye is None:
            eye = _tt(torch.eye(self.N_BINS), dev, dt)
            self._const[("eye", dt)] = eye
        idx = _tt_idx(bins, dev)
        oh = ttnn.embedding(idx, eye, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.reshape(oh, (batch, I, I, self.N_BINS))
        return ttnn.to_layout(oh, ttnn.TILE_LAYOUT)

    COMBINED_ONEHOT_W = 160     # 65 + 65 real columns, padded to 5 tiles

    def _combined_onehot_dev(self, bins, bins_self, batch, I):
        """The distogram and self-conditioning one-hots side by side, padded to a tile multiple.

        Columns 0-64 are the distogram one-hot, 65-129 the self-conditioning one (all zero on the
        first recycle, where bins_self is None), 130-159 zero. Gathered rows of a constant table,
        so every value is 0.0 or 1.0 and the result is elementwise what the two separate one-hots
        produce -- the same argument _onehot_dev makes."""
        dev, dt, n = self.device, self.dtype, self.N_BINS
        w = self.COMBINED_ONEHOT_W
        key = ("comb", bins_self is None, dt)
        tab = self._const.get(key)
        if tab is None:
            ar = torch.arange(n)
            if bins_self is None:
                t = torch.zeros(n, w)
                t[ar, ar] = 1.0
            else:
                t = torch.zeros(n * n, w)
                row = ar.repeat_interleave(n) * n + ar.repeat(n)
                t[row, ar.repeat_interleave(n)] = 1.0
                t[row, n + ar.repeat(n)] = 1.0
            tab = _tt(t, dev, dt)
            self._const[key] = tab
        idx = _tt_idx(bins if bins_self is None else bins * n + bins_self, dev)
        oh = ttnn.embedding(idx, tab, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        oh = ttnn.reshape(oh, (batch, I, I, w))
        return ttnn.to_layout(oh, ttnn.TILE_LAYOUT)

    def _zeros_dev(self, batch, I):
        """The all-zero self-conditioning distogram of the first recycle.

        A pure constant of (batch, I, dtype), so it is allocated once and held, exactly
        like the attention-mask template (p10): at B=8/I=250 uploading it is a 48.9 ms
        transfer of 65 MB of zeros, every step.
        """
        key = ("zeros", batch, I, self.dtype)
        entry = self._const.get(key)
        if entry is None:
            entry = ttnn.zeros((batch, I, I, self.N_BINS), dtype=self.dtype,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
            self._const[key] = entry
        return entry

    def _process_z_weights(self):
        """process_z's two weights back on host, as the bf16 values the device actually holds.

        Read from the uploaded tensors rather than the checkpoint, so the collapsed path stays
        correct under the shared tiled-weight cache, which keeps no host weights. 132 KB, read
        once per module.
        """
        ent = self._const.get("zw")
        if ent is None:
            ent = (ttnn.to_torch(self.process_z_n).float().reshape(-1),
                   ttnn.to_torch(self.process_z_w).float())
            self._const["zw"] = ent
        return ent

    def _process_z_table(self, with_self):
        """The one-hot half of process_z's linear, as a constant lookup table.

        zcat is [z(128) | e_bd(65) | e_bs(65)], so its one-hot columns contribute exactly
        w_n[128+bd]*W[128+bd] (+ w_n[193+bs]*W[193+bs]) to the linear -- one row of a [65,128]
        table, or [65*65,128] (1.1 MB) once the self-conditioning bins exist. Read back from the
        uploaded weights, not the checkpoint: that holds the same bf16 values the shipped chain
        multiplies, and it works under the shared tiled-weight cache, which keeps no host weights.
        """
        key = ("ztab", with_self, self.dtype)
        tab = self._const.get(key)
        if tab is not None:
            return tab
        n, c = self.N_BINS, self.C_Z
        wn, ww = self._process_z_weights()
        t = wn[c:c + n, None] * ww[c:c + n]
        if with_self:
            ts = wn[c + n:c + 2 * n, None] * ww[c + n:c + 2 * n]
            t = (t[:, None, :] + ts[None, :, :]).reshape(n * n, c)
        tab = _tt(t.contiguous(), self.device, self.dtype)
        self._const[key] = tab
        return tab

    def _process_z_invariant(self, Z_init_II, n_ones, batch):
        """(inv, Ainv): the halves of process_z that do not depend on the timestep.

        sum(zcat^2) is sum(z^2) + n_ones, because the one-hot columns contribute one 1.0 per
        one-hot and nothing else. z is Z_init_II, fixed for the whole design, so the rms scale is
        a single tensor for all 200 steps -- and so is the z half of the linear, since

            linear(rms_norm(zcat)) = inv * ((z * w_n_z) @ W_z)  +  inv * T[bin]

        The sum of squares runs in fp32, matching what rms_norm does internally. Held for one
        design at a time and released when Z_init_II changes, because Ainv is O(I^2).
        """
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        dkey = (dev.id(), Z_init_II.data_ptr(), tuple(Z_init_II.shape), tuple(Z_init_II.stride()),
                str(Z_init_II.dtype), Z_init_II._version, batch)
        held = self._zinv
        if held is None or held[0] != dkey:
            for pair in (held[1].values() if held is not None else ()):
                for t in pair:
                    if t.is_allocated():
                        ttnn.deallocate(t)
            held = self._zinv = (dkey, {})
        ent = held[1].get(n_ones)
        if ent is not None and all(t.is_allocated() for t in ent):
            return ent
        c, w = self.C_Z, 2 * self.N_BINS + self.C_Z
        cached = _tt_cached(Z_init_II, dev, dt)
        z = self._batched(cached, batch)
        wn, ww = self._process_z_weights()
        gz = _tt(wn[:c].reshape(1, 1, 1, c), dev, dt)
        wz = _tt(ww[:c].contiguous(), dev, dt)
        tmp = []
        zf = ttnn.typecast(z, ttnn.float32)
        sq = ttnn.multiply(zf, zf)
        ss = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=ckc)
        shifted = ttnn.add(ss, float(n_ones))
        ms = ttnn.multiply(shifted, 1.0 / w)
        eps = ttnn.add(ms, 1e-6)
        rs = ttnn.rsqrt(eps)
        inv = ttnn.typecast(rs, dt)
        zs = ttnn.multiply(z, gz)
        a = ttnn.linear(zs, wz, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        ainv = ttnn.multiply(a, inv)
        tmp += [zf, sq, ss, shifted, ms, eps, rs, zs, a, gz, wz]
        if z is not cached:                     # _batched copies only when batch > 1
            tmp.append(z)
        for t in tmp:
            if t.is_allocated():
                ttnn.deallocate(t)
        held[1][n_ones] = ent = (inv, ainv)
        return ent

    def _process_z_collapsed(self, Z_init_II, bins, bins_self, batch, I, rows=None):
        """process_z in three device ops instead of six.

        The shipped route builds a [B,I,I,288] concat and a [B,I,I,258] slice of it so rms_norm
        can average over 258 columns, 130 of which are a one-hot: 15.14 ms/call at the page
        fixture's [1,685,685,*], 29.93 ms/step over the two recycles. This is 5.03
        (perf/p89/process_z.json). Not bit-exact -- one fp32 accumulation becomes two
        bf16-rounded halves, ~1 bf16 ulp -- and graded on the normal bar; RFD3_PROCESS_Z_COLLAPSE=0
        restores the six-op route.

        `rows` (a pair memo row list, _PZ_ROWS): gather and `ainv + em * inv` in one program, on those row tiles only.
        """
        dev, n = self.device, self.N_BINS
        inv, ainv = self._process_z_invariant(Z_init_II, 1 if bins_self is None else 2, batch)
        tab = self._process_z_table(bins_self is not None)
        flat = bins if bins_self is None else bins * n + bins_self
        if _PZ_GATHER:
            key = ("ztab_rm", bins_self is not None, self.dtype)
            rm = self._const.get(key)
            if rm is None:
                rm = ttnn.to_layout(tab, ttnn.ROW_MAJOR_LAYOUT)
                if _PZ_SPREAD:
                    rm, one = _gather_tiled.spread(rm), rm
                    ttnn.deallocate(one)
                self._const[key] = rm
            idx = ttnn.from_torch(_gather_tiled.pad_index(flat, batch * I, I), layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=dev, dtype=ttnn.uint32)
            em = _gather_tiled.gather_tiled(idx, rm, batch, I,
                                            copies=_gather_tiled.SPREAD_COPIES if _PZ_SPREAD else 1,
                                            fma=None if rows is None else (ainv, inv), rows=rows)
            if rows is not None:
                ttnn.deallocate(idx)
                return em
        else:
            idx = _tt_idx(flat, dev)
            em = ttnn.embedding(idx, tab, layout=ttnn.ROW_MAJOR_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            em = ttnn.to_layout(ttnn.reshape(em, (batch, I, I, self.C_Z)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        if _pz_addcmul_on():
            out = _EF.mask_add(ainv, em, inv)
            ttnn.deallocate(em)
            return out
        scaled = ttnn.multiply(em, inv)
        ttnn.deallocate(em)
        out = ttnn.add(scaled, ainv)
        ttnn.deallocate(scaled)
        return out

    def _pair_memo(self, shape, dtype, bins, bins_self):
        """(Z3, Z4, rows) for this step's pair path under RFD3_PAIR_MEMO, or None to run it in full. Z3 / Z4 are the
        recycle slot's z after transition_2 + block 0 and after block 1, kept on the card between steps; rows lists
        the row tiles holding a pair whose bin key changed since the slot last ran (every row tile the first time).
        The slot belongs to the design _process_z_invariant holds (self._zinv), and goes with it. `shape` is z's logical
        shape [B, I, I, C]."""
        B, I, J, C = (int(d) for d in shape)
        pshape = [B, I, -(-J // 32) * 32, -(-C // 32) * 32]
        nbytes = pshape[0] * pshape[1] * pshape[2] * pshape[3]
        if 4 * 2 * nbytes > _PAIR_MEMO_MB << 20 or dtype != ttnn.bfloat16:
            return None
        slot = int(bins_self is not None)
        key = bins if bins_self is None else bins * self.N_BINS + bins_self      # [B, I, I] int, as process_z's
        dkey = self._zinv[0]
        held = self._memo if getattr(self, "_memo", None) is not None and self._memo[0] == dkey else None
        if held is None:
            self._release_memo()
            held = self._memo = (dkey, {})
        ent = held[1].get(slot)
        jt = pshape[2] // 32
        if ent is None or not all(t.is_allocated() for t in ent[:2]):
            bufs = [ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, self.device,
                                                   ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
            rows = torch.arange(B * I * jt)
        else:
            bufs, prev = ent[:2], ent[2]
            ch = torch.zeros(B, I, jt * 32, dtype=torch.bool)
            ch[..., :key.shape[-1]] = key != prev
            rows = ch.view(B * I, jt, 32).any(-1).flatten().nonzero().flatten()
        held[1][slot] = (bufs[0], bufs[1], key.clone())
        PAIRMEMO[0] += int(rows.numel())
        PAIRMEMO[1] += B * I * jt
        return bufs[0], bufs[1], _WS.row_list(self.device, pshape, rows)

    def _release_memo(self):
        for ent in (self._memo[1].values() if getattr(self, "_memo", None) is not None else ()):
            for t in ent[:2]:
                if t.is_allocated():
                    ttnn.deallocate(t)
        self._memo = None

    def owns(self, z):
        """True when `z` is a pair memo buffer, which the caller must not deallocate."""
        m = getattr(self, "_memo", None)
        return m is not None and any(z is t for ent in m[1].values() for t in ent[:2])

    def _pairformer_memo(self, s, z0, Z3, Z4, rows, full_z=None):
        """run_device's pair path on `rows` only: Z3 = transition_2 then block 0's z_transition of z0, Z4 = Z3 plus
        block 1's, each row tile bit for bit what the full path writes. Falls back to the full path (and drops the
        slot) if a transition declines its rows; `full_z` then recomputes z0 when it holds `rows` only (_PZ_ROWS)."""
        steps = [(tr, src) for tr, src in zip(self.transition_2, (z0, Z3))]
        for tr, src in steps:
            if tr.add_rows(src, Z3, rows) is None:
                ttnn.deallocate(rows[0])
                if full_z is not None:
                    ttnn.deallocate(z0)
                    z0 = full_z()
                return self._memo_fallback(s, z0)
        ttnn.deallocate(z0)
        for blk, (src, dst) in zip(self.pairformer_stack, ((Z3, Z3), (Z3, Z4))):
            def upd(zt, src=src, dst=dst):
                out = zt.add_rows(src, dst, rows)
                assert out is not None, "pair memo: a Pairformer z_transition declined after transition_2 ran"
                return out
            s, _ = blk(s, None, z_update=upd)
        ttnn.deallocate(rows[0])
        return s, Z4

    def _memo_fallback(self, s, z0):
        self._release_memo()
        global _PAIR_MEMO
        _PAIR_MEMO = False
        for tr in self.transition_2:
            z0 = tr.add_into(z0)
        for blk in self.pairformer_stack:
            s, z0 = blk(s, z0)
        return s, z0

    def __call__(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None):
        """R_L_ca: [B, I, 3] (scaled C-alpha positions), S_init_I: [B, I, c_s],
        Z_init_II: [I, I, c_z] or [1, I, I, c_z] (batch 1 -- replicated on device),
        D_II_self: int32 [B, I, I] distogram bin indices or None.
        Returns (S_I [B,I,c_s], Z_II [B,I,I,c_z]) on host.

        The host-boundary wrapper the component parity scripts call. The per-step path
        uses run_device instead and keeps z on the card -- see run_device."""
        s, z = self.run_device(R_L_ca, S_init_I, Z_init_II, D_II_self=D_II_self)
        return ttnn.to_torch(s).float(), ttnn.to_torch(z).float()

    def bins(self, R_L_ca):
        return _scaled_distogram_bins(R_L_ca, sigma_data=self.sigma_data, n_bins=self.N_BINS)

    def run_device(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None, bins=None):
        """Same computation as __call__, returning the ttnn tensors instead of host copies.
        `bins` is self.bins(R_L_ca) when the caller already has it: every recycle of a step
        sees the same coordinates, and the host distogram is ~12 ms a call on one thread.

        z is [B,I,I,c_z]: O(I^2) and by far the largest tensor the step moves. Handing it
        to the DiT as a device tensor removes one D->H untilize plus one H->D upload per
        recycle, four crossings of an I^2 tensor per step."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        f32 = self.fp32_residual
        B, I = R_L_ca.shape[0], R_L_ca.shape[1]
        if not isinstance(S_init_I, ttnn.Tensor) and S_init_I.ndim == 2:
            S_init_I = S_init_I.unsqueeze(0).expand(B, -1, -1).contiguous()
        s = _dev(S_init_I, dev, dt)
        if f32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
        for tr in self.transition_1:
            sc = ttnn.typecast(s, dt, memory_config=s.memory_config()) if f32 else s
            upd = tr(sc)
            s = ttnn.add(s, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config())) if f32 \
                else ttnn.add(s, upd)
        if bins is None:
            bins = self.bins(R_L_ca)
        if Z_init_II.ndim == 3:
            Z_init_II = Z_init_II.unsqueeze(0)
        if _PROCESS_Z_COLLAPSE:
            PZSTATS[0] += 1
            memo = None
            if _PAIR_MEMO and not f32:
                self._process_z_invariant(Z_init_II, 1 if D_II_self is None else 2, B)   # the memo keys on its design
                memo = self._pair_memo([B, I, I, self.C_Z], self.dtype, bins, D_II_self)
            part = memo is not None and _PZ_ROWS and _PZ_GATHER
            z = self._process_z_collapsed(Z_init_II, bins, D_II_self, B, I, rows=memo[2] if part else None)
            if memo is not None:
                full = (lambda: self._process_z_collapsed(Z_init_II, bins, D_II_self, B, I)) if part else None
                return self._pairformer_memo(s, z, *memo, full_z=full)
        else:
            PZSTATS[1] += 1
            z = self._batched(_tt_cached(Z_init_II, dev, dt), B)
            w = 2 * self.N_BINS + self.C_Z
            if _CONCAT_ALIGNED:
                dself = self._combined_onehot_dev(bins, D_II_self, B, I)
                wide = ttnn.concat([z, dself], dim=-1)    # [B,I,I,288], both pieces tile-aligned
                ttnn.deallocate(dself)
                zcat = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, w])
                ttnn.deallocate(wide)
            else:
                d_dev = self._onehot_dev(bins, B, I)
                self_dev = (self._zeros_dev(B, I) if D_II_self is None
                            else self._onehot_dev(D_II_self, B, I))
                zcat = ttnn.concat([z, d_dev, self_dev], dim=-1)  # [B,I,I,258]
            z = ttnn.rms_norm(zcat, weight=self.process_z_n, epsilon=1e-6, compute_kernel_config=ckc)
            z = ttnn.linear(z, self.process_z_w, compute_kernel_config=ckc, dtype=dt,
                            core_grid=CORE_GRID_MAIN)
            ttnn.deallocate(zcat)
        if f32:
            z = ttnn.typecast(z, ttnn.float32, memory_config=z.memory_config())
        for tr in self.transition_2:
            if not f32:
                z = tr.add_into(z)
                continue
            upd = tr(ttnn.typecast(z, dt, memory_config=z.memory_config()))
            z = ttnn.add(z, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config()))
        for blk in self.pairformer_stack:
            s, z = blk(s, z)
        return s, z


def dit_hoist_weights(weights, n_block):
    """_DIT_SHOIST's wide weights from the torch state dict: (gain W [c_s, 2 n_block c], gain b [1, ...], shift W,
    gate W, gate b). Columns (2i + j) x c hold block i's attention (j = 0) or transition (j = 1) term; ln_s is folded
    into the gain and shift rows in float64, since rms_norm(s, ln_s) @ W = rms_norm(s) @ (ln_s[:, None] * W)."""
    sc_w, sc_b, sb_w, g_w, g_b = [], [], [], [], []
    for i in range(n_block):
        for ada, out in (("attention_pair_bias.ada_ln_1.", "attention_pair_bias."),
                         ("transition_block.ada_ln.", "transition_block.")):
            w = lambda k: weights[f"blocks.{i}.{k}"].double()  # noqa: E731
            ln = w(ada + "ln_s.weight")[:, None]
            sc_w.append(w(ada + "to_gain.0.weight").t() * ln)
            sc_b.append(w(ada + "to_gain.0.bias")[None])
            sb_w.append(w(ada + "to_bias.weight").t() * ln)
            g_w.append(w(out + "linear_output_project.0.weight").t())
            g_b.append(w(out + "linear_output_project.0.bias")[None])
    return tuple(torch.cat(x, -1) for x in (sc_w, sc_b, sb_w, g_w, g_b))


class LocalTokenTransformer(Module):
    """RFD3 18-block token DiT. Each block is the dense-additive-mask
    StructureLocalAtomTransformerBlock (conditioned AttentionPairBias + ConditionedTransition)
    at c_token=768, c_s=384, c_tokenpair=128, n_head=16, head_dim=48."""

    C_TOKEN, C_S, C_PAIR, N_HEAD, N_BLOCK = 768, 384, 128, 16, 18

    def __init__(self, state_dict, ckc, n_block=18, dtype=None, fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.blocks = [RFD3AtomBlock(self.scope(f"blocks.{i}"), ckc,
                        c_a=self.C_TOKEN, c_s=self.C_S, c_pair=self.C_PAIR, n_head=self.N_HEAD,
                        dtype=self.dtype, fp32_residual=fp32_residual)
                       for i in range(n_block)]
        for i, block in enumerate(self.blocks):
            block.idx = i
        self._hoist_w = None            # _DIT_SHOIST's wide weights, built on the first call that takes it

        # One [c_pair, 32 * n_block] weight holding every block's pair-bias projection, block i
        # in columns [32*i, 32*i + N_HEAD) and zeros in the rest. Built straight from the torch
        # weights rather than through torch_to_tt, which advances the shared tiled-weight cache
        # counter and would shift every later weight's cache key.
        self.b_w_fused = None
        self._b_w_packed = None         # [c_pair, N_HEAD * n_block], built on first _dit_sdpa_masks
        if _PAIRBIAS_FUSED:
            cols = torch.zeros(self.C_PAIR, _PAIRBIAS_SLOT * n_block)
            for i in range(n_block):
                w = self.weights[f"blocks.{i}.attention_pair_bias.to_b.weight"].t()
                cols[:, _PAIRBIAS_SLOT * i:_PAIRBIAS_SLOT * i + self.N_HEAD] = w.float()
            self.b_w_fused = ttnn.from_torch(cols, layout=ttnn.TILE_LAYOUT, device=self.device,
                                             dtype=self.dtype)

    def _dit_sdpa_masks(self, z, additive_mask, n_key):
        """Every block's SDPA mask from one projection, yielded as [B, N_HEAD, n_key, n_key] per
        block. The ttnn path runs the per-block elementwise ops once over the stack [B, n_block *
        N_HEAD, n_key, n_key] and slices it, so the DiT pays one permute, scale, add and pad pair per
        call instead of 18 (each block's permute alone was 0.39 ms); dit_pair_bias writes each
        block's mask directly."""
        if self._b_w_packed is None:
            cols = torch.cat([self.weights[f"blocks.{i}.attention_pair_bias.to_b.weight"].t().float()
                              for i in range(len(self.blocks))], dim=1)
            self._b_w_packed = ttnn.from_torch(cols, layout=ttnn.TILE_LAYOUT, device=self.device,
                                               dtype=self.dtype)
        if _DIT_DPB:
            if getattr(self, "_b_wt_packed", None) is None:
                wt = torch.cat([self.weights[f"blocks.{i}.attention_pair_bias.to_b.weight"].float()
                                for i in range(len(self.blocks))], dim=0)
                self._b_wt_packed = ttnn.from_torch(wt.reshape(1, 1, *wt.shape), layout=ttnn.TILE_LAYOUT,
                                                    device=self.device, dtype=self.dtype)
            mask = pad_axis(pad_axis(additive_mask, n_key, 3, -1e4), n_key, 2, -1e4)
            if _DPB.eligible(z, self._b_wt_packed, mask):
                out = _DPB.dit_pair_bias(z, self._b_wt_packed, mask, self.blocks[0].head_dim ** 0.5,
                                         self.compute_kernel_config.math_fidelity, n_out=len(self.blocks))
            else:
                out = None
            if mask is not additive_mask:
                ttnn.deallocate(mask)
            if out is not None:
                yield from out
                return
        pb = _tuned_linear(z, self._b_w_packed, ckc=self.compute_kernel_config, dtype=self.dtype,
                           core_grid=CORE_GRID_MAIN)
        pb_t = ttnn.permute(pb, (0, 3, 1, 2))
        ttnn.deallocate(pb)
        masks = _sdpa_dense_mask(pb_t, additive_mask, self.blocks[0].head_dim, n_key)
        ttnn.deallocate(pb_t)
        # Every design in the batch keeps its own pair bias: slicing batch [0, 1) here gave all of
        # them design 0's, and 7 of 8 --fast designs at batch 8 came out clashing.
        B, H = int(z.shape[0]), self.N_HEAD
        for i in range(len(self.blocks)):   # one slice live at a time, as before
            yield ttnn.slice(masks, [0, H * i, 0, 0], [B, H * (i + 1), n_key, n_key])
        ttnn.deallocate(masks)

    def _hoist(self, s):
        """_DIT_SHOIST: every block's c-only terms off one call's `s` as three wide [L, 2 x 18 x 768] tensors:
        sigmoid adaLN gains, adaLN shifts (both off rms_norm(s), each block's ln_s folded into its weights) and
        sigmoid output gates (off s). Block i's attention terms are columns 2i x 768, its transition's 2i + 1."""
        if self._hoist_w is None:
            self._hoist_w = tuple(ttnn.from_torch(t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=self.device,
                                                  dtype=self.dtype)
                                  for t in dit_hoist_weights(self.weights, len(self.blocks)))
        sc_w, sc_b, sb_w, g_w, g_b = self._hoist_w
        ckc = self.compute_kernel_config
        lin = lambda x, w, b=None, act=None: ttnn.linear(x, w, bias=b, activation=act, compute_kernel_config=ckc,  # noqa
                                                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        s_n = ttnn.rms_norm(s, epsilon=1e-6, compute_kernel_config=ckc)
        sig = mm_sigmoid(ckc)
        out = (lin(s_n, sc_w, sc_b, sig), lin(s_n, sb_w), lin(s, g_w, g_b, sig))
        ttnn.deallocate(s_n)
        return out

    def run_device(self, a, s, z, additive_mask):
        n_key = align_tile(int(a.shape[1]))
        B = int(z.shape[0])
        if (_DIT_SDPA and self.dtype == ttnn.bfloat16
                and 2 * B * self.N_HEAD * len(self.blocks) * n_key * n_key <= _DIT_SDPA_STACK_BYTES):
            wide = 3 * 2 * len(self.blocks) * self.C_TOKEN * int(s.shape[0]) * int(s.shape[1]) * 2
            hoist = self._hoist(s) if _DIT_SHOIST and wide <= _DIT_SHOIST_BYTES else None
            for m, block in zip(self._dit_sdpa_masks(z, additive_mask, n_key), self.blocks):
                a = block(a, s, z, additive_mask, sdpa_mask=m, hoist=hoist)
            for t in hoist or ():
                ttnn.deallocate(t)
            return a
        fused, end = None, None
        if self.b_w_fused is not None:
            fused = _tuned_linear(z, self.b_w_fused, ckc=self.compute_kernel_config,
                                  dtype=self.dtype, core_grid=CORE_GRID_MAIN)
            end = [int(fused.shape[i]) for i in range(3)]
        for i, block in enumerate(self.blocks):
            pb = None
            if fused is not None:
                lo = _PAIRBIAS_SLOT * i
                pb = ttnn.slice(fused, [0, 0, 0, lo], end + [lo + self.N_HEAD])
            a = block(a, s, z, additive_mask, pair_bias=pb)
        if fused is not None:
            ttnn.deallocate(fused)
        return a

    def __call__(self, a_host, s_host, z, indices, resident=False):
        """z (and a, s) may be host tensors (the parity-script path) or already-resident ttnn
        tensors handed straight over by DiffusionTokenEncoder.run_device; `resident` returns
        the device tensor."""
        dev, dt = self.device, self.dtype
        a = _dev(a_host, dev, dt)
        s = _dev(s_host, dev, dt)
        if isinstance(z, ttnn.Tensor):
            if z.dtype != dt:
                z = ttnn.typecast(z, dt)
        else:
            z = _tt(z.unsqueeze(0) if z.ndim == 2 else z, dev, dt)
        mask = _tt(_dense_attention_mask(indices), dev, dt)
        out = self.run_device(a, s, z, mask)
        return out if resident else ttnn.to_torch(out).float()


def _default_compute_kernel_config():
    dev = get_device()
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi2 if _HIFI2 else ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def build_token_initializer(state_dict, compute_kernel_config=None, dtype=None):
    """Construct the ttnn TokenInitializer from a flat `token_initializer.*` state dict
    (prefix already stripped) + a compute_kernel_config. Mirrors the construction order
    used by the torch reference so weight keys line up 1:1."""
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()
    return TokenInitializer(state_dict, compute_kernel_config, dtype=dtype)


# --- Host-side attention-index builder (vendored from foundry block_utils) ---
def _build_index_mask(tok_idx, n_seq_neighbours, k_max, chain_id=None, base_mask=None):
    """Build the full-token local mask without atom-grid counting and sorting."""
    device = tok_idx.device
    tok_idx = tok_idx.long()
    length = tok_idx.shape[0]
    k_max = min(k_max, length)
    n_tokens = int(tok_idx.max().item()) + 1
    positions = torch.arange(length, device=device)
    first = torch.full((n_tokens,), length, dtype=torch.long, device=device)
    last = torch.full((n_tokens,), -1, dtype=torch.long, device=device)
    first.scatter_reduce_(0, tok_idx, positions, reduce="amin", include_self=True)
    last.scatter_reduce_(0, tok_idx, positions, reduce="amax", include_self=True)

    token_ids = torch.arange(n_tokens, device=device)
    allowed = (
        (token_ids[:, None] - token_ids[None, :]).abs() <= n_seq_neighbours
    )
    max_atom_distance = torch.maximum(
        (first[:, None] - last[None, :]).abs(),
        (last[:, None] - first[None, :]).abs(),
    )
    allowed &= max_atom_distance <= (k_max // 2)
    if chain_id is not None:
        token_chain = chain_id[first]
        allowed &= token_chain[:, None] == token_chain[None, :]
    if base_mask is not None:
        if base_mask.shape == (n_tokens, n_tokens):
            allowed &= base_mask
        else:
            allowed &= base_mask[first[:, None], first[None, :]]
    return allowed[tok_idx[:, None], tok_idx[None, :]]


def _extend_with_neighbours(mask, seq_idx, D_LL, k, inplace=False):
    """Fill the sequence-neighbour index list out to k with the nearest non-neighbours.

    ``mask`` and ``seq_idx`` are the coordinate-independent half and come from
    _attention_index_prefix; only the distance topk below reads the coordinates.
    """
    if D_LL.ndim == 2:
        D_LL = D_LL.unsqueeze(0)
    _, length, _ = D_LL.shape
    k = min(k, length)
    inf = torch.tensor(float("inf"), dtype=D_LL.dtype, device=D_LL.device)
    # where() allocates and writes a second (D,L,L) -- 361 MB at 3359 atoms and D=8, which
    # is 42.5 ms of the 211.5 ms this build costs. masked_fill_ substitutes exactly the same
    # values into the positions where() would have changed, so it is exact; callers pass
    # inplace only where nothing reads D_LL afterwards.
    masked_distances = D_LL.masked_fill_(mask, inf) if inplace else torch.where(mask, inf, D_LL)
    fill = torch.topk(masked_distances, k, dim=-1, largest=False).indices.flip(
        dims=[-1]
    )
    idx = torch.where((seq_idx == length).expand_as(fill), fill, seq_idx.expand_as(fill))
    return idx.long()


def _neighbours_row_blocked(mask, seq_idx, x, k, block=None):
    """`_extend_with_neighbours` over `torch.cdist(x, x)`, one row block at a time.

    Same arithmetic as the unblocked chain, but the [L, L] fp32 distance matrix is never
    materialised: cdist writes an [R, L] slab, the mask and the topk consume it while it is
    still in cache, and only the [R, k] indices survive. See the note at _ATTN_ROW_BLOCK.
    """
    length = x.shape[1]
    k = min(k, length)
    block = block or _ATTN_ROW_BLOCK
    inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    out = torch.empty(1, length, k, dtype=torch.long, device=x.device)
    for r0 in range(0, length, block):
        r1 = min(r0 + block, length)
        d = torch.cdist(x[:, r0:r1], x, p=2).masked_fill_(mask[r0:r1], inf)
        fill = torch.topk(d, k, dim=-1, largest=False).indices.flip(dims=[-1])
        s = seq_idx[r0:r1].unsqueeze(0)
        out[:, r0:r1] = torch.where((s == length).expand_as(fill), fill, s.expand_as(fill))
    return out


def _neighbours_kdtree(mask, seq_idx, x, k, workers=None, skip=None):
    """`_extend_with_neighbours` over a k-d tree query, already sorted (see _ATTN_KDTREE).

    Row r keeps its sequence entries (seq_idx < L) and adds the first k - n_seq atoms of its
    k-nearest list that lie outside mask[r]; the union has exactly k members, so sorting the
    two lists side by side and keeping k gives the row _create_attention_indices returns.
    Everything after the query is numpy, which starts no OpenMP team, so the build can run
    on a worker thread (_ATTN_THREAD).
    `skip` (r0, r1): the design's moving atoms, whose rows the card answers (_MoverIndex); they come back as L.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    length = x.shape[1]
    k = min(k, length)
    xd = x[0].detach().double().cpu().numpy()
    workers = workers or torch.get_num_threads()
    if _STATIC_KNN:
        from .staticknn import StaticKnn
        nn = _STATIC_KNN_CACHE.setdefault((length, k), StaticKnn()).query(xd, k, workers, skip is not None)
    else:
        _, nn = cKDTree(xd).query(xd, k=k, workers=workers)
        nn = nn.reshape(length, k)
    # A row's output depends only on its nn row (mask and seq_idx are fixed per design), and in a binder or motif
    # design most rows keep last step's neighbours, so only the rows whose nn changed are rebuilt. One tuple read and
    # one write: a concurrent caller sees a consistent previous step or none.
    prev = _NN_ROWS_CACHE.get((length, k))
    # skipped rows hold nn -1, so a later step that answers them itself sees them changed and fills them
    live = np.ones(length, bool)
    if skip is not None:
        live[skip[0]:skip[1]] = False
    if prev is not None and prev[0] is mask and prev[1] is seq_idx:
        rows = np.flatnonzero((nn != prev[2]).any(1) & live)
        vals = prev[3].copy()
        vals[rows] = _fill_rows(mask, seq_idx, nn, rows, length, k)
    else:
        vals = np.full((length, k), length, np.int32)
        rows = np.flatnonzero(live)
        vals[rows] = _fill_rows(mask, seq_idx, nn, rows, length, k)
    _NN_ROWS_CACHE[(length, k)] = (mask, seq_idx, nn, vals)
    return torch.from_numpy(vals.astype(np.int64)).unsqueeze(0)


def _fill_rows(mask, seq_idx, nn, rows, length, k):
    """Rows `rows` of _neighbours_kdtree's output, int32 [len(rows), k]."""
    import numpy as np

    nn, seq = nn[rows], seq_idx.numpy()[rows]
    r = np.arange(length)[rows]
    outside = ~mask.numpy()[r[:, None], nn]
    n_fill = k - (seq < length).sum(1, keepdims=True)
    keep = outside & (np.cumsum(outside, 1) <= n_fill)
    vals = np.concatenate([seq.astype(np.int32), np.where(keep, nn.astype(np.int32), length)], 1)
    vals.sort(1)
    return np.ascontiguousarray(vals[:, :k])


def _mask_and_seq_idx(tok_idx, n_seq_neighbours, k, chain, base_mask, length):
    mask = _build_index_mask(tok_idx, n_seq_neighbours, k, chain, base_mask).contiguous()
    rows = torch.arange(length, device=tok_idx.device).unsqueeze(0).expand(length, length)
    seq_idx = torch.where(mask, rows, length).topk(k, dim=1, largest=False, sorted=True).values
    return mask, seq_idx


_ATTN_INDEX_CACHE: dict = {}
_ATTN_INDEX_CACHE_MAX = 8


def _attention_index_prefix(f, tok_idx, n_keys, n_seq_neighbours):
    """Memoize the coordinate-independent half of _create_attention_indices.

    Everything built here reads only the token layout -- tok_idx, asym_id,
    unindexing_pair_mask -- and the two k values, all of which are fixed for a whole
    design. The sampler nevertheless rebuilt it on every one of ~200 diffusion steps,
    and it is O(L^2): 12.3% of the step at 3359 atoms, 7.0% at batch 8 (p19 step 0).
    Only cdist and the distance topk in _extend_with_neighbours genuinely move with the
    coordinates. Pure memoization of a deterministic function, so it is exact.

    Keyed on the identity of the design-level tensors, then re-validated against tok_idx
    by value, because the DiT passes a freshly built arange on every call: an id() key
    alone would always miss, and once a gc cycle reused an address it could hit stale.
    The entry holds its own references to what it keys on so those ids cannot be reused
    while the entry lives.
    """
    asym = f.get("asym_id")
    upm = f["unindexing_pair_mask"]
    L = len(tok_idx)
    key = (L, n_keys, n_seq_neighbours, id(asym), id(upm))
    hit = _ATTN_INDEX_CACHE.get(key)
    if hit is not None and hit[0] is asym and hit[1] is upm and torch.equal(hit[2], tok_idx):
        return hit[3]

    base_mask = ~upm
    k = min(n_keys, L)
    chain = asym[tok_idx] if asym is not None else None
    parts = {"k": k}
    if chain is not None and len(torch.unique(chain)) > 3:
        ki, kc = max(32, k // 4), k - max(32, k // 4)
        parts.update(ki=ki, kc=kc, chain=chain,
                     intra=_mask_and_seq_idx(tok_idx, n_seq_neighbours, kc, chain, base_mask, L),
                     atom_base_mask=base_mask[tok_idx[None, :], tok_idx[:, None]])
    else:
        parts["single"] = _mask_and_seq_idx(tok_idx, n_seq_neighbours, k, chain, base_mask, L)

    if len(_ATTN_INDEX_CACHE) >= _ATTN_INDEX_CACHE_MAX:
        _ATTN_INDEX_CACHE.pop(next(iter(_ATTN_INDEX_CACHE)))
    _ATTN_INDEX_CACHE[key] = (asym, upm, tok_idx, parts)
    return parts


def _attention_indices_future(f, X_L, tok_idx, n_keys, n_seq_neighbours, device=None):
    """Start the atom index build on a worker thread (see _ATTN_THREAD), or None to build inline.

    The coordinate-independent prefix is built (or hit) here on the calling thread, so the
    worker only runs the k-d tree path, which is scipy's threads plus numpy. With dknn on and a
    device the build is a card program instead (tt_bio.dknn), with the same .result(), as long as
    its per-core buffers fit L1 (about 18k atoms); larger designs keep the tree.
    """
    global _INDEX_POOL
    parts = _attention_index_prefix(f, tok_idx, n_keys, n_seq_neighbours)
    if "single" not in parts:
        return None
    if device is not None and _dknn_on():
        from .. import dknn
        mask, seq_idx = parts["single"]
        if dknn.fits(len(tok_idx), parts["k"]):
            static = _DKNN_STATIC_CACHE.setdefault((len(tok_idx), parts["k"]), dknn.Static()) if _DKNN_STATIC else None
            return dknn.Pending(device, X_L if X_L.ndim == 3 else X_L.unsqueeze(0), mask, seq_idx, parts["k"], static)
    if not (_ATTN_THREAD and _ATTN_KDTREE):
        return None
    if _INDEX_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _INDEX_POOL = ThreadPoolExecutor(1, thread_name_prefix="rfd3-attn-index")
    if (_DKNN_MOVERS and device is not None and _STATIC_KNN and len(tok_idx) >= _DKNN_MOVERS_MIN_L
            and (X_L.ndim == 2 or X_L.shape[0] == 1)):
        hybrid = _MoverIndex.launch(device, X_L.reshape(1, -1, 3), parts)
        if hybrid is not None:
            return hybrid
    return _INDEX_POOL.submit(_create_attention_indices, f, X_L, tok_idx, n_keys, n_seq_neighbours,
                              workers=torch.get_num_threads())


_INDEX_POOL = None


class _MoverIndex:
    """The atom index split by row: the moving atoms' rows on the card (tt_bio.dknn over their id range), every other
    row on the host thread (_neighbours_kdtree with those rows skipped). .result() as a Future's."""

    def __init__(self, read, host, rows, seq, k):
        self.read, self.host, self.rows, self.seq, self.k = read, host, rows, seq, k

    @classmethod
    def launch(cls, device, X_L, parts):
        import numpy as np
        from .. import dknn
        from .staticknn import StaticKnn

        mask, seq_idx = parts["single"]
        L, k = X_L.shape[1], min(parts["k"], X_L.shape[1])
        sk = _STATIC_KNN_CACHE.get((L, k))
        if sk is None or not dknn.fits(L, k):
            return None
        xd = X_L[0].detach().double().cpu().numpy()
        mid = sk.movers(xd)
        if mid is None or not len(mid) or mid[-1] - mid[0] + 1 != len(mid):
            return None
        rows = (int(mid[0]), int(mid[-1]) + 1)
        out = dknn.outside_neighbours(device, xd, mask, seq_idx, k, rows)
        host = _INDEX_POOL.submit(_neighbours_kdtree, mask, seq_idx, X_L, k, torch.get_num_threads(), rows)
        return cls(dknn.read_later(device, [out]), host, rows, seq_idx.cpu().numpy(), k)

    def result(self):
        from .. import dknn
        idx = self.host.result()
        r0, r1 = self.rows
        idx[0, r0:r1] = dknn.join(self.seq[r0:r1], self.read()[0].numpy(), len(self.seq), self.k)[0]
        return idx


def _create_attention_indices(f, X_L, tok_idx, n_keys, n_seq_neighbours, workers=None):
    device = X_L.device; L = len(tok_idx)
    parts = _attention_index_prefix(f, tok_idx, n_keys, n_seq_neighbours)
    if X_L.ndim == 2:
        X_L = X_L.unsqueeze(0)
    if "single" in parts:
        mask, seq_idx = parts["single"]
        # One design at a time. The designs are independent, so this is the same
        # arithmetic, but a (1,L,L) distance slice is 45 MB at 3359 atoms and stays
        # resident, where the batched form streams three (D,L,L) tensors of 361 MB each.
        # Measured bit-identical indices and 220.7 -> 152.5 ms at 3359 atoms D=8,
        # 142.5 -> 68.9 ms at 2702 (scripts/rfd3_port/p24_attn_indices_variants.py).
        if _ATTN_KDTREE:
            return torch.cat([_neighbours_kdtree(mask, seq_idx, x, parts["k"], workers)
                              for x in X_L.unsqueeze(1)], dim=0)
        idx = torch.cat([
            _neighbours_row_blocked(mask, seq_idx, x, parts["k"]) if _ATTN_ROW_BLOCK else
            _extend_with_neighbours(mask, seq_idx, torch.cdist(x, x, p=2), parts["k"],
                                    inplace=True)
            for x in X_L.unsqueeze(1)], dim=0)
    else:
        D_LL = torch.cdist(X_L, X_L, p=2)   # the inter-chain pass below reads it again
        ki, kc, chain = parts["ki"], parts["kc"], parts["chain"]
        mask, seq_idx = parts["intra"]
        intra = _extend_with_neighbours(mask, seq_idx, D_LL, kc)
        atom_base_mask = parts["atom_base_mask"]
        inter = torch.zeros(D_LL.shape[0], L, ki, dtype=torch.long, device=device)
        for b in range(D_LL.shape[0]):
            for c in torch.unique(chain):
                ci = chain[c]; other = (chain != ci) & atom_base_mask[c, :]
                oi = torch.where(other)[0]; ns = min(ki, len(oi))
                if ns > 0:
                    inter[b, c, :ns] = oi[torch.topk(D_LL[b, c, oi], ns, largest=False).indices]
        idx = torch.cat([intra, inter], dim=-1)
    return torch.sort(idx, dim=-1)[0].detach()


def _grouping_indices(tok_idx, batch, dev):
    valid = _build_valid_mask(tok_idx)
    length = tok_idx.numel()
    padded = torch.full(valid.shape, length, dtype=torch.int64)
    padded[valid] = torch.arange(length)
    pack = torch.cat([padded.reshape(-1) + b * (length + 1) for b in range(batch)])
    flat_valid = valid.flatten().nonzero(as_tuple=False).squeeze(1)
    unpack = torch.cat([flat_valid + b * valid.numel() for b in range(batch)])
    return valid, pack, unpack


def _pack_atoms_dev_core(q, idx_dev, valid):
    """Pure-device tail of the pack-by-embedding gather -- takes an ALREADY-uploaded
    uint32 idx tensor (see _grouping_buffers) so a cached idx never triggers a fresh
    device allocation on every call (p27: this was the eager alloc that hung the
    device when interleaved with an open encoder trace -- see RFD3DiffusionModule
    docstring on _grouping_buffers)."""
    batch, length, channels = q.shape
    orig_dt = q.dtype
    # ttnn.embedding requires bf16 weights; the gather is a pure reindex (exact), so
    # round-trip through bf16 only for the embedding op, then restore the compute dtype.
    q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    q = ttnn.pad(q, [[0, 0], [0, 1], [0, 0]], 0.0)
    q = ttnn.reshape(q, (batch * (length + 1), channels))
    if orig_dt != ttnn.bfloat16:
        q = ttnn.typecast(q, ttnn.bfloat16)
    packed = ttnn.embedding(idx_dev, q, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if orig_dt != ttnn.bfloat16:
        packed = ttnn.typecast(packed, orig_dt)
    packed = ttnn.reshape(packed, (batch, valid.shape[0], valid.shape[1], channels))
    return ttnn.to_layout(packed, ttnn.TILE_LAYOUT)


def _scatter_mean(emb, tok_idx, I):
    B, L, C = emb.shape
    out = torch.zeros(B, I, C, dtype=emb.dtype, device=emb.device)
    out.scatter_add_(-2, tok_idx.long().view(1, L, 1).expand(B, L, C).contiguous(), emb)
    cnt = torch.zeros(B, I, 1, dtype=emb.dtype, device=emb.device)
    cnt.scatter_add_(-2, tok_idx.long().view(1, L, 1).expand(B, L, 1).contiguous(),
                       torch.ones(B, L, 1, dtype=emb.dtype, device=emb.device))
    return out / cnt.clamp(min=1)


class RFD3DiffusionModule(Module):
    """RFD3 DiffusionModule (one resident denoise step) on ttnn. Composes the verified
    encoder/decoder/DiffusionTokenEncoder/DiT/sequence-head (host-boundary __call__ wrappers)
    with the new glue (process_a, downcast_c/q, process_r/c, process_time_, to_r_update,
    scale_positions). Heavy linears on device; scatter/fourier/scale/bucketize on host.
    Faithful to upstream RFD3_diffusion_module.py (f_pred=edm, n_recycle=2, n_attn_keys=128,
    n_attn_seq_neighbours=2; DiT n_keys=32, n_local_tokens=8)."""

    C_ATOM, C_ATOMPAIR, C_TOKEN, C_S, C_Z = 128, 16, 768, 384, 128
    C_T_EMBED, SIGMA_DATA, N_RECYCLE = 256, 16.0, 2
    N_ATTN_KEYS, N_ATTN_SEQ = 128, 2
    DIT_KEYS, DIT_SEQ = 32, 8

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        # Selective-fp32 boundary (per af3-diffusion-sampler-selective-fp32-boundary): the
        # diffusion SCORE MODEL (this DM, run every step+recycle) is the precision knob for
        # the low-noise trajectory tail; the step-invariant TokenInitializer stays bf16.
        # Opt-in via RFD3_DIT_FP32=1 so the default path keeps bf16 perf. The compute kernel
        # already accumulates in fp32 (fp32_dest_acc_en=True, HiFi4); this raises the STORAGE
        # dtype of the DM's matmuls/linears/norms to fp32 to stop bf16 rounding compounding
        # across the 18-block DiT stack. Measure before/after; keep default perf intact.
        if dtype is None and env_flag("RFD3_DIT_FP32", False):
            dtype = ttnn.float32
        self.dtype = dtype or ttnn.bfloat16
        dt = self.dtype
        # fp32-residual-stream lever (opt-in via RFD3_FP32_RESIDUAL=1): threads through the
        # pure RFD3AtomBlock stacks (DiT + encoder) AND the DiffusionTokenEncoder's 2-block
        # Pairformer stack (the last shallow residual path). Default off = bf16, bit-identical.
        self._dit_fp32_residual = env_flag("RFD3_FP32_RESIDUAL", False)
        self.process_r_w = self.torch_to_tt("process_r.weight", dtype=dt)
        self.to_r_n = self.torch_to_tt("to_r_update.0.weight", dtype=dt)
        self.to_r_w = self.torch_to_tt("to_r_update.1.weight", dtype=dt)
        self.process_c_n = self.torch_to_tt("process_c.0.weight", dtype=dt)
        self.process_c_w = self.torch_to_tt("process_c.1.weight", dtype=dt)
        self.process_a_w = self.torch_to_tt("process_a.linear.weight", dtype=dt)
        self.fourier_w = [self.weights["fourier_embedding.0.w"].float(),
                          self.weights["fourier_embedding.1.w"].float()]
        self.fourier_b = [self.weights["fourier_embedding.0.b"].float(),
                          self.weights["fourier_embedding.1.b"].float()]
        self.process_n_n = [self.torch_to_tt("process_n.0.0.weight", dtype=dt),
                            self.torch_to_tt("process_n.1.0.weight", dtype=dt)]
        self.process_n_w = [self.torch_to_tt("process_n.0.1.weight", dtype=dt),
                            self.torch_to_tt("process_n.1.1.weight", dtype=dt)]
        self.downcast_c = GatedCrossAttention(self.scope("downcast_c.gca"), ckc,
                                              c_query=self.C_S, c_kv=self.C_ATOM, c_model=self.C_ATOM, n_head=4, dtype=dt)
        self.downcast_q = GatedCrossAttention(self.scope("downcast_q.gca"), ckc,
                                              c_query=self.C_TOKEN, c_kv=self.C_ATOM, c_model=self.C_ATOM, n_head=4, dtype=dt)
        self.downcast_q_s_n = self.torch_to_tt("downcast_q.process_s.0.weight", dtype=dt)
        self.downcast_q_s_w = self.torch_to_tt("downcast_q.process_s.1.weight", dtype=dt)
        self.diffusion_token_encoder = DiffusionTokenEncoder(self.scope("diffusion_token_encoder"), ckc, dtype=dt,
                                                             fp32_residual=self._dit_fp32_residual)
        # fp32 residual stream across the pure RFD3AtomBlock stacks in the DM: the 18-block
        # DiT (deepest compounding path), the 3-block encoder, and the 2-block
        # DiffusionTokenEncoder (the last shallow residual path). The decoder is intentionally
        # excluded (its atom blocks are interleaved with a GatedCrossAttention and truncated by
        # embedding gathers between blocks, so an fp32 residual would leak into a GCA matmul
        # -> host fallback and would not compound anyway). Matmuls stay bf16 (Blackhole fp32
        # matmul is a host-fallback dead-end); only the residual sum is fp32 so bf16 storage
        # rounding does not compound across the block stacks. Opt-in via RFD3_FP32_RESIDUAL=1;
        # default off keeps the verified bf16 behavior.
        self.diffusion_transformer = LocalTokenTransformer(self.scope("diffusion_transformer"), ckc, dtype=dt,
                                                         fp32_residual=self._dit_fp32_residual)
        # The fp32-residual lever threads through the DiT and the encoder (the two pure
        # atom-block stacks); the decoder accepts the flag but ignores it (see its ctor).
        # Default off keeps the verified bf16 behavior.
        # Both RFD3 traces are withdrawn and refuse by name, so no flag can reach a capture.
        # The encoder trace was 74.5% slower than eager (p32) and hung a Wormhole chip in the
        # first eager to_torch after its replay (whglx card 17, 2026-09-23). The decoder trace
        # was 12.3% slower (p32, 3359 atoms: 246.3 -> 280.8 ms/step), deterministically moved a
        # 120-residue design 8.9 A CA-RMSD from eager because it kept the host pair gather the
        # eager path had replaced, and ran out of DRAM at 1280 residues where eager reaches 1536
        # (perf/mgx_trace_region, 2026-09-24).
        for flag, why in (("RFD3_TRACE_ENCODER", "was 74.5% slower than eager and hung a "
                           "Wormhole chip after its first replay"),
                          ("RFD3_TRACE_DECODER", "was 12.3% slower than eager, changed the "
                           "design, and ran out of memory at 1280 residues")):
            if env_flag(flag, False):
                raise ValueError(f"{flag} is withdrawn: the trace {why}. Unset it; RFD3 runs "
                                 f"eager.")
        self._grouping_cache = {}      # batch -> {"valid", "pack_idx_dev", ...}, shared by downcast_c/downcast_q
        self._grouping_owner = None    # (id(tok_idx), shape) the cached slots belong to
        self._downcast_c_memo = None   # (key, inputs, device result, host result): see _downcast_c
        self._row_masks = {}           # _free_rows_dev
        self.encoder = LocalAtomTransformer(self.scope("encoder"), ckc, n_blocks=3, dtype=dt,
                                            fp32_residual=self._dit_fp32_residual)
        self.decoder = CompactStreamingDecoder(self.scope("decoder"), ckc, dtype=dt,
                                               fp32_residual=self._dit_fp32_residual)
        # One sparse-QK cache for both: they are handed the same P_LL and the same
        # attn_indices every step, so sharing collapses three identical builds into
        # one (see _sparse_qk_inputs) and leaves one -1e4 scatter template alive
        # instead of two.
        self.decoder._mask_cache = self.encoder._mask_cache
        self.sequence_head = LinearSequenceHead(self.scope("sequence_head"), ckc, dtype=dt)

    def scale_positions_in(self, X, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]
        return X / torch.sqrt(t ** 2 + self.SIGMA_DATA ** 2)

    def scale_positions_out(self, R_upd, X, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]
        sd = self.SIGMA_DATA
        return (sd ** 2 / (sd ** 2 + t ** 2)) * X + (sd * t / (sd ** 2 + t ** 2) ** 0.5) * R_upd

    def _process_time(self, t_L, i):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        tt = 0.25 * torch.log(torch.clamp(t_L, min=1e-20) / self.SIGMA_DATA)
        emb = torch.cos(2 * math.pi * (tt[..., None] * self.fourier_w[i] + self.fourier_b[i]))
        emb = emb * (t_L > 0).float()[..., None]
        x = _tt(emb, dev, dt)
        x = ttnn.rms_norm(x, weight=self.process_n_n[i], epsilon=1e-6, compute_kernel_config=ckc)
        out = ttnn.linear(x, self.process_n_w[i], compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        return ttnn.to_torch(out).float()

    def _process_time_dev(self, t, motif, i):
        """_process_time without the host round trip. Every row of the time embedding is either
        the step's one fourier row or zero (motif atoms, t == 0), and rms_norm and the bias-free
        linear act per row and send 0 to 0, so one row is embedded and spread by the 0/1 mask."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        tt = 0.25 * torch.log(torch.clamp(t, min=1e-20) / self.SIGMA_DATA)[:, None, None]
        emb = torch.cos(2 * math.pi * (tt * self.fourier_w[i] + self.fourier_b[i]))
        x = _tt(emb * (t > 0).float()[:, None, None], dev, dt)                # [B, 1, 256]
        x = ttnn.rms_norm(x, weight=self.process_n_n[i], epsilon=1e-6, compute_kernel_config=ckc)
        x = ttnn.linear(x, self.process_n_w[i], compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        return ttnn.multiply(x, self._free_rows_dev(motif, int(x.shape[-1])))

    def _free_rows_dev(self, motif, width):
        """[1, N, width] device copy of ~motif as 0/1, uploaded once per design (the sampler
        threads the same `f` tensors through every step)."""
        key = (id(motif), tuple(motif.shape), width)
        ent = self._row_masks.get(key)
        if ent is None or ent[0] is not motif:
            m = (~motif).float().reshape(1, -1, 1).expand(1, motif.numel(), width).contiguous()
            ent = self._row_masks[key] = (motif, _tt(m, self.device, self.dtype))
        return ent[1]

    def _grouping_buffers(self, tok_idx, batch):
        """Cache the atom<->token grouping index (pack_idx_dev/valid) by tok_idx identity,
        shared by _downcast_c and _downcast_q (both group by the SAME atom_to_token_map).

        tok_idx is the SAME python object for an entire design's sampling loop (RFD3Sampler
        threads one `f` dict by reference through every step), exactly like
        CompactStreamingDecoder._design_buffers -- so caching by id(tok_idx) means the
        pack_idx_dev upload happens ONCE per design, not once per step.

        Before this cache, _downcast_q re-derived and re-uploaded pack_idx_dev on every call.
        """
        dev = self.device
        # One slot PER BATCH SIZE, not one slot total: _downcast_c is called with the
        # batch-1 S_I while the encoder downcast is called with the batch-D A_I, so a
        # single-slot cache missed on every call at D>1 and re-uploaded the index --
        # 8.7 ms per call at 3359 atoms, 31x what eight batch-1 calls cost (p11).
        owner = (id(tok_idx), tuple(tok_idx.shape))
        if self._grouping_owner != owner:
            self._grouping_owner, self._grouping_cache = owner, {}
        st = self._grouping_cache.get(batch)
        if st is None:
            valid, pack, _ = _grouping_indices(tok_idx, batch, dev)
            pack_idx_dev = _tt_idx(pack, dev)
            # Additive mask for the downcast GCAs (downcast_c AND downcast_q -- same
            # shape/content, mask is about which atom-slots are valid, independent of
            # which GCA's weights consume it): precomputed and persisted here, NOT passed
            # as a raw host `attn_mask` into run_device (which would upload it internally
            # on every call -- see _downcast_q_device).
            downcast_mask_dev = self.downcast_q._prepare_additive_mask(
                valid.unsqueeze(1), batch, valid.shape[0], 1, valid.shape[1])
            st = dict(valid=valid, pack_idx_dev=pack_idx_dev, downcast_mask_dev=downcast_mask_dev)
            self._grouping_cache[batch] = st
        return st["valid"], st["pack_idx_dev"], st["downcast_mask_dev"]

    def _downcast_c(self, C_L, S_I, tok_idx):
        """S_I + GCA(S_I, C_L). Both inputs are the token initializer's outputs, fixed for the
        design, so every step after the first returns the first step's result: the same device
        program on the same bytes. Keyed like _tt_cached (storage, shape, stride, version), with
        the inputs held so an address cannot be reused under a live key."""
        if C_L.ndim == 2: C_L = C_L.unsqueeze(0)
        if S_I.ndim == 2: S_I = S_I.unsqueeze(0)
        key = tuple((x.data_ptr(), tuple(x.shape), tuple(x.stride()), x._version) for x in (C_L, S_I, tok_idx))
        memo = self._downcast_c_memo
        if memo is None or memo[0] != key:
            dev_out = self._downcast_c_device(C_L, S_I, tok_idx)
            memo = self._downcast_c_memo = (key, (C_L, S_I, tok_idx), dev_out, ttnn.to_torch(dev_out).float())
        return memo[2] if _RESIDENT_STEP else memo[3].clone()

    def _downcast_c_device(self, C_L, S_I, tok_idx):
        dev, dt = self.device, self.dtype
        B, I, _ = S_I.shape
        valid, pack_idx_dev, mask_dev = self._grouping_buffers(tok_idx, B)
        c_g = _pack_atoms_dev_core(_tt_cached(C_L, dev, dt), pack_idx_dev, valid)
        S_I_dev = _tt_cached(S_I, dev, dt)  # uploaded once, reused below (was 2x, p23 perf)
        q = ttnn.unsqueeze(S_I_dev, 2)
        upd = ttnn.squeeze(self.downcast_c.run_device(q, c_g, attn_mask_dev=mask_dev), 2)
        return ttnn.add(S_I_dev, upd)

    def _downcast_q_device(self, q, a, s, mask_dev, kv_pack=None):
        """Pure-device tail of downcast_q (GCA + s-norm/linear). mask_dev is the
        already-persisted additive mask from _grouping_buffers, passed via attn_mask_dev so
        run_device never uploads a fresh mask itself. q is packed, or flat atoms with kv_pack."""
        ckc, dt = self.compute_kernel_config, self.dtype
        upd = self.downcast_q.run_device(a, q, attn_mask_dev=mask_dev, kv_pack=kv_pack)
        a = ttnn.add(a, upd)
        s = ttnn.rms_norm(s, weight=self.downcast_q_s_n, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.linear(s, self.downcast_q_s_w, compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        return ttnn.add(a, s)

    def _downcast_q(self, Q_L, A_I, S_I, tok_idx):
        dev, dt = self.device, self.dtype
        B, I, _ = A_I.shape
        valid, pack_idx_dev, mask_dev = self._grouping_buffers(tok_idx, B)
        q, kv_pack = _dev(Q_L, dev, dt), lambda x: _pack_atoms_dev_core(x, pack_idx_dev, valid)
        if not _GCA_ATOM_ROWS:
            q, kv_pack = kv_pack(q), None
        out = self._downcast_q_device(q, _dev(A_I, dev, dt), _dev(S_I, dev, dt), mask_dev, kv_pack)
        return out if _RESIDENT_STEP else ttnn.to_torch(out).float()

    def __call__(self, X_noisy_L, t, f, Q_L_init, C_L, P_LL, S_I, Z_II, n_recycle=None):
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx); I = int(tok_idx.max().item()) + 1
        # Batch dim B comes from t (the sampler tiles the per-step scalar sigma to
        # shape [D]). X_noisy_L is [D,L,3]; the TokenInitializer outputs are
        # batch-1 ([L,C]/[I,C]/[I,I,C]). Keep Q_L_init/C_L/S_I/P_LL at batch 1
        # until torch combines them with per-design coordinates/time; broadcasting
        # then materializes only the streams that actually diverge. P_LL remains
        # batch 1 throughout because RFD3AtomBlock broadcasts its invariant pair
        # bias over the attention batch. Z_II also stays batch 1: the pair stack
        # needs a batch dim only for its concat with the per-design distograms, and
        # replicating it there costs 0.67 ms on device against 95.8 ms of host
        # expand + upload (DiffusionTokenEncoder._batched).
        B = t.shape[0]
        if Q_L_init.ndim == 2: Q_L_init = Q_L_init.unsqueeze(0)
        if C_L.ndim == 2: C_L = C_L.unsqueeze(0)
        if S_I.ndim == 2: S_I = S_I.unsqueeze(0)
        if Z_II.ndim == 3: Z_II = Z_II.unsqueeze(0)
        if P_LL.ndim == 3: P_LL = P_LL.unsqueeze(0)
        f = dict(f)
        if _ATTN_OVERLAP:
            f["attn_future"] = _attention_indices_future(f, X_noisy_L, tok_idx, self.N_ATTN_KEYS,
                                                         self.N_ATTN_SEQ, device=dev)
            # Recycle 0's DiT index reads the same step-start coordinates, so it queues behind it.
            f["dit_future"] = _attention_indices_future(f, X_noisy_L[..., f["is_ca"], :], torch.arange(I),
                                                        self.DIT_KEYS, self.DIT_SEQ)
        else:
            f["attn_indices"] = _create_attention_indices(f, X_noisy_L, tok_idx, self.N_ATTN_KEYS,
                                                          self.N_ATTN_SEQ)
        is_motif = f["is_motif_atom_with_fixed_coord"]
        t_L = t.unsqueeze(-1).expand(-1, L) * (~is_motif).float().unsqueeze(0)
        t_I = t.unsqueeze(-1).expand(-1, I) * (~f["is_motif_token_with_fully_fixed_coord"]).float().unsqueeze(0)
        R_L_uniform = self.scale_positions_in(X_noisy_L, t)
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)
        # process_a / process_r both take the SAME R_noisy_L -- upload once,
        # reuse the device tensor for both linears (p23 perf: this step runs
        # once per diffusion timestep, ~200x/design; every avoided
        # ttnn.from_torch is a real dispatch-latency win on this
        # host-dispatch-bound small-protein path, bit-identical since it's
        # the same deterministic bf16 cast either way).
        R_noisy_L_dev = _tt(R_noisy_L, dev, dt)
        if _RESIDENT_STEP:
            lin = dict(compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
            A_I = ttnn.linear(_tt(_scatter_mean(R_noisy_L, tok_idx, I), dev, dt), self.process_a_w, **lin)
            S_I = ttnn.add(self._downcast_c(C_L, S_I, tok_idx),
                           self._process_time_dev(t, f["is_motif_token_with_fully_fixed_coord"], 1))
            Q_L = ttnn.add(_tt_cached(Q_L_init, dev, dt), ttnn.linear(R_noisy_L_dev, self.process_r_w, **lin))
            C_L = ttnn.add(_tt_cached(C_L, dev, dt), self._process_time_dev(t, is_motif, 0))
            C_L = ttnn.add(C_L, ttnn.linear(
                ttnn.rms_norm(C_L, weight=self.process_c_n, epsilon=1e-6, compute_kernel_config=ckc),
                self.process_c_w, compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID))
            return self._denoise(n_recycle, X_noisy_L, R_L_uniform, t_L, f, tok_idx, Q_L, C_L, P_LL,
                                 A_I, S_I, Z_II)
        # process_a (host scatter after device linear)
        a_emb = ttnn.to_torch(ttnn.linear(R_noisy_L_dev, self.process_a_w,
                                          compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)).float()
        A_I = _scatter_mean(a_emb, tok_idx, I)
        S_I = self._downcast_c(C_L, S_I, tok_idx)
        Q_L = Q_L_init + ttnn.to_torch(ttnn.linear(R_noisy_L_dev, self.process_r_w,
                                                          compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)).float()
        C_L = C_L + self._process_time(t_L, 0)
        S_I = S_I + self._process_time(t_I, 1)
        C_L = C_L + ttnn.to_torch(
            ttnn.linear(ttnn.rms_norm(_tt(C_L, dev, dt), weight=self.process_c_n, epsilon=1e-6, compute_kernel_config=ckc),
                        self.process_c_w, compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)).float()
        return self._denoise(n_recycle, X_noisy_L, R_L_uniform, t_L, f, tok_idx, Q_L, C_L, P_LL, A_I, S_I, Z_II)

    def _denoise(self, n_recycle, X_noisy_L, R_L_uniform, t_L, f, tok_idx, Q_L, C_L, P_LL, A_I, S_I, Z_II):
        first_tok = None
        tok_bins = self.diffusion_token_encoder.bins(R_L_uniform[..., f["is_ca"], :])
        if _ATTN_OVERLAP:
            first_tok = self.diffusion_token_encoder.run_device(
                R_L_uniform[..., f["is_ca"], :], S_I, Z_II, D_II_self=None, bins=tok_bins)
            fut = f.pop("attn_future", None)
            f["attn_indices"] = fut.result() if fut is not None else _create_attention_indices(
                f, X_noisy_L, tok_idx, self.N_ATTN_KEYS, self.N_ATTN_SEQ)
        Q_L = self.encoder(Q_L, C_L, P_LL, indices=f["attn_indices"], resident=_RESIDENT_STEP)
        A_I = self._downcast_q(Q_L, A_I, S_I, tok_idx)
        recycled = self._forward_with_recycle(
            n_recycle, X_noisy_L=X_noisy_L, R_L_uniform=R_L_uniform, t_L=t_L, f=f, Q_L=Q_L,
            C_L=C_L, P_LL=P_LL, A_I=A_I, S_I=S_I, Z_II=Z_II, first_tok=first_tok, tok_bins=tok_bins)
        return {"X_L": recycled["X_L"], "sequence_logits_I": recycled["sequence_logits_I"],
                "sequence_restype_I": recycled["sequence_restype_I"]}

    def _forward_with_recycle(self, n_recycle, first_tok=None, **kw):
        n_recycle = n_recycle if n_recycle is not None else self.N_RECYCLE
        rec = {}
        for i in range(n_recycle):
            rec = self._process_(D_II_self=rec.get("D_II_self"), X_L_self=rec.get("X_L"),
                                 tok=first_tok if i == 0 else None, last=i == n_recycle - 1, **kw)
        return rec

    def _process_(self, D_II_self, X_L_self, *, R_L_uniform, X_noisy_L, t_L, f, Q_L, C_L, P_LL, A_I, S_I, Z_II,
                  tok=None, last=True, tok_bins=None):
        is_ca = f["is_ca"]
        R_L_ca = R_L_uniform[..., is_ca, :]
        # z stays on the card between these two: it is [B,I,I,128], the only O(I^2)
        # tensor in the step, and the round trip through host fp32 was 58.8% of the step
        # at 3359 atoms against 13.9% at 419 (p19 step 0) -- the size-dependent cost the
        # GPU does not pay. s is O(I*c_s) and still crosses, because the decoder wants it
        # on host anyway.
        s_dev, z_dev = tok or self.diffusion_token_encoder.run_device(R_L_ca, S_I, Z_II, D_II_self=D_II_self,
                                                                      bins=tok_bins)
        # Built before the s readback so the host works while the encoder is still on the card.
        X_L_ca = X_noisy_L[..., is_ca, :] if X_L_self is None else X_L_self[..., is_ca, :]
        fut = f.pop("dit_future", None) if X_L_self is None else None
        dit_idx = fut.result() if fut is not None else _create_attention_indices(
            f, X_L_ca, torch.arange(S_I.shape[-2], device=X_L_ca.device), self.DIT_KEYS, self.DIT_SEQ)
        S_I = s_dev if _RESIDENT_STEP else ttnn.to_torch(s_dev).float()
        A_I = self.diffusion_transformer(A_I, S_I, z_dev, dit_idx, resident=_RESIDENT_STEP)
        if not self.diffusion_token_encoder.owns(z_dev):
            ttnn.deallocate(z_dev)
        # Both of the decoder's outputs feed device consumers only -- q the R update below,
        # a the sequence head -- so neither crosses. Each ttnn.to_torch is a blocking drain,
        # and at 3359 atoms the fifteen of them in a step cost 129 ms for 9.8 MB (p20 s2):
        # the sync count is what this removes, not the bytes.
        A_I, Q_L = self.decoder.run_full_device(A_I, S_I, Q_L, C_L, P_LL,
                                                tok_idx=f["atom_to_token_map"], indices=f["attn_indices"])
        R_upd = ttnn.to_torch(ttnn.linear(ttnn.rms_norm(Q_L,
                                                        weight=self.to_r_n, epsilon=1e-6, compute_kernel_config=self.compute_kernel_config),
                                          self.to_r_w, compute_kernel_config=self.compute_kernel_config,
                                          dtype=self.dtype, core_grid=CORE_GRID_MAIN)).float()
        ttnn.deallocate(Q_L)
        X_out = self.scale_positions_out(R_upd, X_noisy_L, t_L)
        # Only the last recycle's sequence head is returned and only an earlier recycle's
        # distogram is read (by the next one): each runs where its output is used, so a
        # 2-recycle step saves one blocking logits read and one host distogram.
        logits, aatype = self.sequence_head(A_I) if last else (None, None)
        ttnn.deallocate(A_I)
        D_II_self = None if last else _scaled_distogram_bins(X_out[..., is_ca, :].detach(),
                                                             sigma_data=self.SIGMA_DATA, n_bins=65)
        return {"X_L": X_out, "D_II_self": D_II_self, "sequence_logits_I": logits,
                "sequence_restype_I": aatype}


def build_diffusion_module(state_dict, compute_kernel_config=None, dtype=None):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return RFD3DiffusionModule(state_dict, compute_kernel_config, dtype=dtype)
