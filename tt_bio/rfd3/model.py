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
from ..envflags import env_flag
from . import block_sparse as _BS
from .tiles import TILE, align_tile, pad_axis
from ..tenstorrent import Module, get_device, CORE_GRID_MAIN, attn_value_matmul

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
# 4.8e-3), so it is off by default and release-gated.
_PROCESS_Z_COLLAPSE = env_flag("RFD3_PROCESS_Z_COLLAPSE", False)
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
# Eight calls per step (transition_2.{0,1} at H=256 and pairformer_stack.{0,1}.z_transition at
# H=512, each twice for the two recycles): 140.8 -> 121.5 ms/step, -19.3 ms/step.
#
# The chunking is NOT where the win is. With the intermediates left in DRAM, chunking alone is a
# LOSS of 1.3-1.5 ms/call (perf/p64/pair_transition_l1.json arm B): the slice, the closing concat
# and the extra op count cost more than they return. All of the result is the L1 residency.
_PAIR_TRANSITION_L1 = env_flag("RFD3_PAIR_TRANSITION_L1", True)


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
# ttnn.gather (0.68.0) on dim 3 silently returns wrong data for every tile-row after the first
# once the indexed axis exceeds this many elements: 99.87 % of elements wrong at the production
# [1,4,6051,6080], and ~100x slower at the same threshold. The trigger is element count, not
# bytes -- bf16 breaks at 2048 while fp32 at 1920 (7.5 KB/row) is exact. Measured in
# scripts/rfd3_port/p81{,b,c}_*.py, perf/p81/*.json. ttnn.scatter at the same shape is clean.
_TTNN_GATHER_MAX_KEY_AXIS = 1920


def set_gathered_softmax(on):
    """Toggle the gathered atom softmax from a screen without going through the environment."""
    global _GATHERED_SOFTMAX
    _GATHERED_SOFTMAX = bool(on)

# [collapsed calls, shipped calls]. A silent arm is what made p86's batching result wrong, so both
# process_z branches count themselves: the on arm must show shipped == 0 and the off arm
# collapsed == 0, at the same total.
PZSTATS = [0, 0]


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


def _tuned_linear(x, w, *, bias=None, ckc=None, dtype=None, core_grid=BATCH_INVARIANT_GRID,
                  mem=None):
    """`ttnn.linear` with a calibrated, bit-exact program config where one helps.

    No `activation=`: ttnn wants a fused activation on the program config instead, so the two
    silu-gated linears keep the default path (they are the cheap half of a Transition anyway).

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
    key = (tuple(list(x.padded_shape)), tuple(list(w.padded_shape)), x.dtype, dtype,
           bias is not None, core_grid)
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
    """
    # Time the default BEFORE building anything else. Two reasons, both measured
    # (scripts/rfd3_port/p34_calib_cost.py, perf/p34/calib_cost.json): it is the size gate
    # `_tunable` no longer applies, and the setup below is 64.9% of calibration's whole cost,
    # because `_mm_random_like` draws and uploads a full-size operand -- 256 M elements for
    # `[8,250,250,512]`, 1.69 s of the 2.01 s that shape spends. A shape under the floor now
    # pays one timed default and nothing else.
    ref = ttnn.linear(x, w, core_grid=core_grid, **kw)
    default_t = _mm_time(lambda: ttnn.linear(x, w, core_grid=core_grid, **kw))
    if default_t * 1e3 < _TUNE_MIN_MS:
        ttnn.deallocate(ref)
        if _TUNE_LOG:
            print(f"[tune] x={tuple(x.padded_shape)} w={tuple(w.padded_shape)} "
                  f"default={default_t * 1e3:8.3f} ms  under {_TUNE_MIN_MS} ms floor, SKIP",
                  flush=True)
        return None
    rx, rw = _mm_random_like(x, 0), _mm_random_like(w, 1)
    rref = ttnn.linear(rx, rw, core_grid=core_grid, **kw)
    budget = default_t / _TUNE_MIN_GAIN
    best = None
    for pc in _mm_candidates(x, w, get_device().compute_with_storage_grid_size()):
        try:
            if _mm_maxabs(ttnn.linear(rx, rw, program_config=pc, **kw), rref) != 0.0:
                continue
            if _mm_maxabs(ttnn.linear(x, w, program_config=pc, **kw), ref) != 0.0:
                continue
            t = _mm_time(lambda: ttnn.linear(x, w, program_config=pc, **kw))
        except Exception:
            continue  # illegal L1 / subblock combinations are expected and simply skipped
        if t < budget:
            best, budget = pc, t
    for t in (rx, rw, rref, ref):
        ttnn.deallocate(t)
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


def _sinusoidal_embed(pos, valid_mask, n_freqs=32):
    """Host: SinusoidalDistEmbed inputs -> (sincos [L,L,2*n_freqs], V_LL [L,L,1])."""
    D = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    dist = torch.linalg.norm(D, dim=-1)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs)
    angles = dist.unsqueeze(-1) * freq
    sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(torch.float32)
    return sincos, valid_mask.to(torch.float32)


def _build_valid_mask(tok_idx):
    tokens, counts = torch.unique(tok_idx, return_counts=True)
    A = int(counts.max())
    return torch.arange(A, device=tok_idx.device)[None, :] < counts[:, None]


def _scatter_mean_pool(pairwise_atom, tok_idx, I):
    """Host: mean-pool [L,L,c] -> [I,I,c] (pairwise_mean_pool)."""
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


def _tt_host(x, dtype=ttnn.bfloat16):
    """Host-side tiled tensor, for filling a persistent trace-input buffer.

    Cannot use _tt: copy_host_to_device_tensor needs a host tensor whose spec already
    matches the tiled device buffer the trace captured, so the tilize has to happen here.
    Casting in torch first still avoids the expensive part -- letting ttnn convert fp32
    while it tilizes costs 42.7 ms for a 6.9M-element bf16 tensor against 17.3 ms
    pre-cast, and the result is bit-identical."""
    torch_dtype = _TORCH_DTYPE.get(dtype)
    if torch_dtype is not None:
        x = x.to(torch_dtype)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=dtype)


def _pair_transition_chunk_h(batch, w_pad, hidden, height):
    """Rows of the pair tensor one L1-resident SwiGLU chunk may cover.

    Live per chunk is `b` + `m`, both [batch, h, w_pad, hidden] bf16, so the batch is part
    of the footprint. It used to be missing, and that is what closed batching for three
    passes: at b=2 and h=64 each resident is 2*64*704*512*2 = 92 274 688 B, the second one
    fails against 68.4 MB free, and the crash lands on `m` in `_swiglu`. That byte figure
    is exactly the request the batched run died on. Dividing by the batch holds the
    live footprint at the measured-safe 138 MB whatever the batch is, and is a no-op at
    b=1, where the cap is 95 either way and h stays 64.
    """
    cap = _PAIR_TRANSITION_L1_BYTES // (4 * max(1, batch) * w_pad * hidden)
    return max(1, min(height, _PAIR_TRANSITION_H_CHUNK, cap))


def _tt_refresh(x, dev_tensor, dtype=ttnn.bfloat16):
    """Overwrite a persistent trace-input buffer with fresh host data."""
    ttnn.copy_host_to_device_tensor(_tt_host(x, dtype), dev_tensor)


def _trace_output_copy(out):
    """Hand a trace's result to a caller that owns what it is given.

    A replay writes into the buffer the capture allocated, and that buffer has to stay
    alive for every later replay -- so returning it directly hands the caller a tensor it
    must not free. The eager paths return per-call intermediates, and RFD3's consumers
    deallocate accordingly: `RFD3DiffusionModule._process_` does `ttnn.deallocate(Q_L)`
    on the decoder's output once the R update is read back (6985b2f37, "keep the decoder's
    two outputs on the card"). That freed the traced decoder's own output buffer, so
    `RFD3_TRACE_DECODER=1` produced correct coordinates for exactly one step and then
    replayed into freed memory: the third call raised "Buffer is not allocated" out of the
    first op that touched the result. p25/p26 measured the path before the residency change
    landed, and it is opt-in and default-off, so nothing caught the regression (p32).

    Copying keeps both contracts intact and is not worth avoiding: the decoder's output is
    [B, L, C_ATOM] -- 0.86 MB at 3359 atoms against the ~250 ms step that produced it.
    """
    return ttnn.clone(out)


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
        """Whole-tensor by default; L1-resident row blocks on the token pair tensor.

        See `_PAIR_TRANSITION_L1` for the measurements. `RFD3_PAIR_TRANSITION_L1=0` restores
        the whole-tensor path op for op at 3-to-63-row tails; at 1- and 2-row tails (22 of 689
        production sizes, e.g. H=513/514) it differs by 2.44e-4, one bf16 ULP at this magnitude
        (`perf/p131/tail_sweep.json`). Not chased further: the gap is sub-ULP rounding, not a
        wrong result.
        """
        if not (_PAIR_TRANSITION_L1 and len(x.shape) == 4
                and x.shape[2] >= _PAIR_TRANSITION_MIN_W):
            return self._swiglu(x, None)
        H, hidden = x.shape[1], int(self.fc1_w.shape[-1])
        h = _pair_transition_chunk_h(x.shape[0], int(x.padded_shape[2]), hidden, H)
        # Slice lazily rather than `ttnn.chunk`, which materialises a second full copy of the
        # pair tensor up front. The 685-row tail is ragged at every h and gets its own shape,
        # hence its own program-config cache entry, which is what keeps it exact.
        parts = []
        for s in range(0, H, h):
            c = x[:, s:min(s + h, H)]
            parts.append(self._swiglu(c, ttnn.L1_MEMORY_CONFIG))
            ttnn.deallocate(c)
        out = ttnn.concat(parts, dim=1)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def _swiglu(self, x, mem):
        """RMSNorm + silu-gated SwiGLU. `mem=None` keeps every intermediate in DRAM.

        With `mem` set, `fc2`'s output and the gated product stay in L1, so `b` and `m` are never
        written to DRAM and never read back: 1975 of the 3951 MB an H=512 call moves.

        `x_norm` and `fc1`'s output stay in DRAM on purpose, and that is the whole reason this is
        bit-exact. `fc1` is heuristic-blocked (its fused silu cannot ride on an explicit program
        config, and no bit-exact config for it exists -- closed in `rfd3-close-the-page-gap`), so
        it re-blocks K and re-rounds when either its input or its output moves to L1: measured
        0.03125 at hidden=512 either way (perf/p66/audit_perop.json). `fc2` goes through
        `_tuned_linear`, whose pinned config makes the blocking independent of L1 pressure, and
        the multiply is elementwise. Hence L1 for those two and DRAM for the other two.
        """
        xn = ttnn.rms_norm(x, weight=self.norm_w, epsilon=1e-6,
                            compute_kernel_config=self.compute_kernel_config)
        a = ttnn.linear(xn, self.fc1_w, activation="silu",
                         compute_kernel_config=self.compute_kernel_config,
                         dtype=self.dtype, core_grid=BATCH_INVARIANT_GRID)
        b = _tuned_linear(xn, self.fc2_w, ckc=self.compute_kernel_config,
                          dtype=self.dtype, core_grid=BATCH_INVARIANT_GRID, mem=mem)
        ttnn.deallocate(xn)
        m = ttnn.multiply(a, b) if mem is None else ttnn.multiply(a, b, memory_config=mem)
        ttnn.deallocate(b)
        out = _tuned_linear(m, self.fc3_w, ckc=self.compute_kernel_config,
                            dtype=self.dtype, core_grid=CORE_GRID_MAIN)
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
        B, I = s.shape[0], s.shape[1]
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
        bias = _tuned_linear(z, self.to_b_w, ckc=self.compute_kernel_config,
                             dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        bias = ttnn.permute(bias, (0, 3, 1, 2))  # [1,16,I,I]
        # Manual attention (SDPA forbids head_dim=24 padding); bf16 softmax matches the
        # reference (autocast bf16). softmax over keys (dim=-1).
        #
        # Extend the KEY axis out to a tile multiple, the same way the DiT's attention does at
        # :1600 -- zero keys and values scored at -1e4, so their weight is exactly 0 after exp.
        # A ragged key axis here is not a wrong answer (softmax_generic declines it and
        # ttnn.softmax masks its own tail) but it is the fused softmax going dark on every call
        # at every design length that is not a multiple of 32: censused 6 of 6. See
        # tt_bio/token_axis.py and PLAYBOOKS.md §MODEL 2b.
        n_key = align_tile(I)
        k = pad_axis(k, n_key, 2, 0.0)
        v = pad_axis(v, n_key, 2, 0.0)
        kt = ttnn.permute(k, (0, 1, 3, 2))                 # [1,16,24,n_key]
        sc = ttnn.matmul(q, kt, compute_kernel_config=ckc)  # [1,16,I,n_key]
        ttnn.deallocate(kt)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
        sc = ttnn.multiply(sc, self.head_dim ** -0.5)
        bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
        bias_f = pad_axis(bias_f, n_key, 3, -1e4)
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


class PairformerBlock(Module):
    def __init__(self, state_dict, ckc, c_s=384, c_z=128, n_head=16, dtype=None,
                 fp32_residual=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.fp32_residual = fp32_residual
        self.z_transition = Transition(self.scope("z_transition"), ckc, c_z, n=4, dtype=self.dtype)
        self.s_transition = Transition(self.scope("s_transition"), ckc, c_s, n=4, dtype=self.dtype)
        self.attn = PairformerAttention(self.scope("attention_pair_bias"), ckc, c_s, c_z, n_head, dtype=self.dtype)

    def __call__(self, s, z):
        if not self.fp32_residual:
            z = ttnn.add(z, self.z_transition(z))
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

        # ---- P_LL [L, L, C_ATOMPAIR=16] ----
        # motif_pos_embedder (SinusoidalDistEmbed): host sincos -> device output_proj + valid_mask linears
        mp = f["motif_pos"].float()
        vm_mp = (f["is_motif_atom_with_fixed_coord"].unsqueeze(-1) & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)).unsqueeze(-1).float()
        sc, vsc = _sinusoidal_embed(mp, vm_mp)                  # [L,L,64], [L,L,1]
        sc = _tt(sc.unsqueeze(0), dev, dt); vsc = _tt(vsc.unsqueeze(0), dev, dt)
        p = ttnn.multiply(ttnn.linear(sc, self.motif_pos_proj, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vsc)
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(vsc, self.motif_pos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vsc))
        # ref_pos_embedder (no-frame): host inv_dist -> device linears
        rp = f["ref_pos"].float()
        same_tok = (f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)).unsqueeze(-1).float()
        has_seq = (f["is_motif_atom_with_fixed_seq"].unsqueeze(-1) & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)).unsqueeze(-1).float()
        vm_rp = same_tok * has_seq
        D = rp.unsqueeze(-2) - rp.unsqueeze(-3)
        invd = 1.0 / (1.0 + D.pow(2).sum(-1, keepdim=True).clamp(min=1e-6))
        invd = _tt(invd.unsqueeze(0), dev, dt); vm_rp = _tt(vm_rp.unsqueeze(0), dev, dt)
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(invd, self.refpos_invd, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
        p = ttnn.add(p, ttnn.multiply(ttnn.linear(vm_rp, self.refpos_vm, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN), vm_rp))
        # process_single_l/m (ReLU + linear on C_L) -- c_l_h uploaded once,
        # reused for both sl/sm (was 2x, p23 perf; identical bf16 cast either way)
        c_l_h = ttnn.to_torch(c_l).float().squeeze(0)        # [L, C_ATOM]
        c_l_dev = _tt(c_l_h.unsqueeze(0), dev, dt)
        sl = ttnn.relu(c_l_dev)
        sl = ttnn.linear(sl, self.proc_single_l_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,L,16]
        sm = ttnn.relu(c_l_dev)
        sm = ttnn.linear(sm, self.proc_single_m_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        p = ttnn.add(p, ttnn.unsqueeze(sl, -2))             # [1,L,1,16] + [1,L,L,16] -> [1,L,L,16]
        p = ttnn.add(p, ttnn.unsqueeze(sm, -3))
        # process_z(Z_init_II): RMSNorm + linear -> [I,I,16]; gather to atoms [L,L,16]
        # (Z_init_II_dev kept around unmodified -- ttnn ops return new tensors,
        # not in-place -- and reused below for the zupd add instead of a 2nd
        # upload of the same host tensor; p23 perf, bit-identical.)
        Z_init_II_dev = _tt(Z_init_II.unsqueeze(0), dev, dt)
        z_dev = ttnn.rms_norm(Z_init_II_dev, weight=self.proc_z_n, epsilon=1e-6, compute_kernel_config=ckc)
        pz = ttnn.linear(z_dev, self.proc_z_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)  # [1,I,I,16]
        pz_h = ttnn.to_torch(pz).float().squeeze(0)          # [I,I,16]
        pz_h = pz_h[tok_idx][:, tok_idx, :]                   # [L,L,16] (gather both axes)
        p = ttnn.add(p, _tt(pz_h.unsqueeze(0), dev, dt))
        # pair_mlp (ReLU + linear x3) residual
        m = p
        for w in self.pair_mlp_w:
            m = ttnn.relu(m)
            m = ttnn.linear(m, w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        p = ttnn.add(p, m)
        p_h = ttnn.to_torch(p).float().squeeze(0)            # [L,L,16]
        # pooled = scatter_mean_pool(process_pll(P_LL)) -> project_pll -> add to Z
        pll = ttnn.linear(p, self.proc_pll_w, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        pll_h = ttnn.to_torch(pll).float().squeeze(0)       # [L,L,16]
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
        entry = (p_host, ttnn.from_torch(
            p_host.reshape(-1, p_host.shape[-1]).contiguous(),
            layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype))
        tables[key] = entry
    return entry[1]


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

    def run_device(self, q, kv, attn_mask=None, attn_mask_dev=None):
        """q [B,T,Q,Cq], kv [B,T,K,Ckv]; return device [B,T,Q,Cq].
        attn_mask_dev (already-uploaded additive mask) takes priority over
        attn_mask (host bool mask, uploaded here) when both are given."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        batch, tokens, n_query, _ = q.shape
        n_key = kv.shape[2]
        q = ttnn.rms_norm(q, weight=self.ln_q, epsilon=1e-6, compute_kernel_config=ckc)
        kv = ttnn.rms_norm(kv, weight=self.ln_kv, epsilon=1e-6, compute_kernel_config=ckc)
        qq = ttnn.linear(q, self.to_q, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        kk = ttnn.linear(kv, self.to_k, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        vv = ttnn.linear(kv, self.to_v, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        gg = ttnn.linear(q, self.to_g, compute_kernel_config=ckc, dtype=dt, core_grid=CORE_GRID_MAIN)
        qq = ttnn.rms_norm(qq, weight=self.q_norm, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=self.k_norm, epsilon=1e-6, compute_kernel_config=ckc)

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
        qq = split(qq, n_query)
        kk = split(kk, n_key)
        vv = split(vv, n_key)
        scores = ttnn.matmul(
            qq, ttnn.permute(kk, (0, 1, 3, 2)), compute_kernel_config=ckc,
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
        return ttnn.add(ttnn.multiply(a, ttnn.sigmoid(gain)), bias)

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
        pair_bias = _tuned_linear(
            p, self.b_w, ckc=self.compute_kernel_config, dtype=self.dtype,
            core_grid=CORE_GRID_MAIN,
        )
        pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
        if cache is not None:
            cache[(id(self), "pb")] = (key, p, pair_bias)
        return pair_bias

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

    def __call__(self, q, c, p, additive_mask=None, sparse_qk=None, bias_cache=None,
                 pair_bias=None):
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
        norm = self._adaln(
            q_compute, c, self.a_ln_s, self.a_gain_w, self.a_gain_b, self.a_bias_w
        )
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
        if bs_out is not None:
            _BS.STATS[0] += 1
        elif sparse_qk is not None and len(sparse_qk) > 4:
            _BS.STATS[1 if _BS.enabled() else 2] += 1
        fused = False
        dense_fused = False
        gathered = None
        if bs_out is not None:
            pass
        elif sparse_qk is None:
            # `pair_bias` arrives precomputed when the caller hoisted all of its blocks'
            # projections into one matmul; see _PAIRBIAS_FUSED.
            if pair_bias is None:
                pair_bias = _tuned_linear(
                    p, self.b_w, ckc=ckc, dtype=dt, core_grid=CORE_GRID_MAIN,
                )
            pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
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
            # the softmax that consumes it. `dense_bias is None` is L6a's own gate, so the trace
            # path -- which passes its own template -- keeps the old route untouched.
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
            out = attn_value_matmul(attention, vv, ckc, dt)
        else:
            out = bs_out
        out = _merge_heads(out, (batch, length, self.n_head * self.head_dim))
        out = ttnn.multiply(out, ttnn.sigmoid(gg))
        out = _tuned_linear(
            out, self.o_w, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID
        )
        gate = ttnn.linear(
            c, self.a_out_w, bias=self.a_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        upd = ttnn.multiply(out, ttnn.sigmoid(gate))
        if f32:
            q = ttnn.add(q, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config()))
        else:
            q = ttnn.add(q, upd)
        ttnn.deallocate(upd)

        q_compute = ttnn.typecast(q, dt, memory_config=q.memory_config()) if f32 else q
        norm = self._adaln(
            q_compute, c, self.t_ln_s, self.t_gain_w, self.t_gain_b, self.t_bias_w
        )
        left = ttnn.linear(
            norm, self.t_fc1, activation="silu", compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        right = _tuned_linear(
            norm, self.t_fc2, ckc=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        update = _tuned_linear(
            ttnn.multiply(left, right), self.t_fc3, ckc=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        gate = ttnn.linear(
            c, self.t_out_w, bias=self.t_out_b, compute_kernel_config=ckc,
            dtype=dt, core_grid=BATCH_INVARIANT_GRID,
        )
        upd = ttnn.multiply(update, ttnn.sigmoid(gate))
        if f32:
            q = ttnn.add(q, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config()))
        else:
            q = ttnn.add(q, upd)
        return q


class LocalAtomTransformer(Module):
    """Three-block RFD3 atom encoder with parity-preserving sparse QK.

    trace=True opts into a ttnn trace-capture/replay fast path (per
    rfd3-trace-viability-submodule-granularity: this narrow-channel/few-head
    3-block stack measured a real 5.46-5.56x isolated speedup, unlike the
    18-block token DiT's confirmed dead end). Default off -- bit-identical
    eager path unchanged unless the caller opts in. The trace is captured
    once per (L, n_key) shape and replayed with fresh q/c/p/mask data staged
    into persistent device buffers every call (correct regardless of how
    often the data actually changes -- see the class docstring on _trace).

    p26 found this class's OWN trace=True unsafe when wired directly into
    RFD3DiffusionModule (encoder call, trace stays open) followed by `_downcast_q`
    running eagerly right after: `_downcast_q` re-derived and re-uploaded its packing
    index on every call (a fresh device allocation immediately after the trace had just
    executed) and that hung the device (py-spy: stuck in `_downcast_q`'s closing
    `ttnn.to_torch()`, same stack frame every time -- see
    ttnn-trace-interleaved-eager-corruption / rfd3-trace-hang-vs-corruption-two-gate-catch).
    Isolated component-level PCC (direct repeated __call__ invocations, no intervening
    eager allocation) was always clean -- the bug was specific to the full-pipeline
    interleaving.

    p27 fix: RFD3DiffusionModule no longer uses THIS class's own trace mechanism in
    production (self.encoder is always built with trace=False). Production encoder
    tracing now lives in RFD3DiffusionModule._encoder_downcast_traced, which captures
    the encoder's run_device AND downcast_q's core in ONE combined trace (so no eager
    allocation ever runs while that trace is open) and uses a cached packing-index
    buffer (RFD3DiffusionModule._grouping_buffers) instead of re-uploading it every call.
    This class's own trace=True path is kept only for isolated-component testing
    -- do not wire it directly into a production pipeline without the same
    combined-trace treatment."""

    def __init__(self, state_dict, ckc, n_blocks=3, dtype=None, fp32_residual=False, trace=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.blocks = [
            RFD3AtomBlock(self.scope(f"blocks.{i}"), ckc, dtype=self.dtype,
                          fp32_residual=fp32_residual)
            for i in range(n_blocks)
        ]
        self.trace = trace
        self._trace_state = None  # {"id", "shape", "q", "c", "p", "mask", "output"}
        self._mask_cache = {}  # one -1e4 scatter template, see _mask_template

    def run_device(self, q, c, p, additive_mask):
        for block in self.blocks:
            q = block(q, c, p, additive_mask)
        return q

    def _persist(self, x_host):
        host_t = _tt_host(x_host, self.dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, self.device)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    def _capture_trace(self, q_host, c_host, p_host, mask_host, shape_key):
        dev = self.device
        q_p, c_p, p_p, mask_p = (self._persist(x) for x in (q_host, c_host, p_host, mask_host))
        for _ in range(2):  # warmup: compiles every kernel (capture disallows compilation)
            _ = self.run_device(q_p, c_p, p_p, mask_p)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = self.run_device(q_p, c_p, p_p, mask_p)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._trace_state = dict(id=tid, shape=shape_key, q=q_p, c=c_p, p=p_p, mask=mask_p, output=out)

    def _run_device_traced(self, q_host, c_host, p_host, mask_host):
        import tt_bio.tenstorrent as _TTd
        if _TTd.trace_region_size() <= 0:
            raise ValueError(
                "LocalAtomTransformer(trace=True) needs a device opened with a trace "
                "region; call get_device(trace_region_size=1 << 28) (or larger) first.")
        dev, dt = self.device, self.dtype
        shape_key = (tuple(q_host.shape), tuple(c_host.shape), tuple(p_host.shape), tuple(mask_host.shape))
        st = self._trace_state
        if st is None or st["shape"] != shape_key:
            if st is not None:
                ttnn.release_trace(dev, st["id"])
            self._capture_trace(q_host, c_host, p_host, mask_host, shape_key)
        else:
            _tt_refresh(q_host, st["q"], dt)
            _tt_refresh(c_host, st["c"], dt)
            _tt_refresh(p_host, st["p"], dt)
            _tt_refresh(mask_host, st["mask"], dt)
        ttnn.execute_trace(dev, self._trace_state["id"], cq_id=0, blocking=True)
        return self._trace_state["output"]

    def __call__(self, q_host, c_host, p_host, indices):
        dt, dev = self.dtype, self.device
        p_host = p_host.unsqueeze(0) if p_host.ndim == 3 else p_host
        if env_flag("RFD3_SPARSE_QK", True):
            p, n_keys, attn_idx_dev, dense_bias, gathered, block = _sparse_qk_inputs(
                p_host, indices, dev, dt, mask_cache=self._mask_cache
            )
            q, c = _tt(q_host, dev, dt), _tt(c_host, dev, dt)
            sparse_qk = (n_keys, attn_idx_dev, dense_bias, gathered, block)
            for block in self.blocks:
                q = block(q, c, p, sparse_qk=sparse_qk)
            out = q
        else:
            mask_host = _dense_attention_mask(indices)
            if self.trace:
                out = self._run_device_traced(q_host, c_host, p_host, mask_host)
            else:
                q = _tt(q_host, dev, dt)
                c = _tt(c_host, dev, dt)
                p = _tt(p_host, dev, dt)
                mask = _tt(mask_host, dev, dt)
                out = self.run_device(q, c, p, mask)
        return ttnn.to_torch(out).float()


class CompactStreamingDecoder(Module):
    """RFD3 decoder: three device Upcast/atom blocks plus device Downcast.

    trace=True opts into a ttnn trace-capture/replay fast path over the core
    upcast/atom-block loop (per rfd3-trace-viability-submodule-granularity: this
    narrow-channel/few-head 3-block loop measured a real 4.12x isolated speedup).
    The downcast GCA + final s-processing tail stay eager -- not part of the
    traced region. Default off -- bit-identical eager path unchanged unless the
    caller opts in.

    Two independent buffer lifetimes (see rfd3-rfdiffusion3-port-p24 handoff):
    pack_idx/unpack_idx/valid/upcast_mask_dev depend only on tok_idx, which is
    the SAME object for an entire design's sampling loop (RFD3Sampler.sample
    passes one `f` dict by reference through every step) -- cached by id(tok_idx)
    and rebuilt only when that identity (or shape) changes. a/q/c/p/mask change
    every call (q/c/p are step-fixed across the decoder's 2 recycle calls but
    that's a perf nuance, not a correctness one -- refreshing them on every call
    via copy_host_to_device_tensor into the same persistent trace buffers is
    always correct, just leaves one redundant re-upload per step on the table)."""

    def __init__(self, state_dict, ckc, dtype=None, fp32_residual=False, trace=False):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.trace = trace
        self._design_state = None  # {"key", "valid", "pack_idx_dev", "unpack_idx_dev", "upcast_mask_dev"}
        self._trace_state = None   # {"id", "shape", "a", "q", "c", "p", "mask", "output"}
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
        `a` is the FLAT (not pre-split) atom-pair stream; the reshape to
        a_split runs here (not by the caller) so a trace replay re-derives
        a_split from whatever fresh data was staged into `a`'s buffer."""
        a_split = ttnn.reshape(a, (a.shape[0], a.shape[1], 3, 256))
        for upcast, atom_block in zip(self.upcast, self.atom_blocks):
            q_grouped = self._pack_atoms_device(q, pack_idx_dev, valid)
            q_grouped = ttnn.add(
                q_grouped, upcast.run_device(q_grouped, a_split, attn_mask_dev=upcast_mask_dev)
            )
            q = self._unpack_atoms_device(q_grouped, unpack_idx_dev, length)
            q = atom_block(q, c, p, mask, sparse_qk=sparse_qk, bias_cache=bias_cache)
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
            if self._trace_state is not None:
                # a design change can change L/I shapes -- any captured trace is stale.
                self._release_sparse_trace(self._trace_state)
                self._trace_state = None
        return st["valid"], st["pack_idx_dev"], st["unpack_idx_dev"], st["upcast_mask_dev"]

    def _persist(self, x_host):
        host_t = _tt_host(x_host, self.dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, self.device)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    def _persist_index(self, x_host, layout):
        host_t = ttnn.from_torch(x_host, layout=layout, dtype=ttnn.uint32)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, self.device)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    def _capture_sparse_trace(
        self, a_host, q_host, c_host, p_sparse_host,
        attn_idx_host, n_keys, upcast_mask_dev, pack_idx_dev,
        unpack_idx_dev, valid, length, shape_key, step_key,
    ):
        dev = self.device
        a_p, q_p, c_p, p_p = (
            self._persist(x) for x in (a_host, q_host, c_host, p_sparse_host)
        )
        attn_p = self._persist_index(attn_idx_host, ttnn.TILE_LAYOUT)
        batch = q_host.shape[0]
        n_heads = self.atom_blocks[0].n_head
        # tile-multiple key axis, see _mask_template
        dense_bias = ttnn.full(
            (batch, n_heads, length, align_tile(length)), -1e4,
            dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        sparse_qk = (n_keys, attn_p, dense_bias, None, None)
        for _ in range(2):
            _ = self.run_device(
                a_p, q_p, c_p, p_p, None, upcast_mask_dev, pack_idx_dev,
                unpack_idx_dev, valid, length, sparse_qk=sparse_qk,
            )
        ttnn.synchronize_device(dev)
        # TWO traces, so p30's dense-bias reuse survives tracing instead of being traded
        # for it. A trace has no branches: whatever `run_device` issued at capture time it
        # re-issues on every replay, so a single trace containing the pair-bias scatter pays
        # it on BOTH of a step's recycle calls -- exactly the six-scatter cost the eager
        # `_bias_cache` removes (p30, +7%). Capturing the loop twice against ONE cache dict
        # splits it: the first capture misses on every block and bakes in the scatter,
        # writing each block's fp32 bias into a buffer the dict now holds; the second
        # capture hits on every block and bakes in only the reads of those buffers. So
        # replaying `id` for recycle 1 and `id_reuse` for recycle >= 2 issues three
        # scatters per step, not six, and trace and cache compound (p32).
        #
        # The bias buffers are ordinary live device tensors -- the dict's reference keeps
        # them allocated, so the second capture's own intermediates cannot land on them,
        # and the two traces are only ever replayed in order (build then reuse) within a
        # step. Their CONTENTS at capture time are irrelevant: capture records commands
        # without executing them.
        bias_cache = {}
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = self.run_device(
            a_p, q_p, c_p, p_p, None, upcast_mask_dev, pack_idx_dev,
            unpack_idx_dev, valid, length, sparse_qk=sparse_qk,
            bias_cache=bias_cache,
        )
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        tid_reuse = ttnn.begin_trace_capture(dev, cq_id=0)
        out_reuse = self.run_device(
            a_p, q_p, c_p, p_p, None, upcast_mask_dev, pack_idx_dev,
            unpack_idx_dev, valid, length, sparse_qk=sparse_qk,
            bias_cache=bias_cache,
        )
        ttnn.end_trace_capture(dev, tid_reuse, cq_id=0)
        self._trace_state = dict(
            id=tid, id_reuse=tid_reuse, shape=shape_key, step_key=step_key,
            a=a_p, q=q_p, c=c_p, p=p_p, attn_idx=attn_p, dense_bias=dense_bias,
            bias_cache=bias_cache, output=out, output_reuse=out_reuse,
        )

    def _run_device_sparse_traced(
        self, a_host, q_host, c_host, p_host, indices, upcast_mask_dev,
        pack_idx_dev, unpack_idx_dev, valid, length,
    ):
        import tt_bio.tenstorrent as _TTd
        if _TTd.trace_region_size() <= 0:
            raise ValueError("Sparse decoder trace needs an enabled trace region")
        dev, dt = self.device, self.dtype
        step_key = (id(q_host), id(c_host), id(p_host), id(indices))
        st = self._trace_state
        reuse = (st is not None and st.get("step_key") == step_key
                 and st["shape"][1] == tuple(a_host.shape))
        if reuse:
            # The decoder's second recycle call within a step: q/c/p and the
            # neighbour index are already staged and only `a` differs, so the host
            # pair gather (6.5 ms at 250 residues) would be thrown away. Skip it --
            # and replay the trace that reuses the bias the previous call built
            # (see _capture_sparse_trace) rather than rebuilding it.
            _tt_refresh(a_host, st["a"], dt)
        else:
            # The traced decoder stages p_sparse in a persistent buffer that
            # ttnn.embedding cannot write into, so this path keeps the host gather
            # _sparse_qk_inputs no longer needs. RFD3_TRACE_DECODER is opt-in and off
            # in production, so the gather lever above is what a shipped run takes.
            p_sparse, attn_idx, n_keys = _sparse_qk_host(p_host, indices)
            shape_key = (
                "sparse_qk", tuple(a_host.shape), tuple(q_host.shape),
                tuple(c_host.shape), tuple(p_sparse.shape), n_keys,
            )
            if st is None or st["shape"] != shape_key:
                if st is not None:
                    self._release_sparse_trace(st)
                self._capture_sparse_trace(
                    a_host, q_host, c_host, p_sparse, attn_idx, n_keys,
                    upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length,
                    shape_key, step_key,
                )
            else:
                _tt_refresh(a_host, st["a"], dt)
                for host, target in ((q_host, st["q"]), (c_host, st["c"]),
                                     (p_sparse, st["p"])):
                    _tt_refresh(host, target, dt)
                _tt_refresh(attn_idx, st["attn_idx"], ttnn.uint32)
                st["step_key"] = step_key
        st = self._trace_state
        key = "id_reuse" if reuse else "id"
        ttnn.execute_trace(dev, st[key], cq_id=0, blocking=True)
        return _trace_output_copy(st["output_reuse" if reuse else "output"])

    def _release_sparse_trace(self, st):
        for key in ("id", "id_reuse"):
            if st.get(key) is not None:
                ttnn.release_trace(self.device, st[key])

    def _capture_trace(self, a_host, q_host, c_host, p_host, mask_host,
                        upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length, shape_key):
        dev = self.device
        a_p, q_p, c_p, p_p, mask_p = (self._persist(x) for x in (a_host, q_host, c_host, p_host, mask_host))
        for _ in range(2):  # warmup: compiles every kernel (capture disallows compilation)
            _ = self.run_device(a_p, q_p, c_p, p_p, mask_p, upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = self.run_device(a_p, q_p, c_p, p_p, mask_p, upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._trace_state = dict(id=tid, shape=shape_key, a=a_p, q=q_p, c=c_p, p=p_p, mask=mask_p, output=out)

    def _run_device_traced(self, a_host, q_host, c_host, p_host, indices,
                            upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length):
        """q_host/c_host/p_host/indices are the SAME tensor objects across the
        decoder's 2 recycle calls within one step (RFD3DiffusionModule._process_
        takes them from a fixed `kw`, only A_I/S_I are recomputed per recycle) --
        skip re-uploading (and re-deriving mask_host from indices) when the
        object identity hasn't changed since the last call. `a` always changes
        and is always refreshed. This is a perf-only cache: a step boundary
        (new identities) always falls through to a full refresh, so a wrong
        cache hit is impossible, only a missed skip."""
        import tt_bio.tenstorrent as _TTd
        if _TTd.trace_region_size() <= 0:
            raise ValueError(
                "CompactStreamingDecoder(trace=True) needs a device opened with a trace "
                "region; call get_device(trace_region_size=1 << 28) (or larger) first.")
        dev, dt = self.device, self.dtype
        shape_key = (tuple(a_host.shape), tuple(q_host.shape), tuple(c_host.shape), tuple(p_host.shape))
        step_key = (id(q_host), id(c_host), id(p_host), id(indices))
        st = self._trace_state
        if st is None or st["shape"] != shape_key:
            mask_host = _dense_attention_mask(indices)
            self._capture_trace(a_host, q_host, c_host, p_host, mask_host,
                                 upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length, shape_key)
            self._trace_state["step_key"] = step_key
        else:
            _tt_refresh(a_host, st["a"], dt)
            if st.get("step_key") != step_key:
                mask_host = _dense_attention_mask(indices)
                _tt_refresh(q_host, st["q"], dt)
                _tt_refresh(c_host, st["c"], dt)
                _tt_refresh(p_host, st["p"], dt)
                _tt_refresh(mask_host, st["mask"], dt)
                st["step_key"] = step_key
        ttnn.execute_trace(dev, self._trace_state["id"], cq_id=0, blocking=True)
        return _trace_output_copy(self._trace_state["output"])

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
            if self.trace:
                q = self._run_device_sparse_traced(
                    a_host, q_host, c_host, p_host, indices, upcast_mask_dev,
                    pack_idx_dev, unpack_idx_dev, valid, length,
                )
                a = _tt(a_host, dev, dt)
            else:
                p, n_keys, attn_idx_dev, dense_bias, gathered, block = _sparse_qk_inputs(
                    p_host, indices, dev, dt, mask_cache=self._mask_cache
                )
                sparse_qk = (n_keys, attn_idx_dev, dense_bias, gathered, block)
                a, q, c = (_tt(x, dev, dt) for x in (a_host, q_host, c_host))
                # The two recycle calls in a step share p and the neighbour index, so
                # each atom block's dense bias is bit-identical between them; build it
                # on the first call only (see GatedCrossAttention._sparse_bias_f32).
                q = self.run_device(
                    a, q, c, p, None, upcast_mask_dev, pack_idx_dev,
                    unpack_idx_dev, valid, length, sparse_qk=sparse_qk,
                    bias_cache=self._bias_cache,
                )
        elif self.trace:
            q = self._run_device_traced(a_host, q_host, c_host, p_host, indices,
                                         upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length)
            a = _tt(a_host, dev, dt)
        else:
            a = _tt(a_host, dev, dt)
            q = _tt(q_host, dev, dt)
            c = _tt(c_host, dev, dt)
            p = _tt(p_host, dev, dt)
            mask = _tt(_dense_attention_mask(indices), dev, dt)
            q = self.run_device(a, q, c, p, mask, upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length)

        q_grouped = self._pack_atoms_device(q, pack_idx_dev, valid)
        query = ttnn.unsqueeze(a, 2)
        down_mask = valid.unsqueeze(1)
        a_update = ttnn.squeeze(
            self.downcast.run_device(query, q_grouped, down_mask), 2
        )
        s = _tt(s_host, dev, dt)
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
    D_LL = torch.linalg.norm(R_L.unsqueeze(-2) - R_L.unsqueeze(-3), dim=-1)  # [B, N, N]
    lo, hi = min_dist / sigma_data, max_dist / sigma_data
    bins = torch.linspace(lo, hi, n_bins - 1, device=R_L.device)
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

    def _process_z_collapsed(self, Z_init_II, bins, bins_self, batch, I):
        """process_z in three device ops instead of six.

        The shipped route builds a [B,I,I,288] concat and a [B,I,I,258] slice of it so rms_norm
        can average over 258 columns, 130 of which are a one-hot: 15.14 ms/call at the page
        fixture's [1,685,685,*], 29.93 ms/step over the two recycles. This is 5.03
        (perf/p89/process_z.json). Not bit-exact -- one fp32 accumulation becomes two
        bf16-rounded halves, ~1 bf16 ulp -- so it is release-gated behind
        RFD3_PROCESS_Z_COLLAPSE and the shipped default is unchanged.
        """
        dev, n = self.device, self.N_BINS
        inv, ainv = self._process_z_invariant(Z_init_II, 1 if bins_self is None else 2, batch)
        tab = self._process_z_table(bins_self is not None)
        idx = _tt_idx(bins if bins_self is None else bins * n + bins_self, dev)
        em = ttnn.embedding(idx, tab, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        em = ttnn.to_layout(ttnn.reshape(em, (batch, I, I, self.C_Z)), ttnn.TILE_LAYOUT)
        scaled = ttnn.multiply(em, inv)
        ttnn.deallocate(em)
        out = ttnn.add(scaled, ainv)
        ttnn.deallocate(scaled)
        return out

    def __call__(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None):
        """R_L_ca: [B, I, 3] (scaled C-alpha positions), S_init_I: [B, I, c_s],
        Z_init_II: [I, I, c_z] or [1, I, I, c_z] (batch 1 -- replicated on device),
        D_II_self: int32 [B, I, I] distogram bin indices or None.
        Returns (S_I [B,I,c_s], Z_II [B,I,I,c_z]) on host.

        The host-boundary wrapper the component parity scripts call. The per-step path
        uses run_device instead and keeps z on the card -- see run_device."""
        s, z = self.run_device(R_L_ca, S_init_I, Z_init_II, D_II_self=D_II_self)
        return ttnn.to_torch(s).float(), ttnn.to_torch(z).float()

    def run_device(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None):
        """Same computation as __call__, returning the ttnn tensors instead of host copies.

        z is [B,I,I,c_z]: O(I^2) and by far the largest tensor the step moves. Handing it
        to the DiT as a device tensor removes one D->H untilize plus one H->D upload per
        recycle, four crossings of an I^2 tensor per step."""
        dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
        f32 = self.fp32_residual
        B, I = R_L_ca.shape[0], R_L_ca.shape[1]
        if S_init_I.ndim == 2:
            S_init_I = S_init_I.unsqueeze(0).expand(B, -1, -1).contiguous()
        s = _tt(S_init_I, dev, dt)
        if f32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
        for tr in self.transition_1:
            sc = ttnn.typecast(s, dt, memory_config=s.memory_config()) if f32 else s
            upd = tr(sc)
            s = ttnn.add(s, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config())) if f32 \
                else ttnn.add(s, upd)
        bins = _scaled_distogram_bins(R_L_ca, sigma_data=self.sigma_data, n_bins=self.N_BINS)
        if Z_init_II.ndim == 3:
            Z_init_II = Z_init_II.unsqueeze(0)
        if _PROCESS_Z_COLLAPSE:
            PZSTATS[0] += 1
            z = self._process_z_collapsed(Z_init_II, bins, D_II_self, B, I)
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
            zc = ttnn.typecast(z, dt, memory_config=z.memory_config()) if f32 else z
            upd = tr(zc)
            z = ttnn.add(z, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config())) if f32 \
                else ttnn.add(z, upd)
        for blk in self.pairformer_stack:
            s, z = blk(s, z)
        return s, z


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

        # One [c_pair, 32 * n_block] weight holding every block's pair-bias projection, block i
        # in columns [32*i, 32*i + N_HEAD) and zeros in the rest. Built straight from the torch
        # weights rather than through torch_to_tt, which advances the shared tiled-weight cache
        # counter and would shift every later weight's cache key.
        self.b_w_fused = None
        if _PAIRBIAS_FUSED:
            cols = torch.zeros(self.C_PAIR, _PAIRBIAS_SLOT * n_block)
            for i in range(n_block):
                w = self.weights[f"blocks.{i}.attention_pair_bias.to_b.weight"].t()
                cols[:, _PAIRBIAS_SLOT * i:_PAIRBIAS_SLOT * i + self.N_HEAD] = w.float()
            self.b_w_fused = ttnn.from_torch(cols, layout=ttnn.TILE_LAYOUT, device=self.device,
                                             dtype=self.dtype)

    def run_device(self, a, s, z, additive_mask):
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

    def __call__(self, a_host, s_host, z, indices):
        """z may be a host tensor (the parity-script path) or an already-resident ttnn
        tensor handed straight over by DiffusionTokenEncoder.run_device."""
        dev, dt = self.device, self.dtype
        a = _tt(a_host, dev, dt)
        s = _tt(s_host, dev, dt)
        if isinstance(z, ttnn.Tensor):
            if z.dtype != dt:
                z = ttnn.typecast(z, dt)
        else:
            z = _tt(z.unsqueeze(0) if z.ndim == 2 else z, dev, dt)
        mask = _tt(_dense_attention_mask(indices), dev, dt)
        return ttnn.to_torch(self.run_device(a, s, z, mask)).float()


def _default_compute_kernel_config():
    dev = get_device()
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
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


def _create_attention_indices(f, X_L, tok_idx, n_keys, n_seq_neighbours):
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
        # ttnn trace-capture lever (p25, opt-in via RFD3_TRACE_ENCODER=1 / RFD3_TRACE_DECODER=1).
        # BOTH ARE MEASURED NEGATIVE AND STAY DEFAULT-OFF. p32 ran the two-gate check p25-p28
        # never got to (scripts/rfd3_port/p32_trace_ab.py, all four combinations alternated in
        # one process on one hot card, 3359 atoms, 3 alternations): every leg is BIT-EXACT
        # (trajectory maxabs 0.0 against the eager leg -- the correctness gate passes), and
        # every leg is SLOWER. Decoder -12.3% (246.3 -> 280.8 ms/step), encoder -74.5%
        # (-> 967.5), both -75.5%.
        #
        # The reason is structural, not a tuning miss (p32_trace_attribution.py, same shape):
        # the decoder's `run_device` -- exactly the graph the trace replaces -- costs
        # 9.66 ms/step of HOST DISPATCH, 3.9% of the step. That is the ceiling on what tracing
        # it can ever save. Against it the traced path pays 26.5 ms/step refreshing its
        # persistent input buffers plus 6.0 ms/step of host pair gather, because a trace can
        # only read inputs at fixed addresses: the gathered pair features and the
        # head-replicated scatter index (20.7 of 22.8 MB per step) are precisely the two
        # tensors p26 had already made device-resident (_pair_gather_table,
        # _sparse_attn_index's device concat), and staging them for a trace pushes them back
        # across the host boundary. Tracing here undoes a residency win to buy a 3.9% one, so
        # the two levers are mutually exclusive rather than compounding -- and even free
        # staging would leave a wash, not a win. p25/p26's isolated 1.25x measured `run_device`
        # in a tight loop where those 9.66 ms were nearly all the wall time; in the real step
        # they are 3.9%. The encoder is worse for the same reason plus one more: its traced
        # form (_encoder_downcast_device) takes the DENSE pair-bias path, so it re-stages
        # P_LL ([1,L,L,16], 361 MB at 3359 atoms) and a dense [1,1,L,L] mask every step where
        # the eager sparse path moves a [1,L,128,16] gather (13.8 MB) off a resident table.
        #
        # Kept, not deleted: both paths are correct and bit-exact, the ceiling is
        # size-dependent (dispatch is a larger share of a 419-atom step), and the flags are the
        # only way to re-measure if ttnn ever grows a device-side way to stage a trace input.
        # Requires get_device(trace_region_size=1 << 28) or larger; nothing in production opens
        # the device that way, which is consistent with both flags being off.
        #
        # History: RFD3_TRACE_ENCODER's capture-time hang was root-caused in p27 (run_device's
        # internal host-mask upload hard-errors inside an open capture region; fixed by
        # precomputing the mask and folding encoder + _downcast_q into ONE trace, see
        # _encoder_downcast_traced). RFD3_TRACE_DECODER's own crash-after-one-step was
        # root-caused in p32 (see _trace_output_copy). self.encoder itself is always built with
        # trace=False -- production tracing of the encoder lives in _encoder_downcast_traced,
        # not in LocalAtomTransformer's own (isolated-test-only) trace mechanism.
        self._trace_encoder = env_flag("RFD3_TRACE_ENCODER", False)
        self._grouping_cache = {}      # batch -> {"valid", "pack_idx_dev", ...}, shared by downcast_c/downcast_q
        self._grouping_owner = None    # (id(tok_idx), shape) the cached slots belong to
        self._encoder_trace_state = None  # {"id", "shape", "q", "c", "p", "mask", "a", "s", "out_q", "out_a"}
        self.encoder = LocalAtomTransformer(self.scope("encoder"), ckc, n_blocks=3, dtype=dt,
                                            fp32_residual=self._dit_fp32_residual, trace=False)
        self.decoder = CompactStreamingDecoder(self.scope("decoder"), ckc, dtype=dt,
                                               fp32_residual=self._dit_fp32_residual,
                                               trace=env_flag("RFD3_TRACE_DECODER", False))
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

    def _grouping_buffers(self, tok_idx, batch):
        """Cache the atom<->token grouping index (pack_idx_dev/valid) by tok_idx identity,
        shared by _downcast_c and _downcast_q (both group by the SAME atom_to_token_map).

        tok_idx is the SAME python object for an entire design's sampling loop (RFD3Sampler
        threads one `f` dict by reference through every step), exactly like
        CompactStreamingDecoder._design_buffers -- so caching by id(tok_idx) means the
        pack_idx_dev upload happens ONCE per design, not once per step.

        p27 root cause of the encoder-trace hang (rfd3-trace-hang-vs-corruption-two-gate-catch):
        before this fix, _downcast_q re-derived AND RE-UPLOADED pack_idx_dev via
        ttnn.from_torch(..., device=dev) on every single call -- a fresh device allocation
        immediately after the encoder's persistent trace had just executed. Caching this
        (like the decoder already does) means that after the first call, NO allocation
        happens here at all -- only a python-side cache hit -- so nothing can race with an
        open trace region on steady-state steps. Combined with folding the remaining first-call
        interleaving into the encoder+downcast_q trace itself (_encoder_downcast_traced), this
        closed the hang: p32 ran the multi-step trajectory replay end-to-end and it is
        bit-exact (maxabs 0.0). The encoder trace is still default-off, on perf grounds
        rather than correctness -- see the RFD3DiffusionModule.__init__ comment."""
        dev = self.device
        # One slot PER BATCH SIZE, not one slot total: _downcast_c is called with the
        # batch-1 S_I while the encoder downcast is called with the batch-D A_I, so a
        # single-slot cache missed on every call at D>1 and re-uploaded the index --
        # 8.7 ms per call at 3359 atoms, 31x what eight batch-1 calls cost (p11).
        owner = (id(tok_idx), tuple(tok_idx.shape))
        if self._grouping_owner != owner:
            # a design change can change L/I shapes -- any captured combined trace is stale.
            self._grouping_owner, self._grouping_cache = owner, {}
            if self._encoder_trace_state is not None:
                ttnn.release_trace(dev, self._encoder_trace_state["id"])
                self._encoder_trace_state = None
        st = self._grouping_cache.get(batch)
        if st is None:
            valid, pack, _ = _grouping_indices(tok_idx, batch, dev)
            pack_idx_dev = _tt_idx(pack, dev)
            # Additive mask for the downcast GCAs (downcast_c AND downcast_q -- same
            # shape/content, mask is about which atom-slots are valid, independent of
            # which GCA's weights consume it): precomputed and persisted here, NOT passed
            # as a raw host `attn_mask` into run_device (which would upload it internally
            # on every call -- see _encoder_downcast_device / _downcast_q_device).
            downcast_mask_dev = self.downcast_q._prepare_additive_mask(
                valid.unsqueeze(1), batch, valid.shape[0], 1, valid.shape[1])
            st = dict(valid=valid, pack_idx_dev=pack_idx_dev, downcast_mask_dev=downcast_mask_dev)
            self._grouping_cache[batch] = st
        return st["valid"], st["pack_idx_dev"], st["downcast_mask_dev"]

    def _downcast_c(self, C_L, S_I, tok_idx):
        dev, dt = self.device, self.dtype
        if C_L.ndim == 2: C_L = C_L.unsqueeze(0)
        if S_I.ndim == 2: S_I = S_I.unsqueeze(0)
        B, I, _ = S_I.shape
        valid, pack_idx_dev, mask_dev = self._grouping_buffers(tok_idx, B)
        c_g = _pack_atoms_dev_core(_tt_cached(C_L, dev, dt), pack_idx_dev, valid)
        S_I_dev = _tt_cached(S_I, dev, dt)  # uploaded once, reused below (was 2x, p23 perf)
        q = ttnn.unsqueeze(S_I_dev, 2)
        upd = ttnn.squeeze(self.downcast_c.run_device(q, c_g, attn_mask_dev=mask_dev), 2)
        return ttnn.to_torch(ttnn.add(S_I_dev, upd)).float()

    def _downcast_q_device(self, q_g, a, s, mask_dev):
        """Pure-device tail of downcast_q (GCA + s-norm/linear), shared by the eager
        _downcast_q and the combined encoder+downcast_q trace (_encoder_downcast_device).
        mask_dev is the ALREADY-persisted additive mask from _grouping_buffers -- passed
        via attn_mask_dev so run_device never uploads a fresh mask itself (p27: doing so
        inside a trace capture region hard-errors with "Writes are not supported during
        trace capture", the actual root cause of the p26 encoder-trace hang)."""
        ckc, dt = self.compute_kernel_config, self.dtype
        upd = ttnn.squeeze(self.downcast_q.run_device(ttnn.unsqueeze(a, 2), q_g, attn_mask_dev=mask_dev), 2)
        a = ttnn.add(a, upd)
        s = ttnn.rms_norm(s, weight=self.downcast_q_s_n, epsilon=1e-6, compute_kernel_config=ckc)
        s = ttnn.linear(s, self.downcast_q_s_w, compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        return ttnn.add(a, s)

    def _downcast_q(self, Q_L, A_I, S_I, tok_idx):
        dev, dt = self.device, self.dtype
        B, I, _ = A_I.shape
        valid, pack_idx_dev, mask_dev = self._grouping_buffers(tok_idx, B)
        q_g = _pack_atoms_dev_core(_tt(Q_L, dev, dt), pack_idx_dev, valid)
        a = _tt(A_I, dev, dt)
        s = _tt(S_I, dev, dt)
        out = self._downcast_q_device(q_g, a, s, mask_dev)
        return ttnn.to_torch(out).float()

    def _encoder_downcast_device(self, q, c, p, mask, a, s, pack_idx_dev, downcast_mask_dev, valid):
        """Pure-device core for the combined encoder+downcast_q trace: encoder's 3 atom
        blocks, then downcast_q's pack/GCA/norm/linear, all in ONE captured region so no
        eager allocation OR host-mask upload ever runs while the trace is open (see
        _grouping_buffers and _encoder_downcast_traced for why the split version hung -- p27)."""
        q = self.encoder.run_device(q, c, p, mask)
        q_g = _pack_atoms_dev_core(q, pack_idx_dev, valid)
        a_out = self._downcast_q_device(q_g, a, s, downcast_mask_dev)
        return q, a_out

    def _persist_ed(self, x_host):
        host_t = _tt_host(x_host, self.dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, self.device)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    def _capture_encoder_downcast_trace(self, q_host, c_host, p_host, mask_host, a_host, s_host,
                                         pack_idx_dev, downcast_mask_dev, valid, shape_key):
        dev = self.device
        q_p, c_p, p_p, mask_p, a_p, s_p = (
            self._persist_ed(x) for x in (q_host, c_host, p_host, mask_host, a_host, s_host))
        for _ in range(2):  # warmup: compiles every kernel (capture disallows compilation)
            _ = self._encoder_downcast_device(q_p, c_p, p_p, mask_p, a_p, s_p, pack_idx_dev,
                                               downcast_mask_dev, valid)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out_q, out_a = self._encoder_downcast_device(q_p, c_p, p_p, mask_p, a_p, s_p, pack_idx_dev,
                                                       downcast_mask_dev, valid)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._encoder_trace_state = dict(id=tid, shape=shape_key, q=q_p, c=c_p, p=p_p, mask=mask_p,
                                          a=a_p, s=s_p, out_q=out_q, out_a=out_a)

    def _encoder_downcast_traced(self, Q_L_host, C_L_host, P_LL_host, indices, A_I_host, S_I_host,
                                  pack_idx_dev, downcast_mask_dev, valid):
        """RFD3_TRACE_ENCODER=1 production path: encoder + downcast_q as one combined ttnn
        trace (see _encoder_downcast_device). pack_idx_dev/downcast_mask_dev/valid come from
        the ALREADY-cached _grouping_buffers -- by the time this runs, no allocation happens
        for them here."""
        import tt_bio.tenstorrent as _TTd
        if _TTd.trace_region_size() <= 0:
            raise ValueError(
                "RFD3_TRACE_ENCODER=1 needs a device opened with a trace region; call "
                "get_device(trace_region_size=1 << 28) (or larger) first.")
        dev, dt = self.device, self.dtype
        shape_key = (tuple(Q_L_host.shape), tuple(C_L_host.shape), tuple(P_LL_host.shape),
                     tuple(A_I_host.shape), tuple(S_I_host.shape))
        mask_host = _dense_attention_mask(indices)
        st = self._encoder_trace_state
        if st is None or st["shape"] != shape_key:
            if st is not None:
                ttnn.release_trace(dev, st["id"])
            self._capture_encoder_downcast_trace(Q_L_host, C_L_host, P_LL_host, mask_host,
                                                  A_I_host, S_I_host, pack_idx_dev,
                                                  downcast_mask_dev, valid, shape_key)
        else:
            _tt_refresh(Q_L_host, st["q"], dt)
            _tt_refresh(C_L_host, st["c"], dt)
            _tt_refresh(P_LL_host, st["p"], dt)
            _tt_refresh(mask_host, st["mask"], dt)
            _tt_refresh(A_I_host, st["a"], dt)
            _tt_refresh(S_I_host, st["s"], dt)
        st = self._encoder_trace_state
        ttnn.execute_trace(dev, st["id"], cq_id=0, blocking=True)
        return ttnn.to_torch(st["out_q"]).float(), ttnn.to_torch(st["out_a"]).float()

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
        f["attn_indices"] = _create_attention_indices(f, X_noisy_L, tok_idx, self.N_ATTN_KEYS, self.N_ATTN_SEQ)
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
        if self._trace_encoder:
            B = A_I.shape[0] if A_I.ndim == 3 else 1
            valid, pack_idx_dev, downcast_mask_dev = self._grouping_buffers(tok_idx, B)
            P_LL_in = P_LL.unsqueeze(0) if P_LL.ndim == 2 else P_LL
            Q_L, A_I = self._encoder_downcast_traced(Q_L, C_L, P_LL_in, f["attn_indices"], A_I, S_I,
                                                      pack_idx_dev, downcast_mask_dev, valid)
        else:
            Q_L = self.encoder(Q_L, C_L, P_LL, indices=f["attn_indices"])
            A_I = self._downcast_q(Q_L, A_I, S_I, tok_idx)
        recycled = self._forward_with_recycle(
            n_recycle, X_noisy_L=X_noisy_L, R_L_uniform=R_L_uniform, t_L=t_L, f=f, Q_L=Q_L,
            C_L=C_L, P_LL=P_LL, A_I=A_I, S_I=S_I, Z_II=Z_II)
        return {"X_L": recycled["X_L"], "sequence_logits_I": recycled["sequence_logits_I"],
                "sequence_restype_I": recycled["sequence_restype_I"]}

    def _forward_with_recycle(self, n_recycle, **kw):
        n_recycle = n_recycle if n_recycle is not None else self.N_RECYCLE
        rec = {}
        for i in range(n_recycle):
            rec = self._process_(D_II_self=rec.get("D_II_self"), X_L_self=rec.get("X_L"), **kw)
        return rec

    def _process_(self, D_II_self, X_L_self, *, R_L_uniform, X_noisy_L, t_L, f, Q_L, C_L, P_LL, A_I, S_I, Z_II):
        is_ca = f["is_ca"]
        R_L_ca = R_L_uniform[..., is_ca, :]
        # z stays on the card between these two: it is [B,I,I,128], the only O(I^2)
        # tensor in the step, and the round trip through host fp32 was 58.8% of the step
        # at 3359 atoms against 13.9% at 419 (p19 step 0) -- the size-dependent cost the
        # GPU does not pay. s is O(I*c_s) and still crosses, because the decoder wants it
        # on host anyway.
        s_dev, z_dev = self.diffusion_token_encoder.run_device(R_L_ca, S_I, Z_II, D_II_self=D_II_self)
        S_I = ttnn.to_torch(s_dev).float()
        X_L_ca = X_noisy_L[..., is_ca, :] if X_L_self is None else X_L_self[..., is_ca, :]
        dit_idx = _create_attention_indices(f, X_L_ca, torch.arange(I := S_I.shape[1], device=X_L_ca.device),
                                            self.DIT_KEYS, self.DIT_SEQ)
        A_I = self.diffusion_transformer(A_I, S_I, z_dev, dit_idx)
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
        logits, aatype = self.sequence_head(A_I)
        ttnn.deallocate(A_I)
        D_II_self = _scaled_distogram_bins(X_out[..., is_ca, :].detach(), sigma_data=self.SIGMA_DATA, n_bins=65)
        return {"X_L": X_out, "D_II_self": D_II_self, "sequence_logits_I": logits,
                "sequence_restype_I": aatype}


def build_diffusion_module(state_dict, compute_kernel_config=None, dtype=None):
    compute_kernel_config = compute_kernel_config or _default_compute_kernel_config()
    return RFD3DiffusionModule(state_dict, compute_kernel_config, dtype=dtype)
