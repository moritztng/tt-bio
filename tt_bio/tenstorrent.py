import os
import sys
import contextlib
import torch, ttnn, atexit
from torch import nn
from typing import Callable, Mapping
from math import pi, prod
from functools import lru_cache
from types import MappingProxyType

from . import reblock_permute as _reblock
from . import triatt_qkv as _triatt_qkv
from . import triatt_sdpa as _triatt_sdpa
from . import trimul_tail as _trimul_tail
from . import mm_generic as _mm_generic

TRIANGLE_MULT_CHUNK_SIZE = 32
TRIANGLE_ATT_CHUNK_SIZE_FAST = 1024
TRIANGLE_ATT_CHUNK_SIZE = 512
OPM_CHUNK_SIZE = 256
MSA_CHUNK_SIZE = 512
# Chunk OuterProductMean's norm+projection stage once the MSA representation exceeds this.
# Same threshold and the same bit-exactness argument as protenix's MSA row chunking, so a
# target small enough to fit keeps the exact unchunked path. See OuterProductMean.__call__.
OPM_ROW_CHUNK_BUDGET_BYTES = 1 << 30      # 1.0 GiB
# Cap on OPM's per-I-block matmul result, which is (rows*c_a, c_b*tokens) in bf16 and therefore
# grows with the SQUARE of the token count at fixed `rows`. At OPM_CHUNK_SIZE=256 and 992 padded
# tokens that single tensor is 520093696 B -- which is exactly, to the byte, the allocation 9i3p
# dies on. So the row block has to be derived from the token width instead of being a constant.
# Numerically inert: the I axis indexes independent token rows (the matmul contracts depth, not I,
# and each block gets its own full depth accumulation), so regrouping rows cannot change a value.
# The blocked path this guards is entered at I > SEQ_LEN_MORE_CHUNKING, which reads as 1536 here but
# is retuned to ~640 on a small grid (_apply_grid_thresholds), so on Wormhole it is live from ~640
# tokens up -- don't conclude from the 1536 baseline that a 992-token target never reaches it.
# Verified on a Wormhole Galaxy: 9i3p's 520093696 B refusal is gone with this in place and still
# present without it, and 9d72 reproduces all 15 structure md5s bit-for-bit despite going from 3
# row blocks to 5. It does NOT make 9i3p fold -- the target then hits the pair representation
# (980*992*384*2) instead, which is a separate limit this constant has no bearing on.
OPM_Z_BUDGET_BYTES = 1 << 28              # 0.25 GiB
# Pair-tensor byte size above which a chunked path's row/channel blocks are assembled
# on the HOST instead of by ttnn.concat on device. The concat needs a fresh
# full-pair-tensor allocation while the input and every block are still live (k=3
# pair-tensor multiples), and on a 12 GiB Wormhole part whose address space the trunk
# has churned, a >~2 GiB (>=~180 MiB/bank) request is refused even with GiBs nominally
# free (measured: od_9i3p refused 1902x1920x384x2 = 2.61 GiB at 7.2 GiB used). Moving
# the blocks to the host as they are produced keeps at most one block on device, and
# the final upload of the assembled tensor happens when only the input is live, so the
# request always lands in a freshly vacated full-size hole. torch.cat is pure data
# movement -- bit-identical to ttnn.concat, same equivalence class as the row-blocked
# norms. 1.5 GiB matches the tri_att qkv byte cap: residue-scale pair tensors
# (<=0.9 GiB) keep the device concat, so normal targets are byte-identical.
CONCAT_HOST_BYTES = int(os.environ.get("TT_BIO_CONCAT_HOST_BYTES", 1536 * 2 ** 20))  # 1.5 GiB
TRANSITION_W_CHUNK_SIZE = 1024
SEQ_LEN_MORE_CHUNKING = 1536
# Row-block height for the trimul's row-local projections. One number so the input and output
# projections cannot drift apart; a tile multiple, so a block boundary never splits a tile and the
# blocks stay bit-exact against the whole-tensor result.
PAIR_ROW_BLOCK = 128
# Byte gate for row-blocking the trimul INPUT norm, which is a different question from the output
# projections' SEQ_LEN_MORE_CHUNKING gate and must not share it.
#
# Row-blocking the input norm costs 43 % per trimul call -- 664.6 ms against 464.8 ms at H=977,
# c_z=384, 25 reps, stdev under 1 ms (perf/trimul_kernel/inproj_shape_read.py) -- because the norm
# is recomputed once per channel group. It buys nothing unless the whole-tensor norm cannot actually
# be allocated, and that is a byte question, not a token-count one. Gating it on
# SEQ_LEN_MORE_CHUNKING instead cost 1.8 % at the panel median: that constant is rescaled to 608 on
# a small-grid Wormhole (see _apply_small_grid_budgets), so a 977-token refiner was row-blocking a
# 744 MB tensor that fits with room to spare.
#
# The threshold sits between the largest pair tensor that demonstrably fits and the one that gets
# refused: 9i3p's is 2.59 GiB and folds with the whole-tensor norm, 9j4c's is 3.19 GiB and is the
# refusal this exists for.
TRIMUL_IN_NORM_ROWBLOCK_BYTES = int(os.environ.get(
    "TT_BIO_TRIMUL_IN_NORM_ROWBLOCK_BYTES", 3 * 2 ** 30))  # 3 GiB
TRANSITION_BATCH_CHUNKING_THRESHOLD = 1024
TRANSITION_W_CHUNKING_THRESHOLD = 1024
# Per-chunk element budget for the 4D Transition row loop on a small grid, in place of the
# compounded channel ratio below it. That ratio divides by the channel twice -- once in
# `_ref / (w_eff * c)` and again in the small-grid `_ref * 128 // c` -- so the small-grid budget
# falls as 1/c^2 where Blackhole's falls as 1/c. Nothing intended the square: the comment on the
# shrink says "in proportion to the channel's excess over 128", and at c=384 it costs a factor of
# three. MEASURED on the 8x9 Wormhole Galaxy at OpenDDE's c_z=384, real Transition module, 2 warm
# + 5 timed, spread under 1.4 % (perf/wh-opendde/wh_transition_chunk.py):
#
#   W=512  h=3 (shipped) 72.36 ms | h=6 65.53 = 1.1035x | h=7 69.65 | h=8 68.32 | h=9 67.01 | h=10 CLASH
#   W=995  h=3 (shipped) 340.18   | h=6 308.45 = 1.1025x                                    | h=10 CLASH
#   W=320  h=5 (shipped) 27.50    | h=6 26.42          | h=10 25.49 = 1.0786x               | h=16 CLASH
#
# CLASH is a TT_THROW of statically allocated circular buffers against L1 buffers across the whole
# 8x9 grid, so the shrink is load-bearing and Blackhole's own chunk height genuinely does not fit
# here. Two facts the budget has to respect: the wall is NOT monotonic in the chunk height (h=6
# beats h=7/8/9 at W=512 even though all four fit), and the largest height that fits is therefore
# not the one to pick. 1179648 = 6 * 512 * 384 lands on the measured optimum at W=512 and inside
# the measured-fitting bracket at every other width the fold presents (W=320 -> 9, between the
# measured-fitting 6 and 10; W=640/995/1024 -> 6, measured).
#
# Scope, deliberately narrow. Only 256 < c <= 384 takes this path:
#   c=128 (boltz2, esmfold2) is already unshrunk and its shipped h=16 measured fastest at both
#     W=512 and W=1024, so there is nothing to win and it stays byte-identical.
#   c=256 (protenix-v2) measured h=12 = 1.074x at W=512, which is a real win, but protenix-v2 is
#     mid-repair on this machine in its own task and a shared default moved underneath another
#     model is how OpenDDE was regressed 60x once already. Left alone on purpose; the measurement
#     is in perf/wh-opendde/results/transition_chunk_c256_wh.json for whoever owns it.
#   c > 384 is not measured, so it keeps the shipped expression rather than extrapolating.
# The W/H guard keeps this to the regime where every arm was torch.equal: above the 608-token
# thresholds the row loop also W-chunks and the ragged tail block re-rounds with the row height
# (max abs diff 0.015625 on bf16 at W=640 and W=995). Raising it there is worth a further ~0.13 s
# at 512 aa and much more at 1024 aa, and it is NOT bit-exact -- flagged, not shipped.
# Blackhole never reaches any of this: the whole branch is behind _IS_SMALL_GRID.
SMALL_GRID_TRANSITION_ELEMS = int(os.environ.get("TT_BIO_SMALL_GRID_TRANSITION_ELEMS", 1179648))
SMALL_GRID_TRANSITION_MAX_C = 384
TRANSITION_H_CHUNK_SIZE_FAST = 32
TRANSITION_H_CHUNK_SIZE = 16
# Measured 1.87x at the protenix pair shape (microbench M4, W=320/c=256) but 32 clashes
# with in-block L1 pressure at MSA shapes (W=1024/c=128, test_msa[100-1000]) and at the
# opendde pair shape (W=320/c=384). Gate to the verified envelope only.
TRANSITION_H_CHUNK_SIZE_BIG = 32  # verified envelope: W<=384 (298-aa W=320). W=512 (protenix N=512 MSA/pair) clashes in-block L1 -> stays on 16
# The W bound above was validated against the c=256 clash. Per-chunk swiglu L1 scales with
# h_chunk * W * c, so c=64 and c=128 sit at a quarter and a half of that footprint and are
# bound by a number measured for a wider channel. Named so it can be measured rather than
# argued: default 384 keeps every shipped shape byte-identical.
TRANSITION_H_CHUNK_BIG_MAX_W = 384
# Measured ceiling for one Transition row chunk on a small grid, in L1 bytes PER CORE.
# The chunk's live L1 (x_norm + x_1 + x_2) is interleaved across the grid, so what binds is
# aggregate L1 / cores, and the budget above never sees core count. On UF-EV-A13-GWH02
# (8x9 = 72 cores, 1,466,080 B unreserved L1 per core) the fc1/fc2 matmuls fit at <= 384 KiB
# per core and throw a static-CB clash at >= 400 KiB: 14 points over 6 widths, no exceptions
# (perf/wh-protenix/wh_transition_h.py). Rescaled to a part's own L1 in _apply_grid_thresholds.
_TRANSITION_L1_CHUNK_BYTES_BASE = 393216
_WH_MEASURED_L1_PER_CORE = 1466080  # the L1 the base above was measured at
TRANSITION_L1_CHUNK_BYTES_PER_CORE = _TRANSITION_L1_CHUNK_BYTES_BASE

# A fused activation="silu" on Transition fc1 costs 174.0 us/call at the 298 aa pair shape, while the
# same silu as a standalone SFPU pass costs 83.7 -- measured on qb1 card 0, ttnn 0.67.4. The penalty
# is silu-specific and program-config-invariant: a fused relu costs +2.4 us and a fused gelu +141.3
# against its own 135.3 standalone, and the +174 holds across eight explicit
# MatmulMultiCoreReuseMultiCast configs. So unfusing pays a full L1 round trip and still wins,
# because the fused path runs silu at half the SFPU rate the standalone op reaches. Release-gated:
# the unfused form applies silu to the bf16-packed matmul output rather than to the fp32 dest
# accumulator, so it is not bit-exact.
_UNFUSED_SILU = os.environ.get("TT_BIO_UNFUSED_SILU", "0") == "1"
_FAST_MODE = False
_DTYPE_OVERRIDE = None
_DIFFUSION_FP32_DEVICE = False
# Release-gated (DEFAULT OFF): run the attention/triangle-attention SOFTMAX in fp32
# on device, matching the Boltz-2 reference's autocast-disabled fp32-softmax-then-
# cast-back-to-bf16 recipe (src/boltz/model/layers/attention.py:119-127 and
# triangular_attention/primitives.py:127-194). Operands and storage stay bf16; only
# the softmax reduction (and the additive bias it consumes) upcast to fp32. The
# q@k score matmul already uses fp32_dest_acc, so per memory
# boltz-reference-selective-fp32-softmax the softmax is the remaining mismatch. Set
# BOLTZ2_FP32_SOFTMAX=1 to A/B; default OFF until a leg closes against it.
_FP32_SOFTMAX = os.environ.get("BOLTZ2_FP32_SOFTMAX", "0") == "1"
# Benchmark-only escape hatch: compare the pre-decomposition channel moves.
_TRIMUL_RAW_CHANNEL_MOVES = False
# The trimul channel move's hand-written kernel is the fifth knob in this class and its constant
# lives with the code it gates, `reblock_permute.REBLOCK_PERMUTE`, because that module owns the
# shape window `eligible()` measures and importing this one back would be circular. Bit-exact.
# See `_channel_move` below and state/protenix-trunk--y-permute-flip.md.
# KILLED AT FOLD LEVEL, kept only as the A/B toggle. ttnn.experimental.minimal_matmul for the
# two trimul output projections is 1.117x per trimul and 1.0384x on the Pairformer block, but it
# is not bit-exact, and at 298 aa it moves the folded structure by 4.05 A all-atom RMSD against a
# run-to-run noise floor of exactly 0.0 (perf/trimul_kernel/w2_fold_parity.py). That is a
# different fold, not a rounding difference. See _trimul_out_proj.
_TRIMUL_MM_OUT = False
# The trimul in-projection's N is 4 * the channel-chunk width, and above _trimul_l1_max_seq that
# width is pinned to the narrowest value by _trimul_chunk_size -- on a path where the chunks are in
# DRAM and the L1 budget the narrowing protects does not exist. At 512 aa the same 134.2 MB
# activation is therefore streamed from DRAM once per chunk, 1024 MB read to write 512 MB, and
# grouping G chunks into one matmul removes (G-1)/G of that read, bit-exact (torch.equal, max_abs
# 0.0, 2.03x on the matmul: perf/bigswing/trimul_inproj_width.py).
#
# The pair-major column order made that a losing trade, because the loop consumes four [1,S,S,C]
# pieces and a pair-major group needs a 4G-way ttnn.chunk whose pieces are one tile wide: 96 GB/s
# against 166-335 for pieces of 4-8 tiles (perf/bigswing/chunk_width_rate.py), 11.736 -> 13.208 ms
# per trimul at G=8, 0.9724x at the fold at G=4 (perf/bigswing/fold_group_512_qb2c0.json).
#
# _gp_in_chunks now orders the columns role-major, so the split is 4-way at every G and each piece
# is G tiles wide. Measured at 512 aa on qb2 card 0 (perf/trimul_root/): the in-projection unit
# (matmul + the split the loop consumes) is 11.736 -> 5.658 ms per trimul at G=8, 2.07x, and the
# whole trimul is 28.483 -> 22.996 ms, 1.239x, torch.equal against G=1 at G=2/4/8 on both the
# starting and the ending variant. Every downstream op is elementwise, an index move or a
# per-channel matmul, so the width is a partition of the same sum.
# 12 is opendde's whole channel loop in one iteration (trimul hidden 384 / chunk 32). It is a CAP,
# not a width: the search below takes the largest DIVISOR of the model's own channel loop that fits
# the byte budget, so n_pairs=4 models (protenix-v2, boltz2, openfold3, esmfold2) still get 4, byte
# for byte what they got at cap 8. Measured at opendde's own 512 aa shapes (perf/odde512/screen3.json,
# qb2 card 3): the channel loop is 25.4418 ms/call at G=4 over three iterations and 22.3395 ms at
# G=12 in one.
_TRIMUL_INPROJ_GROUP = 12
# Widest fused in-projection output the DRAM path may build, in bytes. The group width is what makes
# the fused projection pay, and it is also the module's whole DRAM-peak risk: at 512 aa the output is
# 512 MiB at G=8 against 64 MiB at G=1, and it grows with N^2, so a constant group would ask for
# 1.9 GiB at 9i3p (973 aa) and 2.6 GiB at 9j4c (1136 aa) while those targets are already the ones
# closest to a refusal. Keyed on the allocation's own size rather than on the sequence length,
# because the sequence length is not what does not fit -- that is the mistake
# `_triangle_mul_memory_config`'s threshold makes.
#
# 1 GiB is measured, not chosen: perf/trimul_abs/cap_sweep.py runs the real module at N = 512 to
# 1136 with 6 GiB of foreign DRAM held (what a 9j4c fold has live, state/capacity_9j4c_dram2.log)
# and every width allocates, so the budget is not a fit/no-fit boundary -- it is a footprint cap.
# The extra DRAM peak a fused projection costs is measured at ~2x its own size, so 1 GiB holds the
# worst case at +1.448 GiB (704 aa, G=8: 8.377 vs 6.929 GiB) and leaves >20 GiB free at every size.
# It keeps the full width at 512-704 aa, takes G=4 at 973 aa (9i3p) and G=2 at 1136 aa (9j4c).
_TRIMUL_INPROJ_FUSED_BYTES = 1024 * 2 ** 20


def _trimul_inproj_group(seq_len: int, chunk: int, batch: int, n_pairs: int) -> int:
    """How many channel chunks the in-projection matmul may fuse at this shape.

    The largest divisor of the channel loop at or below `_TRIMUL_INPROJ_GROUP` whose fused output
    fits the byte budget. This used to halve down from the cap, which cannot reach a 12-pair loop at
    all -- 12 -> 6 -> 3 -> 1 skips every divisor of 12 below 12 itself -- so opendde ran three
    iterations where one would do. Every width is bit-exact against every other: the group is a
    partition of an independent-channel sum and everything below the four-way unpack is elementwise,
    an index move or a per-channel matmul (`torch.equal` at G=2/4/8, perf/trimul_root/).
    Bytes are priced at bf16 even when `_dtype()` is bfloat8_b, so the budget is a bound.
    """
    fused = 4 * chunk * seq_len * seq_len * batch * 2
    for g in range(min(n_pairs, _TRIMUL_INPROJ_GROUP), 1, -1):
        if n_pairs % g == 0 and g * fused <= _TRIMUL_INPROJ_FUSED_BYTES:
            return g
    return 1
# Widest inner K block the pair-track projection config may use; None disables the config.
# 1 keeps the contraction order of the production call and is bit-exact. Above 1 the partial
# sums fold through packer_l1_acc in K-block order instead, which moves the last bf16 bit and
# is NOT bit-exact. 16 lets c_z=256 reach in0_block_w=8 and c_z=384 reach 12, i.e. the whole
# contraction in one block for both: 1.037x on a 298 aa protenix-v2 fold and 1.012x on
# opendde (perf/inblockw/qb1/). Both gates pass; opendde moves 5.54 A from the bw=1 arm but
# moves TOWARD its reference on every metric, off a main that is already outside its floor.
# Release-gated. See state/perfwar-inblockw-qb1-land.md.
_PAIR_PROJ_BW: int | None = 16
# Same knob for the NARROW-output members of the pair-track projection class: the per-head
# PairWeightedAveraging z->bias projection ([1,L,L,c_z] @ [c_z,1]) and the template z
# projection ([c_z,64]). `ttnn.linear(core_grid=)` engages ~16 of 110 cores on both -- its
# core ladder is flat from 16 cores to 110 -- because a one-tile-wide output leaves it
# in0_block_w=1 and out_block_h=per_core_M. 1 keeps the production contraction order and is
# `torch.equal`; above 1 the partials fold through packer_l1_acc in K-block order and it is
# NOT bit-exact, the same parity class as _PAIR_PROJ_BW above. Kept separate from it because
# these two sites are a separate parity decision. Measured on qb1 card 1 at the fold's own
# [1,298,320,256]: 1.15x / 1.23x at 1, 1.98x / 2.08x at 16 (perf/p3narrow/).
_NARROW_PROJ_BW: int | None = 1
# The pair-track output projection's 48.82 MB result never leaves the device: the trimul's
# multiply_ and the Pairformer layer's residual add_ read it back immediately. Writing it to L1
# instead of DRAM removes the projection's write AND the consumer's operand read, and the trunk
# has the room -- every one of the 130 banks reads fully free at all 35 timed matmul classes in a
# live 298 aa fold, so the capacity objection that used to rule this out is not there. Bit-exact:
# a memory config decides where the writer puts a tile, not the order the contraction accumulates.
# Worth 430 ms/fold at 298 aa, of which 133 is the residual add no longer fetching its operand from
# DRAM; the projections themselves do not get faster. In-fold op walls against three baseline
# folds, perf/p3l1/ops_*.json.
_PAIR_PROJ_L1_OUT = True
# in0_block_w cap for the L1-output members. It must track _PAIR_PROJ_BW: at the same cap the L1
# output is `torch.equal` against the DRAM output of the identical config (max abs 0.0, and a live
# 298 aa fold returns the same plDDT to six decimals), so moving the destination is free of any
# parity decision. Dropping it to 1 was measured and REJECTED: it buys bit-exactness against the
# untuned core_grid reference, but it costs 41-115 ms/fold at the projections and turns a
# 430 ms/fold win into 33-106 (perf/p3l1/ops_*.json). Kept as the A/B toggle, not as a choice.
_PAIR_PROJ_L1_BW: int | None = 16
# Read-bound narrow projections take their SOURCE from L1 instead of their destination:
# AttentionPairBias's z->bias reads the whole 48.82 MB layer-normed pair tensor to write 6.10 MB,
# so the write was never the cost. Handing it an L1-resident layer_norm output takes the
# projection from 450.3 to 137.0 us at [1,298,320,256] @ [256,16] and is torch.equal.
_PAIR_BIAS_L1_NORM = True
# The same SOURCE lever at the two sites whose layer_norm feeds SEVERAL narrow projections:
# PairWeightedAveraging's per-head z->bias (one norm, eight [c_z,1] heads) and the template
# embedder's z projection (one norm, nt=4 [c_z,64] templates). Each consumer independently stops
# reading the whole 48.82 MB pair tensor from DRAM, so the read saving is per projection; the
# norm's own removed write is paid once per region, not once per projection. Measured on qb2
# chip 2 at 0.68.0: the PWA region 3572.2 -> 991.0 us and the template region 2180.7 -> 853.1,
# both `torch.equal`. perf/p3l1s068/.
# MEASURED: the two sites need separate toggles, because the L1 residency WINDOW differs even
# though the shape does not. At PWA the eight consumers are the first op of each head's chain. At
# the template embedder the nt consumers are separated by two whole PairformerLayer executions, so
# a naive L1 norm holds 48.82 MB for the entire template loop and the trimul inside those blocks
# THROWS ("statically allocated circular buffers in program 173 clash with L1 buffers", L1 buffer
# at 905216, static CB region ending at 1159680). `_template` gathers its projections above the
# block loop so the residency window is the projections only.
_PWA_L1_NORM = True
_TEMPLATE_L1_NORM = True

# Matmul fidelity for the trunk. The FPU is a 5b x 7b multiplier: srcA contributes a hidden bit plus
# 4 mantissa bits per pass and srcB a hidden bit plus 6, so a 16-bit float's 8-bit significand splits
# 5+3 on srcA and 7+1 on srcB (tt-metal tech_reports/matrix_engine/matrix_engine.md, "Math Fidelity").
# The only term HiFi4 adds over HiFi3 is therefore A_lo*B_lo, ~2^-12 relative, which is below the
# bf16 output's own 2^-8 rounding step -- and it costs a fourth pass, 64 against 48 cycles per tile.
# MEASURED at seven production shapes at N=512 against an fp32 reference computed from the same bf16
# operands (perf/bigswing/fid_512_mm_qb2c0.json): HiFi3 costs 0.08% of relative RMS where storing the
# result in bf16 already costs 1.8%, and is worth 1.071x time-weighted over 831.2 of the Pairformer's
# 872.95 executed TFLOP at 512 aa. HiFi2 is a different question: 2.59x the relative RMS, genuinely
# above the rounding floor, so it is a parity decision and not a free one.
# Deliberately NOT global. fp32 operands do need four passes, and Protenix's diffusion runs fp32 on
# purpose (PROTENIX_DIFFUSION_FP32_DEVICE, and memory af3-diffusion-sampler-selective-fp32-boundary),
# so the setting is scoped to the trunk, whose operands are bf16 and bf8 under --fast. Other models
# opt in at their own construction site after their own envelope run -- a shared default across five
# models is the shape that cost OpenDDE 60x once already.
# Default hifi4 = production unchanged; the A/B and the envelope gate flip it.
_TRUNK_MATH_FIDELITY = os.environ.get("TT_BIO_TRUNK_MATH_FIDELITY", "hifi4").lower()
_MATH_FIDELITIES = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi3": "HiFi3", "hifi4": "HiFi4"}


def trunk_compute_kernel_config(base):
    """`base` with the trunk's matmul fidelity, as a distinct object.

    Distinct on purpose: every trunk submodule holds this one reference, so an A/B arm is a single
    in-place write to `model.trunk.compute_kernel_config.math_fidelity` that provably cannot reach
    the diffusion or confidence stages.
    """
    if _TRUNK_MATH_FIDELITY not in _MATH_FIDELITIES:
        raise ValueError(f"TT_BIO_TRUNK_MATH_FIDELITY must be one of {sorted(_MATH_FIDELITIES)}, "
                         f"got {_TRUNK_MATH_FIDELITY!r}")
    cfg = type(base)(
        math_fidelity=getattr(ttnn.MathFidelity, _MATH_FIDELITIES[_TRUNK_MATH_FIDELITY]),
        math_approx_mode=base.math_approx_mode,
        fp32_dest_acc_en=base.fp32_dest_acc_en,
        packer_l1_acc=base.packer_l1_acc,
    )
    # The constructor takes four of the six fields; carry the other two rather than re-defaulting them.
    cfg.dst_full_sync_en = base.dst_full_sync_en
    cfg.throttle_level = base.throttle_level
    return cfg
# MEASURED LOSS, kept only as the A/B toggle behind perf/trimul_kernel/w2_arms.py.
# Letting the output channel move write straight to DRAM drops the separate clone that used
# to move the chunk there, but it also moves that permute's forced 64-byte writes from L1 to
# DRAM, and that costs more than the clone saves: 7.122 -> 7.431 (start) / 7.863 (end) ms per
# trimul at 298 aa. Bit-exact either way, and still a loss.
_TRIMUL_OUT_MOVE_DRAM = False
TRIANGLE_MULT_L1_MAX_SEQ_FAST = 640
TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10 = 704
TRIANGLE_MULT_L1_MAX_SEQ = 352
# Largest batch * chunk * seq_len^2 whose trimul working set still fits in L1 beside the
# triangle matmul's circular buffers. Everything the chunk loop holds there scales with that
# product, so one budget covers all of it. Measured on a 13x10 Blackhole grid at batch 1:
# chunk 64 at seq 320 and chunk 128 at seq 224 both fit; chunk 64 at seq 352 and chunk 128 at
# seq 256 both throw "statically allocated circular buffers clash with L1 buffers". Scaled by
# core count below, since on a smaller grid the same bytes land on fewer cores.
TRIANGLE_MULT_L1_CHUNK_BUDGET = 64 * 320 * 320
# Set by _apply_grid_thresholds: True on grids smaller than 11x10 (e.g. Wormhole).
# Tightens the L1-edge chunking thresholds and chunk sizes above this comment block.
_IS_SMALL_GRID = False

# Per-process record of trimul chunk widths that threw an L1/circular-buffer clash at
# program creation, keyed by call shape. The budget above was measured on a 130-core
# p150a; on a 110-core Blackhole (p300/p300c) the in-projection's static circular
# buffers take more per core and the budget admits widths that do not fit beside the
# pair tensors live at the call site (issue #11: a 140-token protein+ligand dies in
# the MSA stack's pair layer at the width the budget picks, 256, and even at 128).
# What the budget cannot see is that the live set differs by call site, so no single
# measured constant separates fit from clash on every grid; the clash itself can.
# It throws at program validation, before any kernel runs, so catching it and
# re-running the channel loop at a narrower width is safe, and narrowing is bit-exact:
# the width is a partition of an independent-channel sum (see `_trimul_chunk_size`).
# Value: the smallest width known to clash for the key.
_TRIMUL_CHUNK_CLASH: dict = {}
# Test-only (scripts/release_gate.py's l1-budget leg): start the channel loop at this width
# instead of the one the budget picks, so a fold can be pinned to the width a clash would
# narrow it to and the two runs compared byte for byte. Unset in production. This is the
# knob the l1-budget leg needs and a per-core-L1 override is not: on the part that broke
# issue #11 the per-core unreserved L1 is 1,532,416 B, the same as a p150a's — what differs
# is core count, and TT_BIO_FORCE_GRID already forces that.
_TRIMUL_CHUNK_CAP = int(os.environ.get("TT_BIO_TRIMUL_CHUNK_CAP", "") or 0)

# Seq lengths whose trimul does not fit in L1 even at the minimum width take the DRAM
# path instead: same ops, same arithmetic, the residency threshold's other side.
_TRIMUL_DRAM_SHAPES: set = set()


def _trimul_chunk_key(seq_len: int, hidden: int, batch: int) -> tuple:
    return (int(seq_len), int(hidden), int(batch), bool(_FAST_MODE),
            tuple(COMPUTE_GRID_MAIN))


def _record_trimul_clash(seq_len: int, hidden: int, batch: int, width: int) -> None:
    key = _trimul_chunk_key(seq_len, hidden, batch)
    prev = _TRIMUL_CHUNK_CLASH.get(key)
    if prev is None or width < prev:
        _TRIMUL_CHUNK_CLASH[key] = width

# Wormhole 8x9 re-fit of the two trimul constants above. `_apply_grid_thresholds` derives its
# small-grid values by scaling the Blackhole ones -- the residency threshold by per-core L1 (which
# fell 7 %) and the chunk budget by core count (which fell 45 %) -- and neither scaling has ever
# been measured on a 72-core grid. These two knobs re-fit them from measurement instead. Both are
# read ONLY when `_IS_SMALL_GRID` is True, so on a 110- or 130-core Blackhole every expression
# below is byte-for-byte the one that ships today. 0 / 1.0 keep the derived value.
# The chunk width is a partition of an independent-channel sum, so every width is bit-exact
# (see `_trimul_chunk_size`); the residency threshold picks a memory config, not an arithmetic.
# Measured on 8x9: state/wh-perf-esmfold2.md.
SMALL_GRID_TRIMUL_L1_MAX_SEQ = 0
SMALL_GRID_TRIMUL_BUDGET_SCALE = 1.0


def set_small_grid_trimul_l1_max_seq(seq: int) -> int:
    """A/B switch for the small-grid trimul L1 residency threshold. Returns the previous value.
    Inert on Blackhole: read only when `_IS_SMALL_GRID`."""
    global SMALL_GRID_TRIMUL_L1_MAX_SEQ
    prev, SMALL_GRID_TRIMUL_L1_MAX_SEQ = SMALL_GRID_TRIMUL_L1_MAX_SEQ, int(seq)
    return prev


def set_small_grid_trimul_budget_scale(scale: float) -> float:
    """A/B switch for the small-grid trimul chunk-width budget. Returns the previous value.
    Inert on Blackhole: read only when `_IS_SMALL_GRID`."""
    global SMALL_GRID_TRIMUL_BUDGET_SCALE
    prev, SMALL_GRID_TRIMUL_BUDGET_SCALE = SMALL_GRID_TRIMUL_BUDGET_SCALE, float(scale)
    return prev
SDPA_CHUNK_TILE = 32
SDPA_CHUNK_MAX = 256
# Tiling for row-independent blocks so their activations fit the 12 GB/chip DRAM
# on small grids (Wormhole) while the 6B weights stay resident (no reload — batch
# throughput preserved). 0 = single pass (Blackhole — ample DRAM). Bit-exact
# (independent rows over dim=1). Two regimes, set by _apply_grid_thresholds:
#   SMALL_GRID_SEQ_TILE  — per-TOKEN blocks (ESMC FFN on [B,L,d]): transient ~ rows
#     (no L factor), so a fixed row count bounds it.
#   SMALL_GRID_PAIR_TILE_AREA — PAIR blocks (ESMFold2 transition / OPM on [B,L,L,c]):
#     transient ~ rows*L, so bound rows*L by an area budget -> rows shrink as L
#     grows, keeping the transient flat (else a fixed row count is GBs at L=1024).
#   SMALL_GRID_MSA_TILE_AREA — MSA blocks (ESMFold2 MSA encoder on [B,L,M,c]):
#     transient ~ rows*M, so bound rows*M by an area budget -> rows shrink as the
#     MSA deepens. Without it a 1024 aa fold at M=8192 wants 2 GiB per [B,L,M,c]
#     tensor and ~8x that across one encoder block.
SMALL_GRID_SEQ_TILE = 0
SMALL_GRID_PAIR_TILE_AREA = 0
SMALL_GRID_MSA_TILE_AREA = 0

# Small-grid chunk budgets are calibrated for a full Wormhole B0 core (~1.5 MiB
# L1) and scaled to the device's *actual* per-core unreserved L1 — a part that
# reserves more per core (e.g. this Galaxy's meshed-fabric infrastructure leaves
# ~1.4 MiB/core) gets proportionally tighter chunks so the circular buffers keep
# their margin. Calibration ceiling, so a full-L1 part is unchanged.
_WH_FULL_L1_PER_CORE = 1572864  # 1.5 MiB
_MIN_L1_SCALE = 0.7             # floor: keep chunks workable on a very tight part

PAIRFORMER_PAD_MULTIPLE = 64  # Pad token dim to this multiple to avoid kernel recompilation
MSA_PAD_MULTIPLE = 1024  # Pad MSA dim to this multiple to avoid kernel recompilation
# Upper bound on heavy atoms per token for PROTEIN residues (Trp=14); ties the atom
# bucket to the seq_len bucket. Nucleotide tokens carry more (up to 23), so a DNA/RNA
# target can exceed padded_seq * 14 — _populate_diffusion_cache extends the bucket to
# cover the real atom count in that case instead of asserting.
MAX_ATOMS_PER_TOKEN = 14

ATOM_WINDOW = 32
ATOM_DIM = 128
ATOM_N_HEADS = 4
ATOM_N_LAYERS = 3
TOKEN_DIM = 2 * 384
TOKEN_N_HEADS = 16
TOKEN_N_LAYERS = 24

COMPUTE_GRID_X_11 = 11
COMPUTE_GRID_X_13 = 13
COMPUTE_GRID_Y = 10

CORE_GRID_MAIN = ttnn.CoreGrid(y=COMPUTE_GRID_Y, x=COMPUTE_GRID_X_11)
COMPUTE_GRID_MAIN = (CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)

def _dtype(default=None):
    # Call sites that were hardcoded ttnn.bfloat16 before the fp32-affinity gate pass
    # their former constant as `default`: fast mode must NOT silently demote stored
    # weights/projections to bfloat8_b (regressed esmfold2 confidence to NaN on WH).
    if _DTYPE_OVERRIDE is not None:
        return _DTYPE_OVERRIDE
    if default is not None:
        return default
    return ttnn.bfloat8_b if _FAST_MODE else ttnn.bfloat16


def _no_host_pad(x: ttnn.Tensor, dtype, n: int, n_pad: int) -> ttnn.Tensor | None:
    """The device-side result of padding ``x`` from ``n`` to ``n_pad`` and casting it to
    ``dtype``, or None when a real host pad is needed.

    The openfold3 diffusion pad helpers round-trip through host torch. At 512 aa there is
    nothing to pad (``n_pad == n``), and what the round trip actually does is untilize,
    widen bf16 to fp32 and re-tilize 67 MB on one host thread -- 94.3 ms per diffusion step
    against 0.7 ms for the same bytes on device. bf16 -> fp32 is exact in both directions,
    so ``ttnn.typecast`` reproduces it bit for bit; ``n_pad == n`` also means the dimension
    is tile-aligned, so there is no tile padding for the round trip to have zeroed.

    Returns ``x`` itself when even the cast is unnecessary, so every caller must guard its
    ``ttnn.deallocate`` of the input with ``is not``.
    """
    if (n_pad != n or x.layout != ttnn.TILE_LAYOUT
            or x.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        return None
    return x if x.dtype == dtype else ttnn.typecast(x, dtype)


ADALN_S_HOIST = True      # hoist the AdaLN conditioning half out of the diffusion rollout


def _cached(cache, key, make):
    """``make()``, reused across calls while ``cache`` is a live dict.

    The openfold3 diffusion rollout re-runs conditioning -> encoder -> DiT 200 times, but
    everything that is not a function of the noise level ``t`` or of the noisy coordinates
    is identical on every step. Sites producing such a value wrap it here; the sampler owns
    one dict per rollout and frees it at the end, so a cached tensor must never be
    deallocated by its consumer. ``cache is None`` runs the uncached path unchanged.
    """
    if cache is None:
        return make()
    v = cache.get(key)
    if v is None:
        v = make()
        cache[key] = v
    return v


# Kill switch so a fold-level A/B can run both arms without a checkout. Bit-exact either way --
# see TrunkModule._apply_template_noop.
_TEMPLATE_NOOP_GATE = os.environ.get("TT_BIO_TEMPLATE_NOOP_GATE", "1") != "0"


def _adaln_memory_config(atom_level: bool, large_seq_len: bool) -> ttnn.MemoryConfig | None:
    if not atom_level:
        return None
    return ttnn.DRAM_MEMORY_CONFIG if large_seq_len else ttnn.L1_MEMORY_CONFIG


def _trimul_l1_max_seq() -> int:
    """Longest sequence whose trimul chunks still live in L1."""
    if _FAST_MODE:
        if _IS_SMALL_GRID and SMALL_GRID_TRIMUL_L1_MAX_SEQ:
            return SMALL_GRID_TRIMUL_L1_MAX_SEQ
        return (
            TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10
            if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
            else TRIANGLE_MULT_L1_MAX_SEQ_FAST
        )
    return TRIANGLE_MULT_L1_MAX_SEQ


def _triangle_mul_memory_config(seq_len: int) -> ttnn.MemoryConfig:
    if seq_len in _TRIMUL_DRAM_SHAPES:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG if seq_len <= _trimul_l1_max_seq() else ttnn.DRAM_MEMORY_CONFIG


def _trimul_chunk_size(seq_len: int, hidden: int, batch: int = 1) -> int:
    """Hidden-channel chunk width for the trimul at this sequence length and batch.

    The chunk loop is bound by per-op overhead at production sizes rather than by
    arithmetic: at 117 aa the two trimuls are 52% of a Pairformer block and their matmuls
    are under half of that, the rest being one fused input matmul, a 4-way split, three
    channel moves and a concat per chunk. Widening the chunk removes those per-chunk ops
    without touching any arithmetic, because channels are independent in the triangle
    product and a different chunking is just a different partition of the same sum
    (verified bit-exact at 32 / 64 / 128 on real layer-0 weights,
    perf/trunk_layout/trimul_chunk_ab.py). Measured on a p150a at 117 aa: 8.29 -> 7.08 ms
    per Pairformer block.

    Only the L1 path widens. Above _trimul_l1_max_seq the chunks live in DRAM, where the
    same op-count saving comes with a larger live footprint, and the large targets that
    run there are the ones already sitting on the DRAM ceiling.

    `batch` is the leading dimension of the pair tensor, and it is part of the budget: every
    tensor the chunk loop keeps in L1 is [batch, chunk, seq, seq], so a batched caller holds
    `batch` times the bytes at the same width. ESMFold2's confidence head replicates the pair
    state to one copy per diffusion sample before its own trimul trunk, so it arrives here at
    batch = --diffusion_samples; pricing only chunk * seq_len^2 widened a 117 aa fold at 5
    samples from chunk 32 to 128 and the channel loop's input matmul then threw the
    circular-buffer clash. Narrowing instead of widening is free of any parity decision --
    the chunk width is a partition of an independent-channel sum, bit-exact at every width.
    """
    if seq_len > _trimul_l1_max_seq():
        return TRIANGLE_MULT_CHUNK_SIZE
    gx, gy = COMPUTE_GRID_MAIN
    budget = TRIANGLE_MULT_L1_CHUNK_BUDGET * gx * gy / (COMPUTE_GRID_X_13 * 10)
    if _IS_SMALL_GRID:
        budget *= SMALL_GRID_TRIMUL_BUDGET_SCALE
    c = TRIANGLE_MULT_CHUNK_SIZE
    # Price the chunk on the width it actually occupies. The chunk tensors are
    # [batch, chunk, seq, seq] TILE tensors, so both seq dims round up to 32 and a logical
    # seq understates the real footprint by (tile(seq)/seq)^2 -- 29% at 225 aa, where 256^2
    # against 225^2 is the difference between one doubling and none. That is what still
    # crashed the confidence Pairformer's trimul (minimal_matmul, the `gp_in_fused` below)
    # after the Transition's own tile-padding fix landed: at 225 aa the logical arithmetic
    # bought chunk 64 for a footprint of 4,194,304 against a 3,629,908 budget, while 205 aa
    # sits at 3,211,264 padded and folds. Small grid only, so Blackhole keeps its measured
    # widths byte-for-byte. Narrowing is bit-exact -- the chunk width is a partition of an
    # independent-channel sum, as the note above says -- so this cannot move an output.
    _sq = (-(-int(seq_len) // 32) * 32) if _IS_SMALL_GRID else seq_len
    while hidden % (c * 2) == 0 and batch * (c * 2) * _sq * _sq <= budget:
        c *= 2
    while _TRIMUL_CHUNK_CAP and c > _TRIMUL_CHUNK_CAP and c > TRIANGLE_MULT_CHUNK_SIZE:
        c //= 2
    # Clamp below any width already seen to clash at this exact call shape (issue
    # 11): the budget is a 130-core calibration, and a tighter grid learns its
    # ceiling from the clash itself. Monotonic: a narrower chunk always holds less
    # L1, so below a recorded clash is the safe side.
    failed = _TRIMUL_CHUNK_CLASH.get(_trimul_chunk_key(seq_len, hidden, batch))
    while failed is not None and c >= failed and c > TRIANGLE_MULT_CHUNK_SIZE:
        c //= 2
    return c


@lru_cache(maxsize=None)
def _sdpa_program_config(q_chunk_size: int, k_chunk_size: int) -> ttnn.SDPAProgramConfig:
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID_MAIN,
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )


def _padded_sdpa_len(seq_len: int) -> int:
    return ((seq_len + SDPA_CHUNK_TILE - 1) // SDPA_CHUNK_TILE) * SDPA_CHUNK_TILE


@lru_cache(maxsize=None)
def _capped_sdpa_chunk_size(seq_len: int) -> int:
    if seq_len <= 0:
        return SDPA_CHUNK_TILE
    return min(SDPA_CHUNK_MAX, _padded_sdpa_len(seq_len))


@lru_cache(maxsize=None)
def _sdpa_program_config_for_lengths(q_len: int, k_len: int) -> ttnn.SDPAProgramConfig:
    return _sdpa_program_config(
        q_chunk_size=_capped_sdpa_chunk_size(q_len),
        k_chunk_size=_capped_sdpa_chunk_size(k_len),
    )


@lru_cache(maxsize=None)
def _sdpa_chunks_shipped(q_len: int, k_len: int) -> tuple:
    # Microbench M7/M7b/M7c (Blackhole 13x10, tri-att shape batch=seq, h=8, d=32):
    # the 256-cap is optimal at >=512 (0.59x regression at 64) and 128 is best at
    # <=128, but in the 256<seq<=384 band (298-aa proteins pad to 320, 2 chunks of
    # 256+64 padded) q_chunk=k_chunk=64 is 2.45x faster. Chunking only changes the
    # online-softmax reduction order (measured PCC 0.9999 vs the 256 config).
    # Both sizes were only ever swept together; q is widened past this pick by
    # _tri_att_q_chunks, which is bit-exact, and k stays exactly here.
    #
    # CLOSED, do not re-propose without reading this. The band has no grid term and its 2.45x is a
    # 13x10 Blackhole number, so on an 8x9 Wormhole Galaxy it looks like an obvious defect, and an
    # op-level screen agrees: timing this function's own return value at OpenDDE's tri-att shape
    # (h=12, d=32, batch=seq, bf16) says the capped path wins 1.463x at N=384 and 1.524x at N=352,
    # with both controls at 1.000 (perf/wh-opendde/wh_sdpa_band_odde.py). The audit measured the
    # same inversion at protenix's h=8 shape.
    #
    # The FOLD says the opposite. Gating the band on the grid, A/B'd at 384 aa on the Galaxy,
    # interleaved legs, cold discarded + 3 warm (perf/wh-opendde/results/wh_ab384/):
    #
    #     main            117.942 / 117.855 s  mean 117.899  plDDT 0.798602  cif fc31112e72ed8617
    #     this lever only  126.323 s                          plDDT 0.796598  cif 0f56595198420f80
    #     with the Transition budget too  123.821 / 123.788   plDDT 0.796598  cif 0f56595198420f80
    #     -> the lever alone is 0.9333x, a 6.7 % REGRESSION, and the output moves
    #
    # The mechanism is `_tri_att_q_chunks`: it offers only q_chunks that DIVIDE the padded sequence
    # and treats this function's q value as the production fallback it widens PAST. So the fold
    # never runs the band's (64, 64) -- it runs (wide_q, k=64), and the only thing this function
    # decides in practice is k. Moving k from 64 to 256 makes k stop dividing 384, and the padded
    # q x k mask grid is 84 % of the op's DRAM traffic, so it pads 384 -> 512 and pays it twice.
    #
    # The lesson, because it will recur: an op-level screen on this function's return value
    # measures a configuration the fold never executes. Screen `_tri_att_q_chunks`'s actual pick.
    #
    # K4 moves the k half of the band, and only the k half. 64 divides both 320 and 384, so
    # the fused K1/K2 kernel already serves here and this is not the divisibility bug K3 fixes;
    # it is that 64 is the wrong divisor. MEASURED off-fold at Boltz-2's shape (h=4, d=32),
    # fused-at-64 against fused-at-the-dividing-pick, arms interleaved, on both architectures:
    #
    #     padded            320       384
    #     k, band / K4      64 / 160  64 / 192
    #     Wormhole 8x9      1.4232x   1.5612x   (A/A floor 0.16 %, perf/whb2/out/divk_wh.json)
    #     Blackhole 13x10   1.2994x   1.1670x   (A/A floor 0.5 %, perf/whb2/out/divk_qb1c1.json)
    #
    # Both architectures move the same way, so this cannot trade one against the other. It also
    # does not contradict the 2.45x above: that compares the band against the CAPPED 256 pick,
    # which makes the fused kernel decline outright (measured 1.75-2.70x worse on Wormhole,
    # state doc 11.1). Capped 256 < band 64 < K4's dividing pick, consistently.
    #
    # PREDICTED at the fold, written before the build. 1120 tri-attention calls per fold:
    #     Wormhole 384 aa   1120 * (2.8908 - 1.8517) ms = 1.164 s upper, 0.58 s lower
    #     Wormhole 320 aa   1120 * (1.7529 - 1.2317) ms = 0.584 s upper, 0.29 s lower
    # against walls of 31.919 s and ~27 s, i.e. 1.8-3.6 % and 1.1-2.2 %. The lower bound halves
    # the upper because isolated per-op timing over-syncs roughly 2x against batched work.
    #
    # NOT bit-exact, same reason as K3: k_chunk sets the online-softmax reduction order. The
    # accuracy arm is pLDDT. Separate switch from K3 so the two can be A/B'd apart.
    if 256 < q_len <= 384 and 256 < k_len <= 384:
        if _SDPA_BAND_DIV_K:
            dk = _dividing_sdpa_chunk_size(k_len)
            # Only when it really divides. At padded 288 and 352 no 32-aligned divisor
            # clears the cap/2 floor, so `_dividing_sdpa_chunk_size` hands back the cap,
            # 256, which does not divide either. Taking it there would leave the fused
            # kernel declining exactly as it does today AND move the stock fallback from
            # k=64 to k=256 -- the pick 11.1 measured 1.75-2.70x slower on Wormhole. Those
            # two sizes keep 64 and are untouched by K4.
            if _padded_sdpa_len(k_len) % dk == 0:
                return (64, dk)
        return (64, 64)
    return (_capped_sdpa_chunk_size(q_len), _dividing_sdpa_chunk_size(k_len))


# K3: the k_chunk has to DIVIDE the padded sequence or the fused K1/K2 kernel refuses the call.
# `sdpa_generic.plan` sets `use_padded_mask = (div_up(Sk, k_chunk) * k_chunk != Sk)`, and
# `triatt_sdpa.sdpa` rejects on `fill_preconditions` whenever that is true. `_capped_sdpa_chunk_size`
# returns `min(SDPA_CHUNK_MAX, S)` = 256 above the band, and 256 divides only every fourth multiple
# of 64 -- so with `PAIRFORMER_PAD_MULTIPLE = 64` the fused kernel was silently declining at padded
# 448, 576, 640, 704, 832, 896 and 960 on BOTH architectures while 256/512/768/1024 were served.
# 640 is in that list and it is the first size past Wormhole's `SEQ_LEN_MORE_CHUNKING = 608`.
#
# MEASURED off-fold at Boltz-2's tri-attention shape (h=4, d=32), fused kernel against the stock op
# the fold falls back to today, arms interleaved, A/A control on every unchanged size reading
# 0.9957-1.0004 (perf/whb2/out/divk_qb1c1.json, qb1 13x10; the Wormhole arm is its counterpart):
#
#     padded   448     576     640
#     speedup  2.349x  3.422x  2.577x
#
# Only sizes whose current pick does NOT divide are changed, so 256, 512, 768, 1024 and the whole
# 64/64 band return exactly what they return today, byte for byte, and neutrality there is by
# construction rather than by measurement.
#
# The `>= cap/2` floor keeps today's pick at padded 704 and 832, whose largest 32-aligned divisor is
# 64 -- a quarter of the cap. The fused kernel DOES serve there at 64 (measured: 704 served at
# q=352), so this floor is a precaution, not a measured refusal: a k_chunk that small multiplies the
# per-call chunk count and the screen has not yet priced it against the stock fallback those sizes
# take today. Lower the floor only behind that measurement.
#
# NOT bit-exact: k_chunk sets the online-softmax reduction order. The fold-level accuracy arm is
# pLDDT, not a digest.
_SDPA_DIV_K = os.environ.get("TT_BIO_SDPA_DIV_K", "1") != "0"

# K4: the same dividing pick inside the 256 < seq <= 384 band, where the fused kernel already
# serves at k=64 and 64 is simply not the best divisor. See `_sdpa_chunks_shipped`. Ships OFF
# until its fold-level A/B and pLDDT arm land on both architectures.
_SDPA_BAND_DIV_K = os.environ.get("TT_BIO_SDPA_BAND_DIV_K", "0") != "0"


@lru_cache(maxsize=None)
def _dividing_sdpa_chunk_size(seq_len: int) -> int:
    """The capped k_chunk, or the largest 32-aligned divisor of the padded sequence below it."""
    cap = _capped_sdpa_chunk_size(seq_len)
    padded = _padded_sdpa_len(seq_len)
    if not _SDPA_DIV_K or padded % cap == 0:
        return cap
    for c in range(cap - SDPA_CHUNK_TILE, cap // 2 - 1, -SDPA_CHUNK_TILE):
        if padded % c == 0:
            return c
    return cap


@lru_cache(maxsize=None)
def _tri_att_sdpa_program_config(q_len: int, k_len: int) -> ttnn.SDPAProgramConfig:
    q_chunk, k_chunk = _sdpa_chunks_shipped(q_len, k_len)
    return _sdpa_program_config(q_chunk_size=q_chunk, k_chunk_size=k_chunk)

# Circular-buffer budgets that a q_chunk overflowed on THIS device, so the first fold pays at most
# one throw per shape and every later call skips straight to a config that fits.
_SDPA_Q_CHUNK_OVER_L1: set = set()

# Kill switch so a fold-level A/B can run both arms without a checkout. Bit-exact either way.
_SDPA_WIDE_Q = os.environ.get("TT_BIO_SDPA_WIDE_Q", "1") != "0"

# The atom-level AttentionPairBias widens q from ATOM_WINDOW (32) to ATOM_DIM (128) on dim -2.
# That pad is tile-aligned (32 -> 128 adds exactly 3 tiles of zeros), so leaving TILE layout to
# pad in ROW_MAJOR and tilizing back is a pure round trip. Measured off-fold at the production
# shape [1, 224, 32, 128] bf16, median of 7 after 2 warm, torch.equal against the chain: chain
# 160.53 us, pad in TILE 57.31 us (2.8011x). At the fold: -0.124 s at 512 aa, bit-exact.
_ATOM_PAD_IN_TILE = os.environ.get("TT_BIO_ATOM_PAD_IN_TILE", "1") != "0"

# Boltz-2 diffusion, three levers under A/B. All three are boltz-2-exclusive by construction:
# DiffusionTransformer is built only by tenstorrent.Diffusion, and atom_level=True AdaLN exists
# nowhere else (protenix and openfold3 have their own classes and pass atom_level=False).
#
# L7: cut each layer's head-range out of the attention bias once per fold instead of once per
# denoise step. The bias is uploaded by _populate_diffusion_cache and is constant across all
# sampling steps, so at 512 aa the shipped code issues 6000 identical slices per fold (200 steps
# x 24 token layers + 400 x 3 atom layers). Measured in-fold as stage:DiffusionTransformer minus
# its two layer regions: 449.2 ms. Bit-exact and memory-neutral -- the parts partition the whole
# and the source is freed.
_B2_BIAS_SLICE_HOIST = os.environ.get("BOLTZ2_BIAS_SLICE_HOIST", "1") == "1"

# L6: memoise AdaLN's conditioning half on the atom path. `s` there is the atom conditioning
# `_c_reshaped`, cached once per fold, so 2400 calls per fold recompute 12 answers. This is
# 717d36712 (openfold3's atom transformer, -1.565 s at 512 aa, bit-exact) applied to boltz-2.
# The pair is held in DRAM: 24 retained L1 tensors of 1.83 MB would keep ~44 MB of L1 for the
# whole rollout and clash with a later op's circular buffers.
_B2_ADALN_S_MEMO = os.environ.get("BOLTZ2_ADALN_S_MEMO", "1") == "1"

# S6: route the token-level diffusion transformer's attention through the fused ttnn SDPA,
# deleting the materialised [1, 16, 512, 512] logits tensor and its five DRAM traversals.
# NOT bit-exact: the fused kernel keeps the exponentiated scores in a bf16 circular buffer.
# The trunk's 264 calls are deliberately NOT rerouted -- the trunk hands its pair bias over in
# L1, where ttnn SDPA TT_FATALs, and the forced spill is what confounded the predecessor's arm.
_B2_TOKEN_DIT_SDPA = os.environ.get("BOLTZ2_TOKEN_DIT_SDPA", "0") == "1"

# C2, the triangle bias cast to bfloat8_b before the SDPA. OFF by default and it stays off until a
# fold-level parity gate clears it: at N=512 the op-level error is rmsd/std 0.002547 at PCC
# 1.000000, but W9 measured z rmsd/std 0.04179 on a block at N=320 against a shipped band of
# 0.0185-0.0217, so the block amplifies it ~16x and the fold is the only denominator that decides.
# The cast itself is 4.19 MB read + 2.10 MB written at N=512; the win is the re-read, which the
# SDPA reader pays once per (q_chunk, k_chunk) pair.
_TRIATT_BIAS_B8 = os.environ.get("TT_BIO_TRIATT_BIAS_B8", "0") != "0"


@lru_cache(maxsize=None)
def _tri_att_q_chunks(q_len: int, k_len: int) -> tuple:
    """q_chunk sizes to try for the tri-attention SDPA, widest first, production pick last.

    The kernel re-reads all of K and V once per q-chunk, so one q-chunk spanning the whole padded
    sequence reads them once instead of ceil(seq/256) times. Measured on qb2 card 1 at ttnn 0.68.0,
    torch.equal to the shipped config at every size below -- q_chunk only splits output rows and the
    online softmax reduces over k, so no reduction order changes:

        seq   288  320  352  384  416  448  480  512  544  576
        gain 1.62 1.53 1.68 1.59 1.43 1.34 1.11 1.08 1.81 1.71
        seq   608  640  704  768  896 1024
        gain 1.57 1.41 1.10 1.06 1.28 1.09

    At 256, 672, 736 and 800 no candidate fits, so the policy runs the shipped config and those rows
    measure 1.00x to within 0.42%, which is the op-level A/A floor. There is no size in the measured
    range where widening loses. (perf/triatt_root/phase_b12*.json, phase_b2b*.json,
    c1_intermediate*.json)

    Only q_chunks that DIVIDE the padded sequence are offered. The kernel reads and computes a
    padded_q x padded_k mask grid and that mask is 84% of this op's DRAM traffic, so a q_chunk that
    does not divide the sequence pays the padding twice over: q512 at seq 768 pads 768 -> 1024 and
    measures 0.797x, a loss, while q384 divides it and is 1.06x.

    The ceiling is L1, not a number, so it is discovered rather than declared: q640/k256 fits at
    seq 640 and overflows by 512 B of 1572864 at seq 768, and the per-core budget moves with the
    core count (110 on this part, 130 on a 13x10 one). A hard-coded window calibrated on one grid
    is how the reblock_permute lever became a 0.62x loss on the other.
    """
    prod = _sdpa_chunks_shipped(q_len, k_len)[0]
    if not _SDPA_WIDE_Q:
        return (prod,)
    padded = _padded_sdpa_len(q_len)
    wider = [padded // n for n in range(1, padded // SDPA_CHUNK_TILE + 1)
             if padded % n == 0 and (padded // n) % SDPA_CHUNK_TILE == 0
             and padded // n > prod]
    return tuple(sorted(wider, reverse=True)) + (prod,)


def _tri_att_sdpa(q, k, v, bias, scale: float):
    """SDPA for triangle attention at the widest q_chunk this device's L1 will hold."""
    if _TRIATT_BIAS_B8 and bias is not None and bias.dtype != ttnn.bfloat8_b:
        b8 = ttnn.typecast(bias, ttnn.bfloat8_b)
        try:
            return _tri_att_sdpa_at(q, k, v, b8, scale)
        finally:
            ttnn.deallocate(b8)
    return _tri_att_sdpa_at(q, k, v, bias, scale)


# K5: a k_chunk that DIVIDES the padded sequence even when the only divisors are WIDER than the
# 256 cap. `_dividing_sdpa_chunk_size` searches DOWNWARD from the cap and stops at cap/2, so at a
# padded length whose 32-aligned divisors straddle that window it hands back the non-dividing cap and
# two things go wrong at once: `sdpa_generic.plan` sets `use_padded_mask`, which makes the fused
# K1/K2 kernel decline every call on `fill_preconditions`, and the stock op it falls back to reads a
# mask padded out to the next multiple of the k_chunk. K3 fixed every padded length with a 32-aligned
# divisor in [128, 256); this fixes the ones that have none, where K3's own comment said the cap/2
# floor was "a precaution, not a measured refusal". The measurement says the floor was the wrong
# direction to look: the win is ABOVE the cap, not below it.
#
# Affected padded lengths <= 1024 are exactly 704, 736, 832, 864, 928 and 992. Everywhere else the
# ladder is a single entry and this path is byte for byte today's, 512 / 768 / 896 / 1024 included.
#
# MEASURED at [848, 4, 848, 32] bf16 (PXDesign's Protenix filter cell, padded 864 = 2^5 * 27, whose
# only 32-aligned divisors are 32, 96, 288 and 864), arms interleaved round-robin, the incumbent run
# twice as its own A/A control (0.02%), and the incumbent is the pair the fold actually serves
# (q=288 after 864 is retired over L1), not `_sdpa_chunks_shipped`:
#
#     arm                       path    ms      vs incumbent   rmsd/std vs fp32
#     q288 k256  (incumbent)    stock   17.583  1.000x         0.032688
#     q288 k864                 fused    4.900  3.588x         0.029419
#     q288 k288                 fused    5.992  2.934x         0.032149
#     q288 k96                  fused    8.747  2.010x         0.035024
#     q864 k96                  stock   14.156  1.242x         0.035024
#
# Widest-k wins and it is also the arm CLOSEST to a torch fp32 reference, which follows from one k
# chunk needing no online-softmax rescale at all. So the ladder is k widest-first and the q ladder
# runs INSIDE it: at 848 the best pair is the widest k with the fold's own q, and q-outer ordering
# would settle on (864, 96) at 14.156 ms instead.
#
# NOT bit-exact -- k_chunk sets the online-softmax reduction order -- so the accuracy arm is the
# fold's own structure and confidence, not a digest. See docs/sdpa-wide-k-parity.md.
_SDPA_WIDE_K_DEFAULT = "0"


def _sdpa_wide_k() -> bool:
    """Read live rather than at import so one process can A/B both arms (`_tri_att_k_chunks` is
    deliberately not memoised for the same reason; its divisor loop is ~27 iterations)."""
    return os.environ.get("TT_BIO_SDPA_WIDE_K", _SDPA_WIDE_K_DEFAULT) != "0"


# What THIS process resolved at import, for `scripts/lever_census.py` to report next to the served
# count. The live read above is what decides behaviour; a fold runs one arm per process, so the two
# agree there, and the op screen that flips arms mid-process is the only place they can differ.
SDPA_WIDE_K = _sdpa_wide_k()


def _tri_att_k_chunks(q_len: int, k_len: int) -> tuple:
    """k_chunks to try, widest first, production pick last. One entry unless the shipped pick fails
    to divide the padded sequence, which is the only case K5 changes."""
    prod = _sdpa_chunks_shipped(q_len, k_len)[1]
    padded = _padded_sdpa_len(k_len)
    if not _sdpa_wide_k() or padded % prod == 0:
        return (prod,)
    wider = [padded // n for n in range(1, padded // SDPA_CHUNK_TILE + 1)
             if padded % n == 0 and (padded // n) % SDPA_CHUNK_TILE == 0
             and padded // n > prod]
    return tuple(sorted(wider, reverse=True)) + (prod,)


# [calls served at a k_chunk wider than the shipped pick, calls that fell back to the shipped pick].
# A silently-declined config is indistinguishable from an absent one, so an A/B on this path can
# only be believed if the fold itself says which pair it ran.
SDPA_K_CHUNK_STATS = [0, 0]
# (q_len, k_len) -> [q_chunk, k_chunk, "fused"|"stock"], the pair actually served at that shape.
SDPA_CHUNK_PICKS: dict = {}
# Circular-buffer refusals on the wide-k path, keyed by the FULL config. Deliberately not
# `_SDPA_Q_CHUNK_OVER_L1`: that set is keyed on q_chunk alone, so writing a (q, wide k) refusal into
# it would retire a q_chunk the shipped k runs perfectly well.
_SDPA_QK_OVER_L1: set = set()


def _tri_att_sdpa_at(q, k, v, bias, scale: float):
    q_len, k_len = q.shape[2], k.shape[2]
    k_chunks = _tri_att_k_chunks(q_len, k_len)
    if len(k_chunks) > 1:
        # Only q_chunks that DIVIDE the padded sequence are offered against a wide k. The q ladder's
        # last entry is the production cap, which is the one entry that need not divide, and pairing
        # it with a wide k is the only way this path can lose: it pays the padded q mask twice (the
        # 0.797x the `_tri_att_q_chunks` docstring measures) while today's path serves a q that
        # spans the whole sequence. MEASURED on qb1 card 1 at h=8 d=32 with the cap still offered --
        # padded 544 read 0.9314x and 608 read 0.9104x, both landing on (q256, wide k) against
        # today's (q544/q608, k256), while every length whose widest dividing q fits kept its win
        # (704 3.6954x, 832 1.3201x). Dropping the cap from this loop turns both losses into the
        # single-entry fall-through, which is byte for byte today's path.
        _qs = tuple(qc for qc in _tri_att_q_chunks(q_len, k_len)
                    if _padded_sdpa_len(q_len) % qc == 0)
        for k_chunk in k_chunks[:-1]:
            for q_chunk in _qs:
                cfg = (q_len, k_len, q_chunk, k_chunk)
                if cfg in _SDPA_QK_OVER_L1:
                    continue
                o = _triatt_sdpa.sdpa(q, k, v, bias, scale, q_chunk, k_chunk)
                if o is not None:
                    SDPA_K_CHUNK_STATS[0] += 1
                    SDPA_CHUNK_PICKS[(q_len, k_len)] = [q_chunk, k_chunk, "fused"]
                    return o
                try:
                    o = ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
                        program_config=_sdpa_program_config(q_chunk, k_chunk),
                    )
                    SDPA_K_CHUNK_STATS[0] += 1
                    SDPA_CHUNK_PICKS[(q_len, k_len)] = [q_chunk, k_chunk, "stock"]
                    return o
                except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the L1 budget
                    if "circular buffers" not in str(exc):
                        raise
                    _SDPA_QK_OVER_L1.add(cfg)
        SDPA_K_CHUNK_STATS[1] += 1
    k_chunk = k_chunks[-1]
    fits = [qc for qc in _tri_att_q_chunks(q_len, k_len)
            if (q_len, k_len, qc) not in _SDPA_Q_CHUNK_OVER_L1]
    # The bias is re-read once per batch row by the stock reader; hold it instead. Same
    # preference order as the stock loop below -- `fits` is widest first, production pick last, and
    # the wide q_chunk is worth 1.08-1.81x on its own, so K2 must not silently take the narrow one.
    for q_chunk in fits:
        o = _triatt_sdpa.sdpa(q, k, v, bias, scale, q_chunk, k_chunk)
        if o is not None:
            SDPA_CHUNK_PICKS[(q_len, k_len)] = [q_chunk, k_chunk, "fused"]
            return o
    for q_chunk in fits[:-1]:
        try:
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=scale,
                program_config=_sdpa_program_config(q_chunk, k_chunk),
            )
            SDPA_CHUNK_PICKS[(q_len, k_len)] = [q_chunk, k_chunk, "stock"]
            return o
        except Exception as exc:  # noqa: BLE001 -- re-raised unless it is the L1 budget
            if "circular buffers" not in str(exc):
                raise
            _SDPA_Q_CHUNK_OVER_L1.add((q_len, k_len, q_chunk))
    o = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
        program_config=_sdpa_program_config(fits[-1], k_chunk),
    )
    SDPA_CHUNK_PICKS[(q_len, k_len)] = [fits[-1], k_chunk, "stock"]
    return o



# Kill switch for the batched program config, so a parity gate that spawns one fold per leg can
# A/B it without a checkout.
_BATCHED_MATMUL_ON = os.environ.get("TT_BIO_BATCHED_MATMUL", "1") != "0"

# Output blocks at which a batched matmul reaches the DRAM roof on a p150a. Below it the grid is
# under-occupied and the op falls off the read roof; above it, in1 is re-read from DRAM once per
# extra block for no occupancy gain. Measured on qb1 card 0 at ttnn 0.67.4: every class with more
# than one legal per_core_M is fastest at 32 blocks -- DiT attn@v 0.0337 / 0.0295 / 0.0429 ms at
# 80 / 32 / 16 blocks, AttentionPairBias attn@v 0.0305 / 0.0269 / 0.0434, DiT q@k^T 0.0510 /
# 0.0434 / 0.0580 (perf/bmm_reconcile/pcm_sweep_c0.json).
_BATCHED_MATMUL_SATURATION_BLOCKS = 32


def _batched_matmul_block_w(m_tiles: int, k_tiles: int, n_tiles: int) -> int:
    """The K-block width ttnn itself would pick, which is what keeps the result bit-exact.

    `packer_l1_acc=True` packs each K block's partial back to L1 at the output dtype, so two
    configs agree bit for bit only if they walk the same K blocks. ttnn 0.67.4 picks the width in
    two places (`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp` at tag
    v0.67.4): its 1D factories take `Kt % 2 == 0 ? 2 : 1` (`get_mcast_1d_config`, line 330) and its
    all-DRAM 2D factory starts at `Kt % num_cores_x == 0 ? Kt / num_cores_x : 1` and then lets
    `get_multi_dim_per_core_factor` widen it to whatever the per-core CB budget allows (line 1158,
    1176). The 1D value is reproduced exactly here. The 2D one is not computable outside that
    function, so it is read off the device instead: a wide output block leaves no CB room to widen
    into, and the three 2D classes these models issue measure `torch.equal` at 2 when `Nt <= 4`
    (DiT attn@v) and at 1 otherwise (DiT q@k^T, the trimul class) --
    `perf/bmm_reconcile/width_probe_c0.json`. **Adding a call site means adding its class to
    tests/test_batched_matmul_hw.py**, which is what pins this per shape.
    """
    if k_tiles % 2:
        return 1
    height, width = m_tiles * 32, n_tiles * 32
    narrow = max(height, width) > 8 * min(height, width) or min(height, width) <= 32
    return 2 if narrow or n_tiles <= 4 else 1


@lru_cache(maxsize=None)
def _batched_matmul_search(batch: int, m_tiles: int, k_tiles: int, n_tiles: int, elem_bytes: int,
                           grid: tuple[int, int], l1: int):
    gx, gy = grid
    cores = gx * gy
    if batch < 2 or batch * m_tiles < cores:
        return None
    block_w = _batched_matmul_block_w(m_tiles, k_tiles, n_tiles)
    tile, acc_tile = 1024 * elem_bytes, 4096
    legal = []
    for p in range(1, m_tiles + 1):
        # This factory returns WRONG RESULTS, not just slow ones, whenever a core gets more than
        # one output block and M is split within a batch element. Both dataflow kernels name their
        # per-core loop counter `batch` but the factory passes the per-core BLOCK count into it
        # (matmul_multicore_reuse_optimized_program_factory.cpp:508-547), and each iteration then
        # advances by a whole batch stride: `in0_tensor_start_tile_id += MtKt`
        # (reader_bmm_tile_layout_in0.cpp:115), `in1_tensor_start_tile_id += KtNt` and
        # `out_tensor_start_tile_id += MtNt` (reader_writer_bmm_tile_layout_in1.cpp). That is only
        # the right stride when one block IS one batch element. Two legal escapes:
        #   p == m_tiles     -- one block per batch element, so the stride is correct;
        #   blocks <= cores  -- every core gets one block, so it never increments.
        if m_tiles % p or (p != m_tiles and batch * m_tiles // p > cores):
            continue
        # CB footprint, matmul_multicore_reuse_optimized_program_factory.cpp:286-306: in0 and in1
        # are double-buffered one K block at a time, the output and the fp32 accumulator are whole.
        if 2 * (p + n_tiles) * block_w * tile + p * n_tiles * (tile + acc_tile) > l1:
            continue
        legal.append(p)
    if not legal:
        return None
    # Take the fewest blocks that still saturate the grid; if nothing does, take the most blocks.
    saturating = [p for p in legal if batch * m_tiles // p >= _BATCHED_MATMUL_SATURATION_BLOCKS]
    per_core_M = max(saturating) if saturating else min(legal)
    # out_subblock_h * out_subblock_w must fit the dest register file, which fp32_dest_acc_en
    # halves to 4 tiles. Take the widest legal w, then the tallest h that still fits.
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    sub_h = max(h for h in range(1, min(4 // sub_w, per_core_M) + 1) if per_core_M % h == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=n_tiles,
    )


def _batched_matmul_config(batch: int, m_tiles: int, k_tiles: int, n_tiles: int, elem_bytes: int):
    """Program config that spreads a both-sides-batched matmul over the grid, or None.

    ttnn never picks this factory itself for DRAM-interleaved batched operands. The config it does
    pick is derived from ONE batch element and then walks the whole batch serially inside whichever
    cores that element engaged, so OpenFold3's trunk triangle attention runs 1192 batch elements
    through 10 of 130 cores and the windowed atom attention runs 300 of them through one.
    `MatmulMultiCoreReuseProgramConfig` gives every batch element its own `per_core_M x per_core_N`
    output block instead, which puts the batch on the grid.

    `per_core_N` has to equal `Nt` -- this factory does not split N -- so the only knobs are
    `per_core_M`, which trades in1 re-reads against occupancy, and `in0_block_w`, which is the one
    thing that decides bit-exactness. Both are measured, not guessed:
    `perf/bmm_reconcile/pcm_sweep_c0.json` and `width_probe_c0.json`.
    """
    try:
        l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception:
        return None
    return _batched_matmul_search(batch, m_tiles, k_tiles, n_tiles, elem_bytes,
                                  tuple(COMPUTE_GRID_MAIN), l1)


def _dram_interleaved(t: ttnn.Tensor) -> bool:
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and not mc.is_sharded()


# Shape classes whose tuned config the circular-buffer planner refused once. `_batched_matmul_search`
# budgets against the idle device, so it cannot see what the live block already holds, and the
# clash is raised at program compile inside tt-metal rather than by the allocator. Same contract as
# `_L1_OUT_REFUSED`: one attempt per class per process, then ttnn's own planner for the rest.
_BMM_CFG_REFUSED: set = set()


def batched_matmul(a: ttnn.Tensor, b: ttnn.Tensor, compute_kernel_config=None,
                   dtype=None) -> ttnn.Tensor:
    """`ttnn.matmul` for a batched attention matmul, with the batch spread over the core grid.

    Drop-in for `ttnn.matmul(a, b, compute_kernel_config=..., dtype=...)` on batched
    DRAM-interleaved operands, and falls back to exactly that call whenever the chooser declines.
    Measured on qb1 card 0 at ttnn 0.67.4, 298 aa shapes: 9.5x on the OpenFold3 trunk triangle
    attention attn@v, 1.83x on its q@k^T, 8-14x on the windowed atom attention, 3.8x on the DiT
    attn@v. Every applied class is `torch.equal` against the call it replaces.

    Any rank >= 4 is in range: every leading dim is one batch element, so the openfold3 atom
    transformer's rank-5 `[1,nb,H,Q,dh]` operands are 300 batch elements and not an unsupported
    shape. The leading dims must match on both sides -- the factory reads the batch off in0 alone,
    so a broadcast in1 would stride through memory it does not own.
    """
    sa, sb = tuple(a.shape), tuple(b.shape)
    cfg = None
    if (_BATCHED_MATMUL_ON and len(sa) >= 4 and len(sa) == len(sb) and sa[:-2] == sb[:-2]
            and a.dtype == b.dtype and _dram_interleaved(a) and _dram_interleaved(b)):
        batch = 1
        for d in sa[:-2]:
            batch *= d
        cfg = _batched_matmul_config(
            batch, -(-sa[-2] // 32), -(-sa[-1] // 32), -(-sb[-1] // 32),
            4 if a.dtype == ttnn.float32 else 2)
    kw = {} if dtype is None else {"dtype": dtype}
    if cfg is not None:
        key = (batch, tuple(sa[-2:]), tuple(sb[-2:]), str(a.dtype))
        if key in _BMM_CFG_REFUSED:
            cfg = None
        else:
            try:
                return ttnn.matmul(a, b, compute_kernel_config=compute_kernel_config,
                                   program_config=cfg, **kw)
            except Exception:                                                   # noqa: BLE001
                _BMM_CFG_REFUSED.add(key)
                cfg = None
    return ttnn.matmul(a, b, compute_kernel_config=compute_kernel_config, program_config=cfg,
                       **kw)


# The score tensor is [S, n_heads, S, S], so its element count is n_heads * S**3 -- CUBIC in the
# token count, not quadratic. One fp32 copy is 2 GiB at 512 aa and 16 GiB at 1024, and the plain
# chain below held TWO of them live across multiply / add / softmax. At 1024 aa that was refused per
# bank ("Not enough space to allocate 17179869184 B DRAM buffer across 8 banks ... largest free
# block: 1107296256 B") and OpenFold3 could not fold at all. Two changes fix it, both torch.equal
# with max_abs exactly 0.0 against the chain they replace:
#
#   * the scale folds into the bias add's input-a activation and the softmax reduces in place. That
#     deletes one whole N**3 pass and one of the two fp32 allocations: 20.25 -> 16.25 whole-tensor
#     passes, and the fp32 peak halves (perf/of3sizes/probe_fuse.py, probe_fuse2.py).
#   * rows of the leading dim are blocked when one fp32 score copy exceeds the budget below. Those
#     rows are independent -- the softmax reduces over the last dim only and the bias broadcasts
#     over the leading dim -- so blocking is a partition, not a reordering
#     (perf/of3sizes/screen_triatt_fp32_qb1c0.json: torch_equal true, max_abs 0.0 at 512 and 768).
#
# The budget is keyed on the allocation's own byte count, never on the sequence length, and it sits
# between the largest score tensor MEASURED to allocate (6.75 GiB at 768 aa) and the one measured to
# refuse (16 GiB at 1024). So 128 / 256 / 512 / 768 keep the single-shot path and pay nothing, and
# only 1024 blocks -- into two blocks of 512 rows.
_FP32_SOFTMAX_BLOCK_BYTES = 8 << 30
_FP32_SOFTMAX_FUSED_ADD = True

# The four steps between the two matmuls -- typecast to fp32, the biased add, the softmax, the
# typecast back -- are pure DRAM traffic: measured at 392.4 GB/s against a 383.9 GB/s ttnn.clone
# copy roof on this part, so no program config, compute kernel config or core grid can pay and only
# deleting DRAM passes can. Height-sharding the fp32 score block keeps all four in L1 and leaves
# DRAM with one read of the bf16 scores and one write of the bf16 weights.
#
# Bit-exact by construction -- same ops, same dtypes, same reduction axis, only the memory config
# moves -- and torch.equal with max_abs 0.0 at every block size measured
# (perf/of3x3/screen_l1_chain.py, qb2 card 0, S=512, n_heads=4, 8x8 core grid, whole tail timed
# from the interleaved bf16 scores back to interleaved bf16 weights so each arm pays its own
# transitions):
#
#      rows   fp32 B/core        DRAM        L1        speedup
#         4       262144    0.3582 ms   0.2425 ms      1.4771x
#         8       524288    0.6840 ms   0.4375 ms      1.5634x
#        12       786432    1.0535 ms   0.6436 ms      1.6369x   <- the budget below
#        16      1048576    1.3342 ms   refuses: 1048576 B/core against 937472 B free
#
# The budget is bytes per core, never a sequence length: the whole point is that a block is sized
# to fit L1, and `_triangle_mul_memory_config`'s sequence-length threshold is the mistake not to
# copy. Peak is 1.5x the budget, because the bf16 half of a typecast is live alongside the fp32.
_FP32_SOFTMAX_L1_BYTES_PER_CORE = 768 << 10
_FP32_SOFTMAX_L1_GRID = (8, 8)  # (y, x). 8x8 = 64; this p150a refuses more than 110 shards.

FP32_SOFTMAX_STATS = {"calls": 0, "blocked": 0, "blocks": 0, "fused": 0, "unfused": 0,
                      "l1": 0, "l1_blocks": 0, "l1_refused": 0}

# A block that fits L1 is not automatically a block the sharded softmax fits AROUND: that kernel
# stages its rows through statically allocated circular buffers whose size grows with the row
# width, and at 1024 aa they need 549376 B/core on an 11x10 core range while the model already
# holds ~282 KB/core of its own L1 elsewhere. The block then allocates legally and the softmax
# refuses at program creation. No per-core byte budget can predict that, because the term that
# differs between 512 aa (fits at 786432 B/core) and 1024 aa (does not, at the same 786432) is the
# REST of the model's L1 residency at that size -- so a byte budget tuned at 512 would go silently
# dark above it, which is `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` exactly.
#
# So the gate is empirical and self-setting, the same shape as `_L1_OUT_REFUSED` above: the first
# block at a geometry tries the shard, and a refusal retires that geometry for the process and
# falls back to the interleaved tail, which is bit-identical. It is counted, so it is never a
# silent dark gate: `l1_blocks` vs `l1_refused` in the census says which path a size took.
_FP32_SOFTMAX_L1_REFUSED: set = set()

# The additive pair bias does not depend on the row block, but the tail re-derives its fp32
# copy inside every one of them: 43 blocks per call at 512 aa, so the same 4 MB typecast and
# multiply run 43 times over identical inputs. Hoisting it out of the loop is the same
# recompute elimination as the template and AdaLN hoists, and the same bits.
FP32_SOFTMAX_BIAS_HOIST = True


def _fp32_softmax_l1_rows(per_row: int, height_per_row: int) -> int:
    """Largest leading-dim block whose fp32 score copy is L1-resident. 0 when none is.

    A height shard needs whole tile rows on every core, so the block's flattened height must be a
    multiple of ``cores * 32`` -- the same divisibility constraint as
    ``ttnn-split-work-to-cores-grid-height-holes``, and a block that cannot meet it simply takes
    the interleaved path.
    """
    cores = _FP32_SOFTMAX_L1_GRID[0] * _FP32_SOFTMAX_L1_GRID[1]
    if per_row <= 0 or _FP32_SOFTMAX_L1_BYTES_PER_CORE <= 0:
        return 0
    blk = int(_FP32_SOFTMAX_L1_BYTES_PER_CORE) * cores // per_row
    step = cores * 32
    while blk > 0 and (blk * height_per_row) % step:
        blk -= 1
    return blk


def _fp32_softmax_shard(rows: int, height_per_row: int, width: int):
    """Height-sharded config for one score block, or None when the block does not divide."""
    cores = _FP32_SOFTMAX_L1_GRID[0] * _FP32_SOFTMAX_L1_GRID[1]
    height = rows * height_per_row
    if height % (cores * 32) or width % 32:
        return None
    return ttnn.create_sharded_memory_config(
        shape=(height, width),
        core_grid=ttnn.CoreGrid(y=_FP32_SOFTMAX_L1_GRID[0], x=_FP32_SOFTMAX_L1_GRID[1]),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)


def _fp32_softmax_attention(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    bias: ttnn.Tensor,
    scale_inv: float,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    out_dtype: ttnn.DataType = ttnn.bfloat16,
    bias_scale_inv: float | None = None,
) -> ttnn.Tensor:
    """Manual attention with an fp32 softmax reduction, bf16 operands/storage.

    Mirrors the Boltz-2 reference recipe (attention.py:119-127):
    ``softmax((q@k^T)/sqrt(h) + z)`` in fp32, then ``o = attn @ v`` cast back to bf16.
    The q@k matmul keeps bf16 operands with fp32_dest_acc (the existing einsum recipe);
    only the softmax reduction and the additive bias it consumes upcast to fp32. The
    additive ``bias`` arrives pre-baked by ``sqrt(h)`` (z_weight * sqrt(h)), so it is
    multiplied by ``scale_inv`` to recover the raw reference ``z`` before the add --
    the same undo the fp32_raw_matmul_attention path applies. Replaces the fused
    ``ttnn.transformer.scaled_dot_product_attention`` call (bf16 softmax) when the
    BOLTZ2_FP32_SOFTMAX gate is on.

    Blocks the leading dim when one fp32 score copy exceeds ``_FP32_SOFTMAX_BLOCK_BYTES``.
    """
    FP32_SOFTMAX_STATS["calls"] += 1
    rows = int(q.shape[0])
    height_per_row = int(q.shape[1]) * int(q.shape[2])
    per_row = height_per_row * int(k.shape[2]) * 4
    blk = max(32, int(_FP32_SOFTMAX_BLOCK_BYTES // per_row) // 32 * 32) if per_row else rows
    # The memo is keyed on the SHAPE CLASS, not on the block. A refusal has to retire the block
    # size along with the shard: an L1-sized block that then runs interleaved is the worst of both,
    # 342 blocks per call paying the loop's slice-and-concat with none of the residency back. That
    # is measured, not feared -- it cost 278.23 -> 317.79 s at 1024 aa before this key was widened.
    l1_key = (height_per_row, int(k.shape[2]))
    l1_rows = 0 if l1_key in _FP32_SOFTMAX_L1_REFUSED else _fp32_softmax_l1_rows(per_row,
                                                                                height_per_row)
    if l1_rows:
        blk = min(blk, l1_rows)
        FP32_SOFTMAX_STATS["l1"] += 1

    def shard_for(n):
        if not l1_rows or l1_key in _FP32_SOFTMAX_L1_REFUSED:
            return None
        return _fp32_softmax_shard(n, height_per_row, int(k.shape[2]))

    if rows <= 1 or blk >= rows:
        sh = shard_for(rows)
        FP32_SOFTMAX_STATS["l1_blocks"] += sh is not None
        return _fp32_softmax_attention_block(q, k, v, bias, scale_inv, compute_kernel_config,
                                             out_dtype, bias_scale_inv, sh, l1_key)
    FP32_SOFTMAX_STATS["blocked"] += 1
    parts = []
    # the bias is the same tensor in every block, so its fp32 copy is made once per call
    bias_f = _fp32_softmax_bias(bias, scale_inv, bias_scale_inv) if FP32_SOFTMAX_BIAS_HOIST else None
    for s in range(0, rows, blk):
        e = min(s + blk, rows)
        qs, ks, vs = q[s:e], k[s:e], v[s:e]
        sh = shard_for(e - s)
        FP32_SOFTMAX_STATS["l1_blocks"] += sh is not None
        parts.append(_fp32_softmax_attention_block(qs, ks, vs, bias, scale_inv,
                                                   compute_kernel_config, out_dtype,
                                                   bias_scale_inv, sh, l1_key, bias_f))
        for t in (qs, ks, vs):
            ttnn.deallocate(t)
    FP32_SOFTMAX_STATS["blocks"] += len(parts)
    if bias_f is not None:
        ttnn.deallocate(bias_f)
    o = ttnn.concat(parts, dim=0)
    for part in parts:
        ttnn.deallocate(part)
    return o


def _fp32_softmax_bias(bias, scale_inv, bias_scale_inv):
    """The tail's fp32 bias: cast, then undo the pair-bias pre-bake. A pure function of
    ``bias``, so a blocked call makes it once instead of once per block."""
    bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
    return ttnn.multiply(bias_f, scale_inv if bias_scale_inv is None else bias_scale_inv)


def _fp32_softmax_attention_block(q, k, v, bias, scale_inv, compute_kernel_config,
                                  out_dtype, bias_scale_inv, shard=None, l1_key=None,
                                  bias_f=None):
    """One row block of `_fp32_softmax_attention`. The whole tensor is one block below the budget.

    ``shard`` height-shards the block so the four steps between the two matmuls stay in L1. Both
    matmuls keep interleaved operands either way, so the shard is live only across the tail.
    """
    kt = ttnn.permute(k, (0, 1, 3, 2))
    # bf16 operands, fp32_dest_acc -> bf16 scores (the existing einsum recipe).
    sc = batched_matmul(q, kt, compute_kernel_config=compute_kernel_config)
    ttnn.deallocate(kt)
    attn_bf = None
    if shard is not None:
        try:
            attn_bf = _fp32_softmax_tail(sc, bias, scale_inv, bias_scale_inv, shard, bias_f)
        except RuntimeError:
            # The shard allocated but the sharded softmax could not fit its circular buffers
            # around it. Retire this geometry and take the interleaved tail, which is the same
            # ops on the same dtypes and therefore the same bits.
            _FP32_SOFTMAX_L1_REFUSED.add(l1_key)
            FP32_SOFTMAX_STATS["l1_refused"] += 1
            attn_bf = None
    if attn_bf is None:
        attn_bf = _fp32_softmax_tail(sc, bias, scale_inv, bias_scale_inv, None, bias_f)
    ttnn.deallocate(sc)
    o = batched_matmul(attn_bf, v, compute_kernel_config=compute_kernel_config, dtype=out_dtype)
    ttnn.deallocate(attn_bf)
    return o


def _fp32_softmax_tail(sc0, bias, scale_inv, bias_scale_inv, shard, bias_f=None):
    """bf16 scores -> bf16 attention weights, both interleaved. ``shard`` keeps the middle in L1.

    ``sc0`` is left allocated either way, so a caller can retry interleaved after a refusal.
    """
    if shard is not None:
        # typecast refuses a layout change, so the bf16 scores move to L1 first and the fp32 copy
        # is born sharded. That read is the tail's only DRAM read. The bf16 half is freed the
        # moment the fp32 copy exists: holding both plus the bf16 result at the end of the tail is
        # 1.5 MB/core and refuses.
        sc_l1 = ttnn.to_memory_config(sc0, shard)
        sc = ttnn.typecast(sc_l1, ttnn.float32, memory_config=shard)
        ttnn.deallocate(sc_l1)
    else:
        sc = ttnn.typecast(sc0, ttnn.float32, memory_config=sc0.memory_config())
    # undo the pair-bias pre-bake (sqrt(h) for Boltz/Protenix, 1.0 for openfold3). A blocked
    # call hands the same fp32 copy to every block; ``own`` says who frees it.
    own = bias_f is None
    if own:
        bias_f = _fp32_softmax_bias(bias, scale_inv, bias_scale_inv)
    if _FP32_SOFTMAX_FUSED_ADD:
        # The score scale rides the add's input-a activation instead of taking a pass of its own,
        # and the softmax reduces into the same buffer. Both are bit-exact; the bias is O(S**2) so
        # its own multiply is three orders below the scores and stays where it is.
        FP32_SOFTMAX_STATS["fused"] += 1
        attn = ttnn.add_(sc, bias_f, input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale_inv)])
        if own:
            ttnn.deallocate(bias_f)
        try:
            attn = ttnn.softmax_in_place(attn)
        except RuntimeError:
            ttnn.deallocate(attn)
            raise
    else:
        FP32_SOFTMAX_STATS["unfused"] += 1
        sc = ttnn.multiply(sc, scale_inv)
        sc = ttnn.add(sc, bias_f)
        if own:
            ttnn.deallocate(bias_f)
        attn = ttnn.softmax(sc, dim=-1)  # fp32 softmax reduction
        ttnn.deallocate(sc)
    attn_bf = ttnn.typecast(attn, ttnn.bfloat16, memory_config=attn.memory_config())
    ttnn.deallocate(attn)
    if shard is not None:
        attn_i = ttnn.to_memory_config(attn_bf, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_bf)
        attn_bf = attn_i
    return attn_bf


# The pair-tensor dim0/dim1 transpose is 4.0474 ms on [512,512,256] bf16 DRAM->DRAM, 66.3 GB/s
# against a 389.9 GB/s clone roof, because tiling covers the last two dims and swapping the untiled
# batch dim with the tile-row dim moves single 64-byte rows between tiles. Going out through
# ROW_MAJOR removes that: the permute then sees no tiles at all, and retiling afterwards is a
# whole-tile op. Measured in one process at that shape, warm 3, median of 7, torch.equal against the
# tiled permute (perf/bigswing/pair_tr_ttnn_512_qb2c0.json):
#
#     permute (tiled)  4.0474 ms   66.3 GB/s      transpose(0,1)  4.0664 ms  (same kernel)
#     via ROW_MAJOR    2.5275 ms  106.2 GB/s  1.6014x  <- this route
#     4-D (0,2,1,3)    4.0753 ms  (same kernel)   clone roof      0.6885 ms  389.9 GB/s
#
# It costs one extra tensor of DRAM peak while the round trip is in flight, so it is gated to the
# DRAM destination -- an L1 destination is already 1.4762 ms (2.7417x) and needs no help.
PAIR_TRANSPOSE_VIA_ROW_MAJOR = True
_PT_ROW_MAJOR = os.environ.get(
    "TT_BIO_PAIR_TRANSPOSE_RM", "1" if PAIR_TRANSPOSE_VIA_ROW_MAJOR else "0") == "1"


# Pair-tensor shape classes whose L1 transpose destination the allocator refused once. The
# static budget in `_l1_memory_config_if_it_fits` cannot see what the live block already holds,
# so the honest test is the allocation itself; remembering the refusal keeps it to one attempt
# per class per process, the same pattern `_L1_OUT_REFUSED` uses for the projections.
_TRANSPOSE_L1_REFUSED: set = set()


def _pair_transpose(t: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """``permute(t, (1, 0, 2))``, through ROW_MAJOR where that wins. Bit-exact either way.

    An L1 destination is asked for by `_transpose_memory_config` and can still be refused at
    the call site, so the L1 attempt falls back to DRAM rather than killing the fold.
    """
    if memory_config.buffer_type == ttnn.BufferType.L1:
        key = (tuple(t.padded_shape), str(t.dtype))
        if key not in _TRANSPOSE_L1_REFUSED:
            try:
                return _pair_transpose_impl(t, memory_config)
            except Exception:                                                   # noqa: BLE001
                _TRANSPOSE_L1_REFUSED.add(key)
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    return _pair_transpose_impl(t, memory_config)


def _pair_transpose_impl(t: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    if (_PT_ROW_MAJOR and len(t.shape) == 3
            and memory_config.buffer_type == ttnn.BufferType.DRAM
            and t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT):
        # Both to_layout calls MUST be pinned to the destination's buffer type. Without a
        # memory_config ttnn places the intermediate by its own default, which is L1: that
        # made openfold3 at 576 tokens die on `Out of Memory: Not enough space to allocate
        # 84934656 B L1 buffer across 110 banks` -- exactly this tensor -- while the same
        # fold passed with the route off. The round trip is a DRAM->DRAM move and every
        # tensor in it belongs in DRAM.
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        p = ttnn.permute(rm, (1, 0, 2), memory_config=memory_config)
        ttnn.deallocate(rm)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=memory_config)
        ttnn.deallocate(p)
        return o
    return ttnn.permute(t, (1, 0, 2), memory_config=memory_config)


# L1 total on an 11x10 Blackhole grid is 168.57 MB (110 x 1 532 416 B), so the 512 aa pair
# tensor at 134.22 MB is 79.6 % of it and the largest headroom that fits at all is 1.2559. At
# the old 2.5 only 160 of protenix-v2's 1208 pair transposes per 512 aa fold could take the L1
# route and the other 1048 always paid DRAM. 1.25 admits all 1208 with no allocator refusal,
# and is worth 52.407 -> 51.062 s/fold on qb2 card 1, byte-identical
# (perf/px4pd/e6_tr_qb2c1.json). 1.25 rather than 1.2559: the consumer's circular buffers come
# out of the same banks, and the refusal fallback in `_pair_transpose` is what makes the
# tight value safe.
TRANSPOSE_L1_HEADROOM = 1.25
_TRANSPOSE_L1_HEADROOM = float(
    os.environ.get("TT_BIO_TRANSPOSE_L1_HEADROOM", str(TRANSPOSE_L1_HEADROOM)))


def _transpose_memory_config(t: ttnn.Tensor) -> ttnn.MemoryConfig:
    """L1 for a pair-tensor dim0/dim1 transpose when it fits, else DRAM.

    ttnn's dim0/dim1 permute is a real element transpose, not a tile-block copy: tiling
    covers the last two dims, so swapping the untiled batch dim with the tile-row dim moves
    single rows between tiles. Its writes are therefore row-granular scatter, and DRAM
    punishes that. Measured on a Blackhole P300c chip at the 298-aa pair shape
    320x320x256 bf16 (median of 9 synced calls): 1.479 ms to DRAM = 70.9 GB/s, against
    0.281 ms for a plain ttnn.clone of the same tensor = 373.3 GB/s, so the transpose runs
    at 19% of the copy roof. Into L1 the same permute is 0.600 ms, 2.47x. ttnn.permute,
    ttnn.transpose(0,1) and the 4-D (0,2,1,3) form all land on the same kernel and the same
    1.48 ms, so this is the only lever short of a new kernel.

    A memory config cannot change a value: verified bit-identical (torch.equal) against the
    DRAM permute.

    Not cached: ttnn.Tensor hashes by object identity, so an lru_cache here would miss on
    every call and pin every tensor it ever saw for the life of the process.
    """
    # 2.5x headroom: the consumer still needs its circular buffers on every core.
    return _l1_memory_config_if_it_fits(t, _TRANSPOSE_L1_HEADROOM)


def _l1_layer_norm(x: ttnn.Tensor, headroom: float, **kw):
    """`ttnn.layer_norm` writing to L1 when it fits, else to DRAM. Returns (tensor, in_l1).

    For a narrow-output projection that reads a whole activation tensor to write one tile of
    width, the source is the cost and the destination is not, so the lever is to hand the
    projection an L1-resident operand. `_l1_memory_config_if_it_fits` is a static budget and
    cannot see what the live block already holds, so the allocation itself is the real test and
    a refusal has to leave today's behaviour exactly intact.
    """
    if _l1_memory_config_if_it_fits(x, headroom) is ttnn.L1_MEMORY_CONFIG:
        try:
            return ttnn.layer_norm(x, memory_config=ttnn.L1_MEMORY_CONFIG, **kw), True
        except Exception:                                                     # noqa: BLE001
            pass
    return ttnn.layer_norm(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw), False


def _l1_memory_config_if_it_fits(t: ttnn.Tensor, headroom: float) -> ttnn.MemoryConfig:
    """L1 when `headroom` copies of `t` fit across the grid's banks, else DRAM.

    `headroom` is what the consumer needs on top of the tensor itself: its circular buffers, and
    for a producer whose result is read in place, the other operands that stay live beside it.
    """
    try:
        per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception:
        return ttnn.DRAM_MEMORY_CONFIG
    shape = [int(d) for d in t.shape]
    if len(shape) < 2:
        return ttnn.DRAM_MEMORY_CONFIG
    volume = 1
    for d in shape[:-2]:
        volume *= d
    volume *= ((shape[-2] + 31) // 32) * 32 * ((shape[-1] + 31) // 32) * 32
    elem = 4 if t.dtype == ttnn.float32 else 2
    if headroom * volume * elem <= per_core * COMPUTE_GRID_MAIN[0] * COMPUTE_GRID_MAIN[1]:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


@lru_cache(maxsize=None)
def _triangle_mul_program_config(seq_len_tiles: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    gx, gy = COMPUTE_GRID_MAIN
    per_core_M = -(-seq_len_tiles // gy)
    per_core_N = -(-seq_len_tiles // gx)
    # in0_block_w must divide seq_len_tiles (Kt). Measured on Blackhole: widest
    # legal block is 2.4x faster than 1 at Kt=10 and 2.15x at Kt=16 (microbench M2/M2b);
    # K-tile accumulation order into the fp32 dest register is unchanged.
    in0_block_w = max(d for d in range(min(10, seq_len_tiles), 0, -1) if seq_len_tiles % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


@lru_cache(maxsize=1)
def _l1_bank_bytes() -> int:
    """Bytes of L1 a program config may plan for, per bank.

    `ttnn.get_max_worker_l1_unreserved_size()` is not that number. On Blackhole it reads 1532416 B
    while the L1 allocator reports 1461760 B per bank, so a gate sized against it can admit a
    config that does not fit on a completely idle device, let alone beside live activations. Read
    the allocator instead, once, and fall back to the device number only if the view is
    unavailable.

    Deliberately the idle capacity and not the live free number: a gate that reads free L1 per call
    pays a `get_memory_view` pipeline drain (~6 us, measured by E6) for a decision that a 298 aa
    fold never changes, and both gates below leave 3.4x-7.8x headroom (perf/pcgate).
    """
    try:
        return int(ttnn.get_memory_view(get_device(), ttnn.BufferType.L1).total_bytes_per_bank)
    except Exception:
        return int(ttnn.get_max_worker_l1_unreserved_size())


@lru_cache(maxsize=None)
def _tri_att_qkv_l1_config(
    m_tiles: int, k_tiles: int, n_tiles: int, elem_bytes: int
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
    """Program config for a tri-attention qkv projection whose result can stay in L1, or None.

    The projection is tall and narrow (16384x256 @ 256x768 at 128 tokens), so it costs almost
    nothing to compute and almost everything to write down: 25.17 MB of the op's 33.95 MB of DRAM
    traffic is the result. Keeping it in L1, and letting nlp_create_qkv_heads read it there, takes
    the projection + head-split + SDPA chain from 0.519 ms to 0.264 ms at 128 tokens, measured on
    card 3.

    Two guards, both hard:

    - `in0_block_w` must come out as the whole of K. A narrower K block is a different
      accumulation order and would not be bit-exact against the minimal_matmul this replaces; the
      output CB grows with per_core_M, so past ~128 tokens the whole K stops fitting and the
      projection falls back. That makes the fit test and the bit-exactness test the same test.
    - the result and the q/k/v that follow it must both fit in aggregate L1 with room to spare,
      since the rest of the block is still allocating.
    """
    gx, gy = COMPUTE_GRID_MAIN
    num_cores = gx * gy
    if k_tiles >= num_cores or m_tiles < n_tiles * 8 or n_tiles > 64:
        return None
    per_core_M = next(
        (p for p in range(max(1, -(-m_tiles // num_cores)), m_tiles + 1) if m_tiles % p == 0), 0
    )
    if not per_core_M or -(-m_tiles // per_core_M) > num_cores:
        return None
    l1 = _l1_bank_bytes()
    tile = 1024 * elem_bytes
    # Output CB plus the fp32 accumulation CB are fixed; in0 and in1 scale with the K block. The
    # result itself is L1-resident and ttnn allocates it BEFORE the program factory places a
    # single circular buffer, so its per-bank share comes off the budget too (E6).
    fixed = per_core_M * n_tiles * (tile + 4096) + 128 * 1024
    fixed += -(-(m_tiles * n_tiles) // num_cores) * tile
    per_block = (per_core_M + n_tiles) * tile
    if fixed + k_tiles * per_block > l1:
        return None
    if 2 * m_tiles * n_tiles * tile > 0.6 * num_cores * l1:
        return None
    out_subblock_w = max((w for w in range(min(4, n_tiles), 0, -1) if n_tiles % w == 0), default=1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=k_tiles,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M,
        out_block_w=n_tiles,
        per_core_M=per_core_M,
        per_core_N=n_tiles,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )



@lru_cache(maxsize=None)
def _pair_proj_program_config(
    m_tiles: int, k_tiles: int, n_tiles: int, in0_block_w: int, elem_bytes: int,
    out_l1: bool = False, block_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
    """Program config for a tall pair-track projection, or None if the shape is outside it.

    `ttnn.linear(core_grid=...)` derives `in0_block_w = 1` and `out_block_h = per_core_M` for
    these shapes, and both are the worst legal choice on a 102400x256 @ 256x256 pair-track
    projection. At `in0_block_w = 1` a core builds each output tile in Kt inner blocks, so the
    in1 multicast barrier, the DEST clear and the packer pass that folds the partial through
    `packer_l1_acc` are all paid Kt times per output tile instead of once. At
    `out_block_h = per_core_M` a core finishes its whole output block before the writer moves
    any of it, so a 52.4 MB DRAM write has only the tail of the op to hide behind.

    Dropping `out_block_h` to 5 alone is 0.7521 -> 0.5957 ms on that class, 1.263x, and it is
    `torch.equal` against the production call: with `in0_block_w` unchanged the contraction is
    accumulated in the same order, only the drain schedule moves. Raising `in0_block_w` on top
    is a further 1.55x and is NOT bit-exact (perf/pf_matmul/proj_ab.py).

    Returns None whenever anything does not divide or the circular buffers do not fit, so a
    shape outside the measured set keeps today's behaviour.
    """
    gx, gy = COMPUTE_GRID_MAIN
    num_cores = gx * gy
    if m_tiles < num_cores or k_tiles % in0_block_w:
        return None  # a 1D M-split cannot fill the grid below one tile row per core
    # per_core_M need not divide m_tiles -- only ceil(m_tiles/per_core_M) <= num_cores is
    # required -- but out_block_h must divide per_core_M, and 5 is the measured optimum. The
    # production shape is a batched (298, 298, c_z), so m_tiles = 2980 and the smallest legal
    # per_core_M is 23, a prime that would force out_block_h to 1 or 23. Rounding up to the next
    # multiple of 5 costs 120 of 130 cores instead of 130 and buys the drain schedule.
    per_core_M = -(-(-(-m_tiles // num_cores)) // 5) * 5
    if per_core_M > m_tiles or -(-m_tiles // per_core_M) > num_cores:
        return None
    out_block_h = 5
    # `out_block_w` defaults to the whole output row, which is what every caller before the pair
    # FFN wanted. It is a drain-schedule parameter, not a contraction one, so narrowing it is
    # free of any parity decision and it is the only way an L1 destination fits at n_tiles = 32:
    # the in0/in1 buffers alone are 2*bw*(obh + obw) tiles and at obw = 32 that is 1.21 MB of a
    # 1.46 MB bank before the output is counted.
    out_block_w = n_tiles if block_w is None else block_w
    if out_block_w < 1 or n_tiles % out_block_w:
        return None
    sh = max(h for h in range(min(4, out_block_h), 0, -1) if out_block_h % h == 0)
    sw = max(w for w in range(min(4 // sh, out_block_w), 0, -1) if out_block_w % w == 0)
    l1 = _l1_bank_bytes()
    tile = 1024 * elem_bytes
    # in0 and in1 are double-buffered per K block; the output block carries its bf16 tile plus
    # the fp32 partial the packer accumulates into. An L1 output takes bank space on top of that
    # and has to be subtracted here -- a program-config budget that forgets its output term is
    # how a gate lets through a config the allocator then refuses at the real call site.
    need = (2 * in0_block_w * (out_block_h + out_block_w) * tile
            + out_block_h * out_block_w * (tile + 4096) + 128 * 1024
            + (per_core_M * n_tiles * tile if out_l1 else 0))
    if need > l1:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=n_tiles,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


def _pair_proj_config(x: ttnn.Tensor, w: ttnn.Tensor, bw_cap: int | None = -1,
                      out_l1: bool = False, block_w: int | None = None) -> object | None:
    """_pair_proj_program_config for a concrete operand pair, or None if it does not apply.

    `bw_cap` defaults to the module's `_PAIR_PROJ_BW`; the narrow-output sites pass
    `_NARROW_PROJ_BW` and the L1-output sites `_PAIR_PROJ_L1_BW`, so the three carry
    independent parity decisions."""
    cap = _PAIR_PROJ_BW if bw_cap == -1 else bw_cap
    if cap is None or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return None
    try:
        xs, ws = list(x.shape), list(w.shape)
        if len(xs) < 2 or len(ws) != 2:
            return None
        # Tiles as ttnn stores them: the last two dims are the tile grid and every leading dim is
        # batch, which fuse_batch flattens into M. The pair track is (B, N_tok, N_tok, c_z), so
        # a logical 298 pads to 320 per batch row and m_tiles is 298 x 10, not (298 x 298)/32.
        batch = 1
        for d in xs[:-2]:
            batch *= int(d)
        m_tiles = batch * -(-int(xs[-2]) // 32)
        k_tiles = -(-int(xs[-1]) // 32)
        n_tiles = -(-int(ws[-1]) // 32)
        if k_tiles != -(-int(ws[-2]) // 32):
            return None
        bw = max((d for d in (k_tiles, 8, 4, 2, 1)
                  if d <= cap and k_tiles % d == 0), default=1)
        return _pair_proj_program_config(m_tiles, k_tiles, n_tiles, bw, 2, out_l1, block_w)
    except Exception:
        return None


# Operand classes whose L1 output the allocator refused once. The static budget in
# `_pair_proj_program_config` cannot see what a live block already holds, so the honest test is
# the allocation itself; remembering the refusal keeps it to one attempt per class per process.
_L1_OUT_REFUSED: set = set()


# The DRAM-output leg of `_pair_proj_linear` is a `ttnn.linear` with a program config tuned for
# the out_block_h drain schedule. For a square K=256, N=256 pair projection `minimal_matmul` with
# the swept block config is simply faster, and bit-exact: K_block covers the whole contraction in
# both, so nothing accumulates in a different order. Measured on qb2 card 2, `torch.equal` at
# every size (perf/triatt_opt/stage1_sweep.json):
#     298   0.4016 -> 0.3154 ms   320   0.4132 -> 0.3312   384   0.5888 -> 0.4589
#     512   0.9949 -> 0.7844      576   1.2548 -> 0.9889   640   1.5561 -> 1.2052
# The L1-output leg still wins where it applies (298: 0.2838), so this sits BELOW it and above
# the DRAM linear. Scoped to a single-block contraction (kt == 8) because that is the class where
# the identical accumulation order was verified.
PAIR_PROJ_MINIMAL_MATMUL = True
_PAIR_PROJ_MM = os.environ.get(
    "TT_BIO_PAIR_PROJ_MM", "1" if PAIR_PROJ_MINIMAL_MATMUL else "0") == "1"


def _pair_proj_minimal_matmul(x, w, ckc, dtype):
    """`minimal_matmul` for a pair projection whose contraction fits one K block, else None."""
    if not _PAIR_PROJ_MM or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return None
    if len(w.shape) != 2 or -(-int(w.shape[-2]) // 32) != 8:
        return None
    cfg = _qkv_mm_config(x, w)
    if cfg is None:
        return None
    try:
        return ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=dtype, config=cfg)
    except Exception:
        return None


# The pair FFN's fc1 halves, [B*rows, N, c_z] x [c_z, d_ff]: k_tiles 8, n_tiles 32. The default
# gate refuses an L1 destination for this class at every row height and every block width, so all
# three matmuls of the row-blocked pair FFN fell back to a plain `ttnn.linear(core_grid=...)` with
# a DRAM output and fc1's 2.15 GB/call round trip was never removed. Naming out_block_w = 16 fits
# it. `in0_block_w` must stay 1: of the 80 configs swept at [1,32,512,256] x [256,1024], all 20 at
# bw = 1 are `torch.equal` against the shipped call and all 60 above it differ by one bf16 ulp,
# which is the K accumulation order. obw = 32 is faster bare (0.164 vs 0.206 ms) and CLASHES in
# the chain once one half is already resident; obw = 8 costs 2.45 ms/call. MEASURED on qb2 card 2,
# whole FFN at [1,512,512,256] rows=32: 17.918 -> 14.662 ms, `torch.equal`
# (perf/esm3p4/screen_a_c2.json, screen_b_c2.json).
_PAIR_FFN_FC1_BW = 1
_PAIR_FFN_FC1_BLOCK_W = 16


def _pair_proj_linear(x, w, ckc, dtype, l1_out: bool = False,
                      l1_bw: int | None = None, l1_block_w: int | None = None):
    """`ttnn.linear` on a pair-track projection, with the tuned config when the shape allows.

    `l1_out` is for the members whose consumer reads the result straight back on device (the
    trimul's `multiply_`, the Pairformer layer's residual `add_`): the projection's 48.82 MB DRAM
    write and the consumer's operand read both disappear. Falls back to today's DRAM path if the
    allocator refuses, which is the only test that knows what the live block is already holding.
    """
    if l1_out and _PAIR_PROJ_L1_OUT:
        key = (tuple(x.padded_shape), tuple(w.shape), str(dtype), l1_bw, l1_block_w)
        if key not in _L1_OUT_REFUSED:
            cfg = _pair_proj_config(
                x, w, bw_cap=_PAIR_PROJ_L1_BW if l1_bw is None else l1_bw,
                out_l1=True, block_w=l1_block_w)
            if cfg is not None:
                try:
                    return ttnn.linear(
                        x, w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype,
                        compute_kernel_config=ckc, program_config=cfg,
                    )
                except Exception:
                    _L1_OUT_REFUSED.add(key)
    mm = _pair_proj_minimal_matmul(x, w, ckc, dtype)
    if mm is not None:
        return mm
    cfg = _pair_proj_config(x, w)
    if cfg is not None:
        return ttnn.linear(
            x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
            compute_kernel_config=ckc, program_config=cfg,
        )
    return ttnn.linear(
        x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
        compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
    )


def _narrow_proj_linear(x, w, ckc, dtype, l1_out: bool = False):
    """The tuned config for a NARROW-output pair-track projection, or None to leave the
    call alone. Scoped to an output of at most two tiles: that is the class whose
    `core_grid=` ladder is flat from 16 cores to 110, and it keeps every wider `_lin` call
    on today's path. `l1_out` is for the sites whose next op reads the result on device."""
    if -(-int(list(w.shape)[-1]) // 32) > 2:
        return None
    cfg = _pair_proj_config(x, w, bw_cap=_NARROW_PROJ_BW, out_l1=l1_out)
    if cfg is None:
        return None
    key = (tuple(x.padded_shape), tuple(w.shape), str(dtype))
    if l1_out and key not in _L1_OUT_REFUSED:
        try:
            return ttnn.linear(
                x, w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype,
                compute_kernel_config=ckc, program_config=cfg,
            )
        except Exception:
            _L1_OUT_REFUSED.add(key)
            cfg = _pair_proj_config(x, w, bw_cap=_NARROW_PROJ_BW)
            if cfg is None:
                return None
    return ttnn.linear(
        x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
        compute_kernel_config=ckc, program_config=cfg,
    )


@lru_cache(maxsize=None)
def _attn_value_program_config(
    m_tiles: int, k_tiles: int, n_tiles: int, batch: int, elem_bytes: int
) -> ttnn.MatmulMultiCoreReuseProgramConfig | None:
    """Program config for a batched `attn @ v`, or None if the shape is outside it.

    `attn @ v` in a manual attention block has N = head_dim/32, one or two tiles. ttnn's config
    builder sees a single N block, takes its 1D `mcast_in1` branch and sets `per_core_M = 1`, so
    the op engages exactly `Mt` cores -- 10 of 110 at 298 tokens -- and walks the head batch
    serially inside each of them. That is an occupancy defect, not a bandwidth one: the naive
    path runs at 9% of this card's DRAM read roof.

    Splitting M across the grid with `MatmulMultiCoreReuseProgramConfig` fixes it, and the one
    field that must not move is `in0_block_w`. It is the K-blocking, and the K-blocking is the
    only thing a config can change that regroups the fp32 accumulation (RFD3 p15), so mirroring
    ttnn's own value makes the tuned arm bitwise identical to the shipped path. Measured over 8
    shape classes x ~50 arms on two cards and two ttnn versions: every arm whose `in0_block_w`
    matched came back `torch.equal`, every arm that differed did not, no exception. The same
    result rules out `core_grid=` here, which hard-codes the K block to 1.

    `per_core_M` is the smallest legal split. G1's correctness predicate bounds it: the reuse
    factory's dataflow kernels advance a whole batch stride per per-core loop iteration while the
    factory hands them a per-core block count, so the answer is only right when each core gets
    exactly one block. It is not always the fastest legal split -- a per-shape sweep beats it by
    13-18% at two of the four live classes (perf/attn_sites/) -- but no closed-form rule
    reproduced the measured ordering across dtype and N, so the rest of the margin is left to a
    calibrating variant instead of guessed at.

    Engaged-core count is deliberately not a gate. At the RFD3 atom block (Mt=126, batch=4) the
    naive path already spreads one M-tile per core over 126 of them and the tuned split uses 84,
    yet the tuned split is 1.41x faster: each core owns one whole batch element instead of
    walking all four serially, and a per_core_M of 6 amortises the write barrier the 1-tile
    output block pays on every tile.
    """
    gx, gy = COMPUTE_GRID_MAIN
    num_cores = gx * gy
    # The mirror is only correct if ttnn takes a 1D branch. N must fit in one block of the square
    # factor its builder picks (8 for every shape here; 16 never fits L1) and M must not, or the
    # shape lands on MatmulMultiCore and there is no naive `in0_block_w` to mirror.
    if n_tiles > 8 or m_tiles <= 8 or n_tiles >= m_tiles:
        return None
    in0_block_w = 2 if k_tiles % 2 == 0 else 1
    per_core_M = next(
        (p for p in range(1, m_tiles + 1)
         if m_tiles % p == 0 and batch * (m_tiles // p) <= num_cores),
        0,
    )
    if not per_core_M:
        return None
    tile = 1024 * elem_bytes
    # ttnn's own get_estimated_size_of_cbs for DRAM-interleaved operands: in0 and in1 double
    # buffered per K block, plus the output tile and the fp32 partial the packer accumulates
    # into (both models run fp32_dest_acc_en=True).
    need = (2 * in0_block_w * (per_core_M + n_tiles) * tile
            + per_core_M * n_tiles * (tile + 4096))
    if need > _l1_bank_bytes():
        return None
    sh = max(h for h in range(min(4, per_core_M), 0, -1) if per_core_M % h == 0)
    sw = max(w for w in range(min(4 // sh, n_tiles), 0, -1) if n_tiles % w == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=per_core_M,
        per_core_N=n_tiles,
    )


def attn_value_matmul(attn, v, ckc, dtype):
    """`attn @ v` on the tuned config where the shape allows, otherwise exactly as before."""
    cfg = None
    try:
        a_s, v_s = list(attn.shape), list(v.shape)
        if len(a_s) >= 3 and len(a_s) == len(v_s) and attn.dtype == v.dtype:
            batch = 1
            for d in a_s[:-2]:
                batch *= int(d)
            cfg = _attn_value_program_config(
                -(-int(a_s[-2]) // 32), -(-int(a_s[-1]) // 32), -(-int(v_s[-1]) // 32),
                batch, 4 if attn.dtype == ttnn.float32 else 2,
            )
    except Exception:
        cfg = None
    if cfg is None:
        return ttnn.matmul(attn, v, compute_kernel_config=ckc, dtype=dtype)
    return ttnn.matmul(attn, v, compute_kernel_config=ckc, dtype=dtype, program_config=cfg)


def _qkv_l1_config(x: ttnn.Tensor, w: ttnn.Tensor, dtype) -> object | None:
    """_tri_att_qkv_l1_config for a concrete operand pair, or None if it does not apply."""
    if dtype != ttnn.bfloat16:
        return None  # the CB budget above is sized in bf16 tiles
    try:
        xs, ws = list(x.shape), list(w.shape)
        m = 1
        for d in xs[:-1]:
            m *= int(d)
        k, n = int(xs[-1]), int(ws[-1])
        if m % 32 or k % 32 or n % 32:
            return None
        return _tri_att_qkv_l1_config(m // 32, k // 32, n // 32, 2)
    except Exception:
        return None


def _apply_grid_thresholds(grid: tuple[int, int], device=None) -> None:
    """Retune L1-edge thresholds and chunk sizes for grids smaller than the
    11x10 Blackhole baseline (e.g. Wormhole 8x8 has ~55% of its aggregate L1),
    so chunking kicks in early enough to avoid L1/CB clashes."""
    global _IS_SMALL_GRID, SEQ_LEN_MORE_CHUNKING, TRANSITION_BATCH_CHUNKING_THRESHOLD
    global TRANSITION_W_CHUNKING_THRESHOLD, TRIANGLE_ATT_CHUNK_SIZE_FAST
    global TRANSITION_W_CHUNK_SIZE, TRIANGLE_MULT_L1_MAX_SEQ_FAST, SMALL_GRID_SEQ_TILE
    global SMALL_GRID_PAIR_TILE_AREA, SMALL_GRID_MSA_TILE_AREA, TRIANGLE_MULT_L1_MAX_SEQ
    global TRANSITION_L1_CHUNK_BYTES_PER_CORE
    _IS_SMALL_GRID = grid[0] * grid[1] < COMPUTE_GRID_X_11 * COMPUTE_GRID_Y
    if not _IS_SMALL_GRID:
        return  # Keep Blackhole baseline values
    # Scale every budget to this part's actual per-core unreserved L1, clamped to
    # <= the full-L1 calibration (so an ample-L1 Wormhole is byte-for-byte
    # unchanged — no perf regression — and only a tighter part, e.g. the Galaxy,
    # tightens). Values are 32-tile-aligned; at full L1 each budget below reduces
    # to its 32-snapped target (640/512/320/288/256/65536). The L=1024 --fast ESMC
    # budgets (SEQ_TILE/PAIR_TILE_AREA) likewise track the available L1.
    try:
        _l1 = ttnn.get_max_worker_l1_unreserved_size()
    except Exception:
        _l1 = _WH_FULL_L1_PER_CORE
    _s = min(1.0, max(_MIN_L1_SCALE, _l1 / _WH_FULL_L1_PER_CORE))
    _snap = lambda v, q=32: max(q, int(round(v * _s / q)) * q)
    SEQ_LEN_MORE_CHUNKING = _snap(640)
    TRANSITION_BATCH_CHUNKING_THRESHOLD = _snap(640)
    TRANSITION_W_CHUNKING_THRESHOLD = _snap(640)
    TRIANGLE_ATT_CHUNK_SIZE_FAST = _snap(512)
    TRANSITION_W_CHUNK_SIZE = _snap(512)
    TRIANGLE_MULT_L1_MAX_SEQ_FAST = _snap(320)  # stays a multiple of TRIANGLE_MULT_CHUNK_SIZE (32)
    # Non-fast triangle-mult keeps the pair tensor in L1 up to this seq; the
    # non-fast kernel is less L1-efficient than the fast one, so on a tight grid
    # it must spill to DRAM at least as early. Without this, a complex padding to
    # ~290-352 tokens (e.g. a protein+2xDNA at 259 tokens -> 320) keeps a too-big
    # pair tensor resident and clashes, while smaller (<=256) and larger
    # (>352 -> already DRAM) complexes fold — a non-monotonic L1 cliff.
    TRIANGLE_MULT_L1_MAX_SEQ = min(_snap(288), TRIANGLE_MULT_L1_MAX_SEQ_FAST)
    SMALL_GRID_SEQ_TILE = _snap(256)
    SMALL_GRID_PAIR_TILE_AREA = _snap(65536, 1024)  # area = rows*L; rows snapped downstream
    SMALL_GRID_MSA_TILE_AREA = _snap(262144, 1024)  # area = rows*M; rows snapped downstream
    # Referenced to the L1 it was measured at, not to _WH_FULL_L1_PER_CORE: _s is already a
    # 1.5 MiB ratio, and reusing it here would shrink a budget that was fitted at 1,466,080 B
    # a second time. Clamped to 1.0 -- a part with more L1 may well fit a bigger chunk, but
    # nothing above 384 KiB has been measured to fit, so do not extrapolate upwards.
    TRANSITION_L1_CHUNK_BYTES_PER_CORE = int(
        _TRANSITION_L1_CHUNK_BYTES_BASE * min(1.0, _l1 / _WH_MEASURED_L1_PER_CORE))
    # Same treatment for the tri-att row-chunk spill threshold: a part with less L1 per core
    # must let go of the block sooner, and one at the calibration L1 keeps the measured value.
    global TRIATT_CHUNK_L1_SPILL_BYTES
    TRIATT_CHUNK_L1_SPILL_BYTES = int(
        _TRIATT_CHUNK_L1_SPILL_BASE * min(1.0, _l1 / _WH_FULL_L1_PER_CORE))

    # SEQ_LEN_MORE_CHUNKING is the one budget above whose resource is NOT per-core L1. It bounds
    # how many full pair tensors are live at once (the code comment on the path it guards says it
    # drops "from 3 full pair tensors live to 2"), and that resource is DRAM capacity. Scaling it by
    # per-core L1 put it at 608 on the Galaxy, so everything from 640 to 1024 aa -- exactly the range
    # JapanFold's max_residues 1024 opens -- took a chunked path nobody had measured against the
    # unchunked one.
    #
    # MEASURED at 1024 aa on that part. Footprint, perf/whb2/out/cap_wh/: unchunked peaks at
    # 3.461 GiB of 12.0, chunked at 2.962. Wall, perf/whb2/out/leverC_wh/: 183.371 s unchunked
    # against 211.532 s chunked, and 77.571 against 87.043 at 640 aa. So the chunking buys 0.499 GiB,
    # 4.2 % of the part's DRAM, and costs 28.161 s, 15 % of the wall, on a chip with 8.5 GiB spare.
    # It was protecting nothing in this range.
    #
    # Re-expressed against DRAM and anchored on that measurement rather than on a chosen safety
    # margin: the pair tensors dominate the peak and scale as L^2, so the largest L whose unchunked
    # trunk sits at the same fraction of DRAM that L=1024 was actually validated at (3.461/12.0 =
    # 28.8 %, taken as 1/3) is 1024 * sqrt((dram/3) / 3.461 GiB). On this part that is 1088, which
    # covers every size the service serves with the 640 and 1024 aa points directly measured.
    # `max` keeps it from ever landing below today's value and `min` from exceeding the Blackhole
    # baseline. Blackhole does not reach this code at all -- the function returns early on a
    # full-size grid -- so its neutrality is by construction, not by clamp.
    _GIB = 2 ** 30
    try:
        _mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        _dram = int(_mv.total_bytes_per_bank) * int(_mv.num_banks)
    except Exception:
        _dram = 0
    if _dram:
        _l = int(1024 * ((_dram / 3.0) / (3.461 * _GIB)) ** 0.5)
        SEQ_LEN_MORE_CHUNKING = min(1536, max(SEQ_LEN_MORE_CHUNKING, (_l // 32) * 32))

    # Lever C's screen hook. Every budget above is scaled by this part's per-core L1, which is the
    # right resource for the L1-edge ones and, for SEQ_LEN_MORE_CHUNKING specifically, arguably the
    # wrong one: that gate bounds how many full pair tensors are live at once, and the resource
    # behind THAT is DRAM capacity (a Galaxy Wormhole chip has ~12 GB against a p150a's 32 GB), not
    # per-core L1 (which is 104.5 % of Blackhole's). Whether the chunked path this part takes above
    # 608 is actually faster than the unchunked one has never been measured, so the screen needs to
    # force the constant without editing it. Unset in production; the scaled value is unchanged.
    _c = os.environ.get("TT_BIO_SEQ_LEN_MORE_CHUNKING")
    if _c:
        SEQ_LEN_MORE_CHUNKING = int(_c)


def _configure_active_compute_grid(device: ttnn.Device) -> None:
    """Snap to a tuned 13x10 or 11x10 Blackhole grid when available; on smaller
    archs (e.g. Wormhole B0 8x8 with ETH dispatch) adopt the device's grid."""
    global CORE_GRID_MAIN, COMPUTE_GRID_MAIN

    gx, gy = COMPUTE_GRID_X_11, COMPUTE_GRID_Y
    try:
        a = device.compute_with_storage_grid_size()
        ax, ay = int(a.x), int(a.y)
        if ax >= COMPUTE_GRID_X_13:
            gx = COMPUTE_GRID_X_13
        elif ax < COMPUTE_GRID_X_11 or ay < COMPUTE_GRID_Y:
            gx, gy = ax, ay
    except Exception:
        pass
    # TT_BIO_FORCE_GRID="x,y" (default off): pin the main grid, e.g. 11,10 on a 13x10
    # p150a to discriminate grid-path defects from hardware (issue #9).
    _force = os.environ.get("TT_BIO_FORCE_GRID")
    if _force:
        gx, gy = (int(v) for v in _force.split(","))

    if (gx, gy) == COMPUTE_GRID_MAIN:
        return

    CORE_GRID_MAIN = ttnn.CoreGrid(y=gy, x=gx)
    COMPUTE_GRID_MAIN = (gx, gy)
    _apply_grid_thresholds((gx, gy), device)
    _sdpa_program_config.cache_clear()
    _sdpa_program_config_for_lengths.cache_clear()
    _triangle_mul_program_config.cache_clear()
    _tri_att_qkv_l1_config.cache_clear()
    _l1_bank_bytes.cache_clear()
    _pair_proj_program_config.cache_clear()
    _attn_value_program_config.cache_clear()


def set_fast_mode(enabled: bool) -> None:
    """Set fast block-fp8 mode for the current worker process."""
    global _FAST_MODE
    _FAST_MODE = bool(enabled)


@contextlib.contextmanager
def device_dtype_override(dtype):
    """Temporarily override the dtype used while constructing a device module."""
    global _DTYPE_OVERRIDE
    old = _DTYPE_OVERRIDE
    _DTYPE_OVERRIDE = dtype
    try:
        yield
    finally:
        _DTYPE_OVERRIDE = old


@contextlib.contextmanager
def diffusion_fp32_device(enabled: bool):
    """Scope the default-off fp32 token-diffusion hybrid (fp32 storage, native bf16
    SDPA) to one model load — used for both the affinity diffusion (BOLTZ2_AFFINITY_
    DIFFUSION_FP32_DEVICE) and the plain structure diffusion (BOLTZ2_STRUCTURE_
    DIFFUSION_FP32_DEVICE); see worker.py load_model/predict_affinity."""
    global _DIFFUSION_FP32_DEVICE
    old = _DIFFUSION_FP32_DEVICE
    _DIFFUSION_FP32_DEVICE = bool(enabled)
    try:
        yield
    finally:
        _DIFFUSION_FP32_DEVICE = old


@lru_cache(maxsize=1)
def arch_name() -> str:
    """Tenstorrent architecture, e.g. 'wormhole_b0' or 'blackhole'. Cheap; no
    device open required."""
    return ttnn.get_arch_name()


@lru_cache(maxsize=1)
def num_chips() -> int:
    """Physical Tenstorrent chips on this host (Galaxy = 32, QuietBox = 4, ...)."""
    import glob as _glob
    return len(_glob.glob("/dev/tenstorrent/[0-9]*"))


def is_wormhole() -> bool:
    return arch_name() == "wormhole_b0"


def msa_row_tile(L: int, M: int) -> int:
    """Rows per block for an MSA [B,L,M,*] row-independent op so the transient
    (~rows*M*width) stays bounded as L and the MSA depth grow. Returns 0 (single
    pass) on big grids or when L*M is already small enough. 32-tile-aligned."""
    area = SMALL_GRID_MSA_TILE_AREA
    if not area or L * M <= area:
        return 0
    rows = max(32, (area // max(M, 1) // 32) * 32)
    return rows if rows < L else 0


# The largest [B,L,M,c] MSA-encoder buffer a 12 GB Wormhole chip is MEASURED to
# allocate. At L=640 with the default depth 8192 the fold succeeds (230.4 s); at L=788,
# same depth, it fails on a 1,652,555,776 B request -- which is exactly 788*8192*128*2 --
# against a 135.9 MiB largest free block (state/japanfold-esmfold2-wh-unusable.md, S15).
# Both numbers are the same tensor, so the thing to bound is the product L*M, not a
# residue count: 640*8192 caps that buffer at the 1.25 GiB that demonstrably fits.
WORMHOLE_MSA_AREA = 640 * 8192


def msa_depth_cap(num_residues: int, max_sequences: int) -> int:
    """MSA depth to actually use for an ESMFold2 fold of this many residues.

    Returns ``max_sequences`` unchanged everywhere except a Wormhole chip asked for an
    L*M above the measured-good product, where it returns the depth that fits. A 1024 aa
    fold gets a 5120-deep MSA instead of an allocation failure; anything at or below the
    proven point, and every Blackhole fold, is untouched.

    A shallower MSA costs accuracy, so this only ever fires where the alternative is no
    structure at all. Depth is bounded, never raised.
    """
    if num_residues <= 0 or max_sequences <= 0 or not is_wormhole():
        return max_sequences
    return max(1, min(max_sequences, WORMHOLE_MSA_AREA // num_residues))


def pair_row_tile(L: int) -> int:
    """Rows per tile for a pair [B,L,L,*] row-independent op so the transient
    (~rows*L*width) stays bounded as L grows. Returns 0 (single pass) on big
    grids or when L is already small enough. 32-tile-aligned."""
    area = SMALL_GRID_PAIR_TILE_AREA
    if not area or L <= SMALL_GRID_SEQ_TILE:
        return 0
    rows = max(32, (area // L // 32) * 32)
    return rows if rows < L else 0


_device = None
_trace_region_size = 0
_device_lease = None
# Bumped on every device close. Module-level device-tensor caches (e.g.
# protenix._WIN_KV_IDX) must key on this: a tensor created on a closed MeshDevice
# keeps the dead mesh alive and throws SubDeviceManagerTracker on next use.
_device_generation = 0

_DEVICE_INIT_LOCK_PATH = "/tmp/tt-bio-device-open.lock"


@contextlib.contextmanager
def _device_init_lock():
    """Serialize TT device bring-up/teardown across every process on the host.

    Opening (or closing) a chip runs through the user-mode driver's cross-process
    device-init path: tt::umd::LocalChip::start_device -> LockManager::acquire_mutex,
    coordinating via robust mutexes in /dev/shm (TT_UMD_LOCK.*). That path is NOT
    concurrency-safe on a Galaxy, in two ways:
      * it deadlocks when several processes hit it at once (observed live: many
        design-shard cold-opens all blocked in acquire_mutex during start_device);
      * it races the per-chip fabric/MMIO bring-up, so a chip can come up
        "remote-only" — no local dispatch core, SubDeviceManagerTracker never
        initialized — and then throws on the FIRST program dispatch
        ("...contains only remote devices (no local device)", mesh_device.cpp).
    The platform opens 32 single-chip workers at startup, so WITHOUT serialization
    that race is hit on many boots (a few workers come up bad and silently fail
    every job routed to them). Serializing the opens is the fix.

    A single host-wide advisory lock makes every open/close strictly one-at-a-time,
    so the UMD init path is never raced. Blocking on purpose: a best-effort timeout
    that let opens proceed concurrently after waiting is exactly what reintroduced
    the deadlock. The kernel drops the lock if a holder dies, and the pool
    supervisor + per-run stall watchdog bound any pathological case, so this can
    never wedge worse than opening unserialized. Opens are one-time per worker (the
    chip is reused for every job, predict AND design — design runs in-process on the
    already-open chip, never cold-opening), so serialization only lengthens startup
    slightly and never adds any runtime latency."""
    import fcntl
    try:
        f = open(_DEVICE_INIT_LOCK_PATH, "w")
    except Exception:
        yield  # can't create the lock file -> don't block bring-up
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX)  # wait our turn; do NOT proceed concurrently
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        except Exception:
            pass


def _open_device_locked(device_id, kwargs):
    """Open + configure the device with bring-up strictly serialized host-wide."""
    with _device_init_lock():
        dev = ttnn.open_device(device_id=device_id, **kwargs)
        _configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


def _assert_local_dispatch(dev):
    """Verify a freshly-opened chip can actually dispatch a program.

    A chip that came up "remote-only" from a raced bring-up opens fine but throws on
    the first program dispatch (SubDeviceManagerTracker not initialized / "only
    remote devices"). Probe with one trivial op so a mis-initialized worker fails
    HERE, at startup, and gets respawned with a serialized clean reopen — instead of
    silently accepting jobs it will fail. Runs unlocked: it's an ordinary compute
    dispatch on an already-open chip, not the UMD init path, so it needn't serialize
    (the tiny kernel is cached after the first compile)."""
    import torch
    try:
        t = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.add(t, t)
        ttnn.synchronize_device(dev)
    except Exception as e:
        with _device_init_lock():
            try:
                ttnn.close_device(dev)
            except Exception:
                pass
        raise RuntimeError(f"device bring-up failed the local-dispatch check "
                           f"(likely a remote-only init): {e}") from e


def get_device(trace_region_size=0):
    """Open (or return cached) TT device 0.

    Worker processes set TT_VISIBLE_DEVICES before importing ttnn, so the
    assigned physical chip appears as logical device 0.

    trace_region_size: bytes to reserve for ttnn trace capture. Pass a nonzero
    size (e.g. 1 << 30) to enable the Protenix denoise trace via fold(trace=True)
    or BoltzGen's diffusion trace (Boltz.__init__(diffusion_trace=True)); the
    default 0 leaves the device layout unchanged. If the arg is 0,
    ``TT_BIO_TRACE_REGION_SIZE`` is consulted as a dev-only escape hatch so a
    single-BH open can reserve a trace region without the caller threading the
    kwarg.
    """
    global _device, _trace_region_size, _device_lease
    if _device is None:
        # Enforce an exclusive, host-local lease on the physical card BEFORE opening it,
        # so two processes on this host can never open the same card at once regardless of
        # how they were launched (fleet worker, detached campaign, cross-host fanout, manual).
        # The flock is auto-released by the kernel on any process death, so a crashed/killed
        # holder never leaves a phantom claim. See tt_bio/device_lease.py.
        # And a process that takes a card must not outlive whoever wanted the result:
        # an orphaned holder keeps its flock and defers every later job on that card.
        from tt_bio.device_lease import DeviceLease, arm_orphan_guard
        arm_orphan_guard()
        _device_lease = DeviceLease().acquire()
        try:
            _device = _open_and_init_device(trace_region_size)
        except Exception:
            _device_lease.release()
            _device_lease = None
            raise
    return _device


_DRAM_PEAK = {}   # tag -> high-water device DRAM bytes, when TT_BIO_DRAM_PEAK is set


def dram_peak(tag=None):
    """Record (and return) the device DRAM high-water mark, in bytes.

    Off unless TT_BIO_DRAM_PEAK names a file to append samples to, so production folds pay
    nothing. A FILE and not stdout because `tt-bio predict` runs the fold in a spawned
    worker whose stdout the live-progress view owns (and drops when it is not a TTY), so a
    printed measurement is invisible exactly when it is being collected non-interactively.

    The ttnn allocator is host-side bookkeeping, but reading it is NOT cheap enough to
    time under: get_memory_view behaves like a pipeline drain, so a timed run with this
    enabled measures the probe, not the model (measured 2026-08-07 on a 117-aa protenix
    fold: 12.0 s with the probe off vs 28.8 s on with main's sparse tags, 44.7 s with the
    denser census tags — never A/B perf with this set). This is what the release gate's
    capacity leg reads: a footprint change is invisible to a numerical parity fixture, so
    the footprint has to be measured directly at the largest supported input.
    Call with no tag to read the current peak across all tags.

    Lives here rather than in a model module because the trunk (MSA/Pairformer, below) is
    the largest DRAM consumer and cannot import a model module without a cycle."""
    path = os.environ.get("TT_BIO_DRAM_PEAK")
    if not path:
        return 0
    if tag is not None:
        mv = ttnn.get_memory_view(get_device(), ttnn.BufferType.DRAM)
        used = (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks
        if used > _DRAM_PEAK.get(tag, 0):
            _DRAM_PEAK[tag] = used
            # Largest contiguous free block per bank (min over banks): the binding
            # constraint for an interleaved allocation is size/12 contiguous in EVERY
            # bank, so this -- not total free -- decides whether a big request is
            # refused after allocator churn. Diagnostic only; the release gate's regex
            # anchors on "GiB used" and ignores the suffix.
            lcf = mv.largest_contiguous_bytes_free_per_bank
            if isinstance(lcf, (list, tuple)):
                lcf = min(lcf)
            line = (f"[DRAM] {tag}: {used / 2**30:.3f} GiB used "
                    f"(of {mv.total_bytes_per_bank * mv.num_banks / 2**30:.1f} GiB) "
                    f"maxfree={lcf / 2**20:.0f}MiB/bank\n")
            try:
                with open(path, "a") as fp:      # append: the worker is a separate process
                    fp.write(line)
            except OSError:
                pass                            # a diagnostic must never break a fold
    return max(_DRAM_PEAK.values(), default=0)


def _host_concat(x: ttnn.Tensor) -> bool:
    """Whether a chunked path whose output is x's shape assembles its blocks on the host.

    See CONCAT_HOST_BYTES. bf16 only: a bf8 (fast-mode) or fp32 (affinity) block would
    round-trip through torch bf16 lossily, so those configs keep the device concat.
    """
    return (x.dtype == ttnn.bfloat16 and _dtype() == ttnn.bfloat16
            and x.logical_volume() * 2 > CONCAT_HOST_BYTES)


def _acc_concat(acc: list, dim: int, host: bool) -> ttnn.Tensor:
    """Assemble accumulated row/channel blocks, on the host when they were offloaded.

    Host branch: the blocks are torch tensors (bit-identical bytes); the upload is one
    full-size allocation made when the accumulator holds nothing on device. Device
    branch: ttnn.concat, then free the blocks (same as the call sites always did).
    """
    if host:
        return ttnn.from_torch(
            torch.cat(acc, dim=dim), layout=ttnn.TILE_LAYOUT,
            device=get_device(), dtype=ttnn.bfloat16)
    if len(acc) == 1:
        # ttnn.concat of a single tensor aliases its input, so the deallocate below would
        # free the very buffer being returned. One block is the real case for any trimul
        # whose hidden width equals its chunk width (n_pairs == 1, e.g. the protenix
        # template pair stack) and for a row-blocked tail shorter than one row block.
        return acc[0]
    out = ttnn.concat(acc, dim=dim)
    for t in acc:
        ttnn.deallocate(t)
    return out


_TRIATT_CHUNK_L1_SPILL_BASE = 524288          # 512 KiB per core, at full 1.5 MiB L1
TRIATT_CHUNK_L1_SPILL_BYTES = _TRIATT_CHUNK_L1_SPILL_BASE


def _chunk_l1_per_core(t: ttnn.Tensor) -> int:
    """Bytes of L1 this interleaved block occupies on each core of the main grid.

    Tiles are spread round-robin over the grid, so a core holds ceil(tiles / cores) of them.
    A bf16 tile is 2048 B and a bfloat8_b tile is 1088 B (1024 mantissa bytes + 64 of shared
    exponents), which is why the same block is a different budget on the two arms.
    """
    tile = 1088 if t.dtype == ttnn.bfloat8_b else (4096 if t.dtype == ttnn.float32 else 2048)
    shape = [int(d) for d in t.padded_shape]
    tiles = 1
    for d in shape[:-2]:
        tiles *= d
    tiles *= -(-shape[-2] // 32) * -(-shape[-1] // 32)
    cores = max(1, COMPUTE_GRID_MAIN[0] * COMPUTE_GRID_MAIN[1])
    return -(-tiles // cores) * tile


def _acc_append(acc: list, t: ttnn.Tensor, host: bool) -> None:
    """Add a produced block to the accumulator, offloading it when host-assembling."""
    if host:
        acc.append(ttnn.to_torch(t))
        ttnn.deallocate(t)
    else:
        acc.append(t)


def _open_and_init_device(trace_region_size):
    """Open + configure TT device 0 (the physical card is already leased by the caller)."""
    global _trace_region_size
    if trace_region_size == 0:
        env_sz = os.environ.get("TT_BIO_TRACE_REGION_SIZE")
        if env_sz:
            trace_region_size = int(env_sz)
    if trace_region_size >= 2 ** 32:
        # A trace region of exactly 4 GiB or more wedges tt-metal instead of erroring: the
        # capture records fine, then end_trace_capture blocks forever inside
        # MeshTrace::populate_mesh_buffer -> enqueue_write_shard_to_sub_grid -> finish(), and
        # the completion-queue reader thread spins at 100 % CPU on a completion that never
        # arrives. Measured on qb2 / ttnn 0.68.0 / P300: 3.9 GiB closes a capture in 2.6 ms,
        # 4.0 GiB never closes, for a bare ttnn.add with no model. Refuse it here rather than
        # let a caller hang, and keep the byte count so the message is unambiguous.
        raise ValueError(
            f"trace_region_size={trace_region_size} is >= 2**32; tt-metal truncates it and "
            "end_trace_capture never returns. Use 1 GiB (what the denoiser trace asks for).")
    device_id = int(os.environ.get("TT_BIO_LOGICAL_DEVICE_ID", "0"))
    # A lone P300 chip is a custom topology and open_device() is a TT_FATAL without a mesh
    # graph descriptor ("Custom fabric mesh graph descriptor path must be specified for CUSTOM
    # cluster type"). tt-bio's CLI entry points set this in the parent before spawning workers;
    # anything that reaches get_device() directly -- every tool under perf/, every ad-hoc
    # script -- did not, and got the TT_FATAL. Setting it here covers both, and an explicit
    # TT_MESH_GRAPH_DESC_PATH still wins. The import is function-local because tt_bio.main
    # imports this module.
    from .main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    # Wormhole: dispatch on Ethernet cores so the full 8x8 Tensix grid
    # (rather than 8x7 after worker-dispatch reservation) is available.
    # BUT on a multi-chip system (Galaxy / multi-card mesh) the ETH cores are
    # consumed by the inter-chip fabric, so ETH dispatch has no free cores
    # ("No more available dispatch cores"). We must NOT attempt-then-reopen in
    # the same process: the failed ETH open leaves the device mid-initialized
    # ("dispatch kernels still running", "unexpected run_mailbox value") and a
    # subsequent in-process open is unstable (later from_torch / close_device
    # hang). So decide up front from the physical chip count and open cleanly
    # once. Default (Tensix) dispatch yields an 8x7 grid that
    # _configure_active_compute_grid picks up and tunes for.
    eth_dispatch = is_wormhole() and num_chips() <= 1
    kwargs = (
        {"dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)}
        if eth_dispatch else {}
    )
    # Opt-in ttnn trace region for the Protenix denoise trace (dispatch-bound
    # diffusion). Default 0 -> device layout unchanged when tracing is off.
    if trace_region_size > 0:
        kwargs["trace_region_size"] = trace_region_size
    dev = _open_device_locked(device_id, kwargs)
    _assert_local_dispatch(dev)   # raises (and closes) on a remote-only bring-up
    _trace_region_size = trace_region_size
    return dev


def trace_region_size():
    """Bytes reserved for ttnn trace on the open device (0 if none / no device)."""
    return _trace_region_size


def device_generation():
    """Monotonic id of the current device lifetime; changes when cleanup() closes
    the device. Cache keys for module-level device tensors must include it."""
    return _device_generation


def cleanup():
    global _device, _trace_region_size, _device_lease, _device_generation
    if _device is not None:
        try:
            # Drain queued work before closing so teardown is deterministic.
            ttnn.synchronize_device(_device)
        except Exception:
            pass
        # Closing also runs through the UMD device path, so serialize it with
        # opens (see _device_init_lock): a close racing an open on another chip
        # contends on the same cross-process init mutexes.
        with _device_init_lock():
            ttnn.close_device(_device)
        _device = None
        _trace_region_size = 0
        _device_generation += 1
    # Release the physical-card lease AFTER the chip is closed, so the card is not
    # advertised as free while UMD teardown is still in flight. The kernel also drops
    # the flock on process exit, so a skipped cleanup (e.g. SIGKILL) still frees it.
    if _device_lease is not None:
        _device_lease.release()
        _device_lease = None


atexit.register(cleanup)


class WeightScope:
    """Immutable scoped view over a flat checkpoint state-dict."""

    def __init__(self, data: Mapping[str, torch.Tensor]):
        self._data = MappingProxyType(dict(data))

    @classmethod
    def wrap(cls, data: Mapping[str, torch.Tensor] | "WeightScope") -> "WeightScope":
        return data if isinstance(data, cls) else cls(data)

    @property
    def data(self) -> Mapping[str, torch.Tensor]:
        return self._data

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._data)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def child(self, scope: str, strip_prefix: str = "") -> "WeightScope":
        if not scope:
            return self
        scope_prefix = f"{scope}."
        out = {}
        for key, value in self._data.items():
            if not key.startswith(scope_prefix):
                continue
            child_key = key[len(scope_prefix) :]
            if strip_prefix and child_key.startswith(strip_prefix):
                child_key = child_key[len(strip_prefix) :]
            out[child_key] = value
        return WeightScope(out)


Weights = Mapping[str, torch.Tensor] | WeightScope

# Process-global shared tiled-weight cache. None disables it (default: every model
# tiles+uploads its own weights via from_torch). Set via weight_cache() around a
# module construction to route torch_to_tt through a /dev/shm tile cache shared by
# data-parallel fanout workers (see torch_to_tt and esmc.load_esmc6b_shared).
_weight_cache = None


@contextlib.contextmanager
def weight_cache(cache_dir: str, mode: str):
    """Route torch_to_tt through a shared tiled-weight cache in ``cache_dir``.

    mode="dump": tile on host and publish each weight to the cache (then upload).
    mode="load": load each pre-tiled weight from the cache and upload (no host tiling,
    no checkpoint read). The call-index ordering must match between dump and load,
    which holds because both construct the identical module.
    """
    global _weight_cache
    prev = _weight_cache
    _weight_cache = {"dir": cache_dir, "mode": mode, "counter": 0}
    try:
        yield
    finally:
        _weight_cache = prev


class Module:
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.weights = WeightScope.wrap(state_dict)
        self.compute_kernel_config = compute_kernel_config
        self.device = get_device()

    def scope(self, scope: str, strip_prefix: str = "") -> WeightScope:
        return self.weights.child(scope, strip_prefix)

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
        dtype=None,
    ) -> ttnn.Tensor:
        if dtype is None:
            dtype = _dtype(ttnn.bfloat16)
        wc = _weight_cache
        if wc is None:
            return ttnn.from_torch(
                transform(self.weights[key]),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=dtype,
            )
        # Shared tiled-weight cache (data-parallel fanout): every worker constructs
        # the identical module in the identical order, so the running call index is a
        # stable key. The builder tiles fp32->device-dtype on host once and dumps the
        # tile to /dev/shm; peers load_tensor it (RAM, no disk read, no re-tiling) and
        # only pay the per-card DMA. Bit-exact: the dumped tile is what from_torch
        # would have produced.
        i = wc["counter"]
        wc["counter"] = i + 1
        path = os.path.join(wc["dir"], f"{i}.tensorbin")
        if wc["mode"] == "load":
            host = ttnn.load_tensor(path)
        else:
            host = ttnn.from_torch(transform(self.weights[key]), layout=ttnn.TILE_LAYOUT, dtype=dtype)
            tmp = f"{path}.{os.getpid()}.tmp.tensorbin"  # dump_tensor requires a .tensorbin name
            ttnn.dump_tensor(tmp, host)
            os.replace(tmp, path)  # atomic publish
        return ttnn.to_device(host, self.device)

    def _lin(self, x, w, bias=None, dtype=None, **kw):
        """Shared linear projection on this module's kernel config and core grid."""
        if dtype is None:
            dtype = _dtype(ttnn.bfloat16)
        return ttnn.linear(
            x, w, bias=bias, compute_kernel_config=self.compute_kernel_config,
            dtype=dtype, core_grid=CORE_GRID_MAIN, **kw,
        )

    def _split_heads(self, qkv, n_heads):
        """Packed [B, L, 3*d] -> per-head (q, k, v) [B, H, L, d_head] via the
        tile-aware nlp head split. Frees the packed input."""
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=n_heads, num_kv_heads=n_heads,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _merge_heads(self, ctx):
        """Per-head ctx [B, H, L, d_head] -> [B, L, H*d_head]."""
        return ttnn.squeeze(
            ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1
        )


def _in_proj_matmul(x, w, ckc, memory_config):
    """The trimul in-projection: the dual-NOC drain where it applies, else today's call.

    `mm_dualnoc.in_proj` is byte-identical to the call below when it fires -- it drives the same
    kernels through `generic_op` with `_MM_DEFAULT`, which IS the block config
    `determine_default_block_sizes` returns for an unconfigured `minimal_matmul` under
    `fp32_dest_acc_en`. It returns None on anything outside the class it was verified on.
    """
    from . import mm_dualnoc as DN
    out = DN.in_proj(x, w, ckc, _dtype(), memory_config)
    if out is not None:
        return out
    return ttnn.experimental.minimal_matmul(
        x, w, memory_config=memory_config, dtype=_dtype(), compute_kernel_config=ckc)


def _trimul_out_proj(
    x: ttnn.Tensor, weight: ttnn.Tensor, ckc: ttnn.DeviceComputeKernelConfig
) -> ttnn.Tensor:
    """The trimul output projection: [1,L,L,c_z] @ [c_z,c_z], no bias.

    `ttnn.linear(core_grid=...)` reaches 20.6 TFLOP/s on this shape where
    `minimal_matmul` reaches 35.7 (perf/trimul_kernel/layout_micro.py), the same
    1.7x gap the tri-attention projections show. Not bit-exact: the two kernels
    block the contraction differently, so bf16 accumulates in a different order.
    """
    if _TRIMUL_MM_OUT:
        return ttnn.experimental.minimal_matmul(
            x, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=_dtype(),
            compute_kernel_config=ckc,
        )
    # Both of a trimul's output projections take an L1 result: `multiply_` folds them together
    # in place and the layer's residual `add_` reads the product, so neither ever needs to reach
    # DRAM. Two live 48.82 MB L1 tensors is 750.9 kB of each bank's 1427.5 kB, which fits beside
    # this config's circular buffers and does not beside `core_grid=`'s.
    return _pair_proj_linear(x, weight, ckc, _dtype(), l1_out=True)


# F1: the tail's two output projections and its gate in one `generic_op` (`tt_bio/trimul_tail.py`).
# `p_out` and `g_out` never become tensors, so the tail reads 2 pair tensors and writes 1 instead of
# 4 and 3. Bit-exact against the three ops it replaces -- `torch.equal` at 11 shapes from N=32 to
# 576, `perf/trimul_f1/f1_parity.py`.
# It is real but small: -679.47 ms on the trimul body wall at 512 aa, against a 33.46 ms
# A/A floor, byte-identical CIF (`perf/trimul_f1/fold_ab_f1_main_qb1c1.json`). Deleted bytes
# at this site return ~39 % of what the 271.5 GB/s write roof prices them at, which is why the
# levers that would have been stacked on top of it (both layer norms inside the same kernel) were
# repriced to ~-0.39 s each and not built.
# Default ON since 2026-08-15, when the release gate ran: `torch.equal` at 14 shapes from N=32 to
# 1024 on the release box, and at 512 aa esmfold2/protenix-v2 fire it with byte-identical CIFs
# while boltz2/opendde/openfold3 never fire it at all (their trimuls are not kt=8, so the three
# ops below run unchanged). ESMFold2 512 aa page cell 32.329 -> 31.994 s, median of 3
# (`perf/trimul_f1/page_esmfold2_f1_qb2c2.json`).
TRIMUL_TAIL_F1 = True
_TRIMUL_TAIL_F1 = os.environ.get(
    "TT_BIO_TRIMUL_TAIL_F1", "1" if TRIMUL_TAIL_F1 else "0") == "1"


def _channel_move_back(chunk: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """``permute(chunk, (0, 2, 3, 1))``, through the hand-written kernel where it wins.

    The stock decomposition is ``transpose(1,2)`` then ``transpose(2,3)``, which reads and writes the
    whole tensor twice; the kernel does it in one pass. Bit-exact against either
    (``torch.equal``): both are a pure index reordering plus the same within-tile ``transpose_wh``.
    Outside ``reblock_permute.eligible_back`` the two transposes stay.
    """
    if _reblock.eligible_back(chunk, memory_config):
        return _reblock.reblock_permute_back(chunk, memory_config)
    out = ttnn.transpose(chunk, 1, 2, memory_config=memory_config)
    res = ttnn.transpose(out, 2, 3, memory_config=memory_config)
    ttnn.deallocate(out)
    return res


def _channel_move(chunk: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """``permute(chunk, (0, 3, 1, 2))``, through the hand-written kernel where it wins.

    Same index move as ``ttnn.permute`` and bit-exact against it (``torch.equal``). It wins because
    it issues the 64 unavoidable NOC transactions per source tile from 100 cores. It only wins in a
    shape window, so everything outside ``reblock_permute.eligible`` keeps the stock call; see
    ``tt_bio/reblock_permute.py`` for the measured gate.
    """
    if _reblock.eligible(chunk, memory_config):
        return _reblock.reblock_permute(chunk, memory_config)
    return ttnn.permute(chunk, (0, 3, 1, 2), memory_config=memory_config)


class TriangleMultiplication(Module):
    def __init__(
        self,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        gated_move: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        # Opt in to the fused chunk+gate forward move (E6). Per instance and not a global, because
        # the same kernel wins on opendde's channel widths and loses on boltz2's call mix.
        self.gated_move = gated_move
        self.in_norm_weight = self.torch_to_tt("norm_in.weight")
        self.in_norm_bias = self.torch_to_tt("norm_in.bias")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        g_in_t, p_in_t = [
            self.weights[k].t() for k in ["g_in.weight", "p_in.weight"]
        ]
        # The chunk width is a per-call decision (_trimul_chunk_size), so the fused input
        # weights are built per width on first use and kept. Each variant is the same
        # g_in/p_in bytes in a different column order -- 0.5 MB per trimul at c_z=256 --
        # and a fold holds one sequence length, so in practice one variant exists.
        self._g_in_t, self._p_in_t = g_in_t, p_in_t
        self._hidden = g_in_t.shape[1] // 2
        self._gp_cache: dict[tuple[int, int], list[ttnn.Tensor]] = {}
        self.g_out_weight = self.torch_to_tt("g_out.weight")
        self.out_p_weight = self.torch_to_tt("p_out.weight")

    def _gp_in_chunks(self, C: int, group: int = 1) -> list[ttnn.Tensor]:
        """Fused [g_a | g_b | p_a | p_b] input weights, `group` consecutive chunks per weight.

        The columns are role-major: all `group` chunks of g_a, then g_b, then p_a, then p_b. A
        4-way split hands back four `group * C`-wide pieces, one per role, so the channel loop
        runs `group * C` channels at a time instead of C. Everything downstream is elementwise,
        an index move, or a per-channel matmul, so a wider group is a different partition of the
        same sum and stays bit-exact. At group = 1 the order is the narrow path's.
        """
        cached = self._gp_cache.get((C, group))
        if cached is not None:
            return cached
        g, p = self._g_in_t, self._p_in_t
        n_pairs = g.shape[1] // C // 2
        assert n_pairs % group == 0, f"group {group} does not divide {n_pairs} pairs"
        chunks = [
            ttnn.from_torch(
                torch.cat(
                    [
                        w[:, (j + off) * C : (j + off + 1) * C]
                        for w, off in ((g, 0), (g, n_pairs), (p, 0), (p, n_pairs))
                        for j in range(i * group, (i + 1) * group)
                    ],
                    dim=1,
                ),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            for i in range(n_pairs // group)
        ]
        self._gp_cache[(C, group)] = chunks
        return chunks

    def _transform_chunk(
        self, chunk: ttnn.Tensor, permute_dims: tuple[int, ...], memory_config: ttnn.MemoryConfig,
        realloc: bool = True,
    ) -> ttnn.Tensor:
        # Bring the channel chunk to the batch axis for the per-channel matmul.
        # The two cases are (0,3,1,2) [no inner swap] and (0,3,2,1) [also swaps
        # the inner L,L]. The latter, done as a single ttnn.permute, is ~3x more
        # expensive than (0,3,1,2) on the large-L DRAM path (the inner L,L
        # transpose is DRAM-bandwidth bound: ~10ms vs ~3ms at L=1024). There we
        # decompose it into the cheap channel-move permute (0,3,1,2) followed by
        # ttnn.transpose(-2,-1) (a tile-local op, ~0.2ms) — BIT-EXACT with
        # permute(0,3,2,1) (pure index reordering). On the small-L L1 path the
        # single permute is marginally faster (the extra op's launch overhead
        # outweighs the cheaper transpose), so keep it there.
        inner_swap = permute_dims == (0, 3, 2, 1)
        decompose = (
            inner_swap
            and memory_config.buffer_type == ttnn.BufferType.DRAM
            and not _TRIMUL_RAW_CHANNEL_MOVES
        )
        ops = [(ttnn.typecast, ttnn.bfloat16)] if _FAST_MODE else []
        if decompose:
            ops.append((_channel_move,))
            ops.append((ttnn.transpose, -2, -1))
        elif permute_dims == (0, 3, 1, 2):
            ops.append((_channel_move,))
        else:
            ops.append((ttnn.permute, permute_dims))
        if _FAST_MODE:
            ops.append((ttnn.typecast, ttnn.bfloat8_b))
        # The reallocate compacts the chunk so the NEXT iteration's allocations find contiguous
        # space; with one iteration there is no next one and nothing to fragment. It is a full
        # round trip of the chunk through DRAM (134.2 MB each way at 512 aa, measured 0.711 ms for
        # the pair at 101 % of the combined roof -- at the roof for what it does, and unnecessary).
        if realloc:
            ops.append((ttnn.reallocate,))
        old = chunk
        for op, *args in ops:
            chunk = op(chunk, *args, memory_config=memory_config)
            ttnn.deallocate(old)
            old = chunk
        return chunk

    def _transform_chunk_gated(
        self, gp: ttnn.Tensor, gate: tuple[int, int, int], permute_dims: tuple[int, ...],
        memory_config: ttnn.MemoryConfig, realloc: bool,
    ) -> ttnn.Tensor:
        """`_transform_chunk` when the gate rides along inside the channel move.

        The fused projection is never split and never gated in DRAM: the move's reader takes the
        value and gate slices in place and its compute kernel applies the sigmoid and the multiply
        on the way. That deletes `ttnn.chunk` and both `multiply_` calls, 4.876 ms per call at
        512 aa, all three of them at 95-99 % of the memory roof and so with no tuning left in them.
        Bit-exact against the sequence it replaces, `torch.equal` at 24 shapes; see
        `tt_bio/reblock_permute.py` and `perf/trimul_f2/e6_parity.py`.

        `gp` stays alive: both roles read the same fused projection, so the caller owns it.
        """
        ops = []
        if permute_dims == (0, 3, 2, 1):
            ops.append((ttnn.transpose, -2, -1))
        if realloc:
            ops.append((ttnn.reallocate,))
        chunk = _reblock.reblock_permute_gated(gp, *gate, memory_config=memory_config)
        old = chunk
        for op, *args in ops:
            chunk = op(chunk, *args, memory_config=memory_config)
            ttnn.deallocate(old)
            old = chunk
        return chunk

    def _in_proj_rows(self, x, w, H, batch, memory_config):
        """`LN(x) @ w`, computed in row blocks so the full-size LN'd pair tensor never exists.

        layer_norm normalises over the LAST dim and the matmul contracts the LAST dim, so
        output row r depends only on input row r: the blocks reassemble bit-identically to
        the whole-tensor result, exactly as for the row-blocked output projections below.
        The norm is recomputed once per channel group rather than cached, which trades a
        memory-bound reread for the 3.24 GiB allocation that made the large cells unfoldable.

        Only reached past TRIMUL_IN_NORM_ROWBLOCK_BYTES, so anything whose pair tensor can
        simply be allocated is unchanged: same ops, same order, same allocations.
        """
        out_bytes = batch * H * H * int(w.shape[-1]) * 2
        host = (x.dtype == ttnn.bfloat16 and _dtype() == ttnn.bfloat16
                and out_bytes > CONCAT_HOST_BYTES)
        blocks = []
        for s in range(0, H, PAIR_ROW_BLOCK):
            rows = ttnn.layer_norm(
                x[:, s:min(s + PAIR_ROW_BLOCK, H)],
                weight=self.in_norm_weight,
                bias=self.in_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            _acc_append(blocks, _in_proj_matmul(
                rows, w, self.compute_kernel_config, memory_config), host)
            ttnn.deallocate(rows)
        return _acc_concat(blocks, 1, host)

    def prewarm(self, H: int, batch: int = 1) -> None:
        """Build the fused input-weight cache this shape will use, before the call that uses it.

        `_gp_in_chunks` is otherwise built inside the first `__call__`, interleaved with its
        compute: 96 tensors of (c_z, 4*chunk) across a 4-block Pairformer, ~9.4 MB. Uploading
        them up front costs no compute. Numerically inert, measured: a warm first call and a
        cold one produce the same numbers.

        The (chunk_size, group) pair is computed exactly as `__call__` computes it, from the
        same inputs, so the entry warmed is the entry used.
        """
        memory_config = _triangle_mul_memory_config(H)
        large_seq = memory_config.buffer_type == ttnn.BufferType.DRAM
        chunk_size = _trimul_chunk_size(H, self._hidden, batch)
        n_pairs = self._hidden // chunk_size
        group = _trimul_inproj_group(H, chunk_size, batch, n_pairs) if large_seq else 1
        self._gp_in_chunks(chunk_size, group)

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x_in = x  # keep the pair tensor reachable for the row-blocked tail below
        shp = [int(d) for d in x.shape]
        H = shp[1]
        # Past TRIMUL_IN_NORM_ROWBLOCK_BYTES the LN'd pair tensor is never materialised whole. It
        # was the last full-size pair allocation this op still made: the output projections below
        # have been row-blocked for a while, but the input norm was computed whole on every path.
        # On OpenDDE's structural-token axis that is a [1,2113,2113,384] bf16 request, 3.19 GiB,
        # refused against a 208.7 MiB largest free block -- the AbAg-XM 9j4c cell.
        #
        # Gated on the tensor's OWN bytes, not on SEQ_LEN_MORE_CHUNKING: the row-blocked path costs
        # 43 % per call, and only a tensor too big to allocate is worth paying that for. See the
        # constant.
        row_norm = prod(shp) * (2 if x.dtype == ttnn.bfloat16 else 4) > TRIMUL_IN_NORM_ROWBLOCK_BYTES
        x_norm_in = None if row_norm else ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        dram_peak(f"trimul({'end' if self.ending else 'start'}) x_norm_in [z={'x'.join(str(d) for d in x.shape)}]")
        memory_config = _triangle_mul_memory_config(H)
        # Every L1 tensor the channel loop holds is [batch, chunk, H, H], so the width
        # budget has to see the batch a confidence head arrives with, not just H.
        batch = prod(shp[:-3])
        large_seq = memory_config.buffer_type == ttnn.BufferType.DRAM
        chunk_size = _trimul_chunk_size(H, self._hidden, batch)
        n_pairs = self._hidden // chunk_size
        # The matmul's N and the channel-chunk width are two different numbers this code has been
        # forcing to be one. Only the matmul widens (_TRIMUL_INPROJ_GROUP), and only on the DRAM
        # path: on the L1 path _trimul_chunk_size has already widened the chunk itself and the L1
        # budget it protects is real. Everything below the four-way unpack is unchanged.
        group = (
            _trimul_inproj_group(H, chunk_size, batch, n_pairs) if large_seq else 1
        )
        gp_in_chunks = self._gp_in_chunks(chunk_size, group)
        seq_len_tiles = (H + 31) // 32
        program_config = _triangle_mul_program_config(seq_len_tiles)
        if not row_norm and H > SEQ_LEN_MORE_CHUNKING:
            # Compact large input activation for better large-sequence placement.
            x_norm_in = ttnn.reallocate(x_norm_in)
        # Unsqueeze mask once before chunk loop (mask is [1,S,S] or [1,S])
        mask_u = ttnn.unsqueeze(mask, -1) if mask is not None else None
        # Collect the per-channel output chunks and concat them ONCE at the end. A
        # running concat copies the accumulator on every step (O(n_pairs^2)
        # channel-bytes moved); one concat of all chunks copies each chunk once.
        # Bit-exact either way (same chunk order). On the L1 path the chunks are
        # moved to DRAM as they are produced, so holding all n_pairs of them costs
        # no L1: measured 7.05 -> 6.94 ms per trimul at 298 aa
        # (perf/trimul_kernel/opsplit298.py).
        # Assemble the per-channel chunks on the host when the full result is large
        # enough that the concat's full-size allocation would risk a fragmented-DRAM
        # refusal; the loop then holds at most one chunk on device (CONCAT_HOST_BYTES).
        host_acc = large_seq and _host_concat(x_in)
        # The channel loop's L1 footprint is set by chunk_size, and the budget that
        # picked it is a 130-core calibration: on a tighter grid (110-core p300/p300c)
        # the picked width can clash at program creation (issue #11). The clash throws
        # at program validation, before anything runs, so re-running the loop at a
        # narrower width is safe, and narrowing is bit-exact (a partition of an
        # independent-channel sum). The clash is recorded, so a shape pays one failed
        # compile per process and every later call starts narrow.
        x_chunks = []
        while True:
            try:
                for i in range(n_pairs // group):
                    gp_in_fused = (
                        self._in_proj_rows(x_in, gp_in_chunks[i], H, batch, memory_config)
                        if row_norm else
                        _in_proj_matmul(x_norm_in, gp_in_chunks[i],
                                        self.compute_kernel_config, memory_config)
                    )
                    perm_a = (0, 3) + ((2, 1) if self.ending else (1, 2))
                    perm_b = (0, 3) + ((1, 2) if self.ending else (2, 1))
                    slice_c = int(gp_in_fused.shape[-1]) // 4
                    # The fused path only replaces the (0,3,1,2) move, which is the leg `_transform_chunk`
                    # decomposes to on the DRAM path. A mask multiply or --fast's typecasts would have to
                    # ride inside the kernel too, so those keep the four-way split.
                    gated = (
                        self.gated_move
                        and mask_u is None
                        and not _FAST_MODE
                        and not _TRIMUL_RAW_CHANNEL_MOVES
                        and memory_config.buffer_type == ttnn.BufferType.DRAM
                        and _reblock.eligible_gated(gp_in_fused, slice_c, memory_config)
                    )
                    if gated:
                        a_chunk = self._transform_chunk_gated(
                            gp_in_fused, (2 * slice_c, 0, slice_c), perm_a, memory_config,
                            n_pairs // group > 1,
                        )
                        b_chunk = self._transform_chunk_gated(
                            gp_in_fused, (3 * slice_c, slice_c, slice_c), perm_b, memory_config,
                            n_pairs // group > 1,
                        )
                        ttnn.deallocate(gp_in_fused)
                    else:
                        g_in_a, g_in_b, p_in_a, p_in_b = ttnn.chunk(gp_in_fused, chunks=4, dim=-1)
                        ttnn.deallocate(gp_in_fused)
                        a_chunk = ttnn.multiply_(
                            p_in_a, g_in_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
                        )
                        b_chunk = ttnn.multiply_(
                            p_in_b, g_in_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
                        )
                        ttnn.deallocate(g_in_a)
                        ttnn.deallocate(g_in_b)
                        if mask_u is not None:
                            a_chunk = ttnn.multiply_(a_chunk, mask_u)

                        a_chunk = self._transform_chunk(
                            a_chunk, perm_a, memory_config=memory_config, realloc=n_pairs // group > 1,
                        )
                        b_chunk = self._transform_chunk(
                            b_chunk, perm_b, memory_config=memory_config, realloc=n_pairs // group > 1,
                        )
                    x_chunk = ttnn.matmul(
                        a_chunk,
                        b_chunk,
                        compute_kernel_config=self.compute_kernel_config,
                        memory_config=memory_config,
                        program_config=program_config,
                        dtype=ttnn.bfloat16,
                    )
                    ttnn.deallocate(a_chunk)
                    ttnn.deallocate(b_chunk)
                    # Move the channel chunk from the batch axis back to the last axis:
                    # permute(0,2,3,1). On the large-L DRAM path, a single permute is a
                    # 3-way rotation of the last three axes (~6ms at L=1024); the
                    # equivalent transpose(1,2) then transpose(2,3) is ~2.6ms (the inner
                    # transpose is tile-local) and BIT-EXACT. On the small-L L1 path the
                    # single permute is marginally faster, so keep it there.
                    if large_seq and not _TRIMUL_RAW_CHANNEL_MOVES:
                        x_chunk_t = _channel_move_back(x_chunk, memory_config)
                        ttnn.deallocate(x_chunk)
                        x_chunk = x_chunk_t
                    else:
                        # The channel move is the last touch of the chunk before the concat, so
                        # on the L1 path it writes its result straight to DRAM: the separate
                        # clone that used to move it there was a whole extra round trip of the
                        # chunk (13.1 MB each way at 298 aa) for no arithmetic. Index-only, so
                        # bit-exact either way.
                        x_chunk = ttnn.permute(
                            x_chunk, (0, 2, 3, 1),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG if _TRIMUL_OUT_MOVE_DRAM
                            else memory_config,
                        )
                    if large_seq or _TRIMUL_OUT_MOVE_DRAM:
                        _acc_append(x_chunks, x_chunk, host_acc)
                    else:
                        # L1-resident chunk: move it to DRAM so all n_pairs can be held at
                        # once for the single concat.
                        moved = ttnn.clone(x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(x_chunk)
                        x_chunks.append(moved)
                break
            except RuntimeError as e:
                if large_seq or "clash with L1 buffers" not in str(e):
                    raise
                for _t in x_chunks:
                    ttnn.deallocate(_t)
                x_chunks = []
                # Drop the interrupted iteration's intermediates: whatever was live
                # at the throw still holds L1, and the retry must allocate against a
                # clean slate, not against the corpse of the failed attempt.
                gp_in_fused = g_in_a = g_in_b = p_in_a = p_in_b = None
                a_chunk = b_chunk = x_chunk = None
                _record_trimul_clash(H, self._hidden, batch, chunk_size)
                # tt-metal logs the clash at `critical` before raising, which reads like a
                # fatal error to anyone watching the fold. Say what actually happened.
                print(f"[tt-bio] trimul L1/circular-buffer clash at chunk width {chunk_size} "
                      f"(seq {H}, {COMPUTE_GRID_MAIN[0]}x{COMPUTE_GRID_MAIN[1]} grid): "
                      f"retrying narrower. The tt-metal 'critical' line above is expected and "
                      f"handled; the result is unchanged.", file=sys.stderr, flush=True)
                if chunk_size > TRIANGLE_MULT_CHUNK_SIZE:
                    chunk_size = _trimul_chunk_size(H, self._hidden, batch)
                    n_pairs = self._hidden // chunk_size
                    gp_in_chunks = self._gp_in_chunks(chunk_size, group)
                else:
                    # Minimum width still clashes: this shape's trimul does not fit in
                    # L1 on this grid at all. Take the residency threshold's other
                    # side -- the same ops with the pair tensors in DRAM.
                    _TRIMUL_DRAM_SHAPES.add(H)
                    memory_config = ttnn.DRAM_MEMORY_CONFIG
                    large_seq = True
                    host_acc = _host_concat(x_in)
                    group = _trimul_inproj_group(H, chunk_size, batch, n_pairs)
                    gp_in_chunks = self._gp_in_chunks(chunk_size, group)
        if x_norm_in is not None and H > SEQ_LEN_MORE_CHUNKING:
            # x_norm_in is dead on the row-blocked tail path (both norms are
            # recomputed per row block from x_in). Freeing it before the concat
            # drops that peak from 4 pair-tensor multiples to 3 -- the difference
            # between fitting and the 9i3p/9j4c refusal.
            ttnn.deallocate(x_norm_in)
        x = _acc_concat(x_chunks, -1, host_acc)
        dram_peak(f"trimul({'end' if self.ending else 'start'}) channel loop done [z={'x'.join(str(d) for d in x_in.shape)}]")
        # x_norm_in is None only when the byte gate row-blocked the input norm; that can
        # happen below SEQ_LEN_MORE_CHUNKING in a batched confidence head (the byte gate
        # sees the batch, this constant does not), and the full-size tail needs x_norm_in.
        # Take the row-blocked tail then too: it recomputes both norms from x_in and is
        # bit-identical, so the guard only ever fires where the else path would crash.
        if H > SEQ_LEN_MORE_CHUNKING or x_norm_in is None:
            # Row-block the output projections instead of computing them full-size.
            # Both layer_norms are row-local, so recomputing them per row block from the
            # (alive, unmutated) inputs is bit-identical to slicing the full-size results,
            # and the full-size norm_out output never exists: at these shapes it is one
            # pair-tensor-sized allocation attempted while z and the hidden are live, which
            # is exactly the refusal the large targets die on. Peak here is z + accumulated
            # blocks + concat destination, with the hidden freed before the concat. The
            # input norm above the gate is row-blocked the same way (_in_proj_rows), so no
            # full-size LN'd pair tensor exists anywhere on this path.
            blocks = []
            for s in range(0, H, PAIR_ROW_BLOCK):
                e = min(s + PAIR_ROW_BLOCK, H)
                z_rows = ttnn.layer_norm(
                    x_in[:, s:e],
                    weight=self.in_norm_weight,
                    bias=self.in_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
                g_block = ttnn.linear(
                    z_rows,
                    self.g_out_weight,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=_dtype(),
                    compute_kernel_config=self.compute_kernel_config,
                    core_grid=CORE_GRID_MAIN,
                )
                ttnn.deallocate(z_rows)
                x_rows = ttnn.layer_norm(
                    x[:, s:e],
                    weight=self.out_norm_weight,
                    bias=self.out_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
                p_block = ttnn.linear(
                    x_rows,
                    self.out_p_weight,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=_dtype(),
                    compute_kernel_config=self.compute_kernel_config,
                    core_grid=CORE_GRID_MAIN,
                )
                ttnn.deallocate(x_rows)
                _acc_append(blocks, ttnn.multiply_(
                    p_block, g_block, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
                ), host_acc)
                ttnn.deallocate(g_block)
            ttnn.deallocate(x)
            dram_peak(f"trimul({'end' if self.ending else 'start'}) tail blocks done [z={'x'.join(str(d) for d in x_in.shape)}]")
            return _acc_concat(blocks, 1, host_acc)
        x = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        if _TRIMUL_TAIL_F1:
            # `fused_tail` returns None for any call its descriptor does not cover (at 512 aa that
            # is the narrow-hidden trimuls, k_tiles=2), and the three ops below run unchanged.
            fused = _trimul_tail.fused_tail(
                x, x_norm_in, self.out_p_weight, self.g_out_weight,
                _mm_generic.ckc_args(self.compute_kernel_config), tuple(COMPUTE_GRID_MAIN))
            if fused is not None:
                ttnn.deallocate(x)
                ttnn.deallocate(x_norm_in)
                return fused
        p_out = _trimul_out_proj(x, self.out_p_weight, self.compute_kernel_config)
        ttnn.deallocate(x)
        dram_peak(f"trimul({'end' if self.ending else 'start'}) p_out done [z={'x'.join(str(d) for d in x_norm_in.shape)}]")
        g_out = _trimul_out_proj(x_norm_in, self.g_out_weight, self.compute_kernel_config)
        ttnn.deallocate(x_norm_in)
        x = ttnn.multiply_(
            p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        return x


# The qkv and g projections are the fold's two biggest `minimal_matmul` sites after the trimul
# in-projection. An off-fold sweep of every legal MinimalMatmulConfig at their exact shapes
# (perf/bigswing/mmcfg/mmcfg_sweep_512_qb2c0.json, warm 2, median of 5, torch.equal against the
# unconfigured default) found one winner each and nothing at all for the in-projection:
#
#     [512,512,256] x [256,768]  2.2021 -> 2.1021 ms  1.0476x  M=4 K=8 N=1 sub=4x1
#     [512,512,256] x [256,256]  0.9073 -> 0.8135 ms  1.1153x  M=2 K=8 N=1 sub=2x1
#     [1,512,512,256] x [256,128]  best legal config 0.9904x   -- no win, left alone
#
# Both are bit-exact. Gated because a per-call ratio is a screen, not a fold gain.
QKV_MM_CONFIG = True
_MM_CFG = os.environ.get("TT_BIO_QKV_MM_CONFIG", "1" if QKV_MM_CONFIG else "0") == "1"
# n_tiles -> (M_block, K_block, N_block, subblock_h, subblock_w).
# The nt=8 entry was (2,8,1,2,1), tuned at a small M. Swept over M on qb2 card 2 at K=256,
# N=256 (perf/triatt_opt/stage1_sweep.json), the 4-block entry wins at EVERY M measured and the
# margin grows with M, `torch.equal` throughout (same K_block, so the same accumulation order):
#     M      8192   16384   32768   65536  131072  262144  409600
#     ratio 1.044x  1.085x  1.051x  1.110x  1.209x  1.219x  1.204x
# The old entry cost 1.21x on every nt=8 pair-track projection in the repo at production M, not
# just the triangle-attention gate.
# Keyed on (kt, nt) -- the contraction tiles AND the output tiles -- not on nt alone. Keying on nt
# was protenix-only by accident: the two entries below are its widths, so K1, K1b and the swept
# block config have been shipped and idle for every other model since c9bfcaef. Adding nt=12 under
# the old key would ALSO have been picked up by opendde's gate projection, whose kt is 12 and which
# passes the `kt % blk[1]` guard because 12 % 4 == 0 -- and that config is MEASURED at 0.5318x and
# not bit-exact (perf/other512/s1_widths.json). The key is what prevents that.
#
# Every entry sets K_block == kt, so the contraction is one K block and the accumulation order is
# the unconfigured op's. MEASURED bit-exact by torch.equal at each width, with the op speedup:
#     (4,12) 1.2107x   (4,4) 1.6859x   (2,12) 1.2431x   (2,2) 2.1644x
# opendde's (12,36) and (12,12) are served by `_MM_DEFAULT`, not by a swept entry. A sweep over the
# DIVISORS of kt = 12 found nothing bit-exact (0.5794x / 0.5318x at M_block 1, the only M passing
# `mt % M` at its 995-token mt = 30939), which read as "no entry is possible at this width". The op's
# own default is K_block = 8, a NON-divisor: `determine_default_block_sizes` in tt-metal v0.68.0
# returns (M, K, N) = (8, 8, 8) unconditionally with subblocks (2, 2) under `fp32_dest_acc_en` (which
# every ckc on this path sets, line 4486), and `padded_K_tiles = round_up(K_tiles, K_block)` pads the
# last K block. An entry equal to that default is byte-identical to `config=None` BY CONSTRUCTION --
# MEASURED torch.equal at max_abs 0.0 on both opendde widths (perf/odde4x/screen1.py). It buys the
# matmul itself nothing (1.0010x / 1.0086x); it exists so K1's head-major transcription has a
# descriptor to read, which is worth 1.7733x on the qkv projection (perf/odde4x/screen2.json).
# If a site ever ran with fp32_dest_acc_en=False the default would be subblocks (4, 2) and this entry
# would stop being the default -- the per-arm CIF digest in perf/odde4x/ab_opendde_512.json is what
# checks that, and it is byte-identical.
_MM_DEFAULT = (8, 8, 8, 2, 2)

_MM_BLOCK = {
    (8, 24): (4, 8, 1, 4, 1),   # protenix-v2 qkv          -- unchanged, byte-identical to before
    (8, 8): (4, 8, 1, 4, 1),    # protenix-v2 gate + pair  -- unchanged
    (4, 12): (4, 4, 1, 4, 1),   # boltz2 / openfold3 qkv   at c_z=128
    (4, 4): (4, 4, 1, 4, 1),    # boltz2 / openfold3 gate  at c_z=128
    (2, 12): (4, 2, 1, 4, 1),   # openfold3 qkv            at c_z=64
    (2, 2): (4, 2, 1, 4, 1),    # openfold3 gate           at c_z=64
    # opendde tri-att at c_z=384. These two are NOT bit-exact -- K_block = 12 folds the contraction
    # differently from the unconfigured op, one bf16 ULP at max_abs 0.5. MEASURED at the fold, 512 aa
    # (perf/odde4x/ab_opendde_512_mm12.json): 96.578 -> 92.803 s, 1.0407x on a 0.063 s A/A floor,
    # and at 298 aa the structural cost is 0.3249 A Kabsch CA against an A/A floor of exactly zero
    # (perf/odde4x/ab_opendde_298_mm12.json + perf/other512/cif_rmsd.py). Ask 4649 fixed the merge
    # threshold at <= 0.35 A CA BEFORE the number existed, and protenix-v2 is byte-identical at the
    # full 64-hex digest with these entries live (perf/odde4x/ab_px_leak.json), so nothing else moves.
    # The byte-identical alternative at the same two keys is `_MM_DEFAULT`, worth 96.785 -> 94.523 s
    # instead; swap these two values for it and the tail guard in triatt_qkv.py turns itself on.
    (12, 36): (4, 12, 1, 2, 1),
    (12, 12): (8, 12, 1, 2, 1),
}


def _mm_block_for(w):
    """The swept block entry for this weight, or None. The single reader of the (kt, nt) key."""
    return _MM_BLOCK.get(((int(w.shape[-2]) + 31) // 32, (int(w.shape[-1]) + 31) // 32))


@lru_cache(maxsize=None)
def _mm_core_coord(gx, gy):
    return ttnn.CoreCoord(gx, gy)


def _qkv_mm_config(inp, w):
    """The swept block config for this (activation, weight) pair, or None to leave the op alone."""
    if not _MM_CFG:
        return None
    kt = (int(w.shape[-2]) + 31) // 32
    nt = (int(w.shape[-1]) + 31) // 32
    blk = _mm_block_for(w)
    if blk is None:
        return None
    shape = [int(d) for d in inp.shape]      # ttnn.Shape does not support slicing
    mt = 1
    for d in shape[:-1]:
        mt *= d
    mt = (mt + 31) // 32
    M, K, N, sh, sw = blk
    # The three divisibility guards protect bit-exactness against a config that folds the
    # contraction in a different order than the unconfigured op. `_MM_DEFAULT` cannot: it IS that
    # order. Scope the relaxation by object identity on the one shared tuple, so the six swept
    # entries keep all three guards and every other model stays byte-for-byte on today's path.
    if blk is not _MM_DEFAULT and (kt % K or mt % M or nt % N):
        return None
    return ttnn.MinimalMatmulConfig(
        M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
        compute_with_storage_grid_size=_mm_core_coord(*COMPUTE_GRID_MAIN))


class TriangleAttention(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
        scale_pair_bias: bool = True,
        fp32_softmax: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.affinity = affinity
        self.fp32_softmax = fp32_softmax
        self.scale = self.head_dim**0.5
        # Boltz/Protenix fold sqrt(head_dim) into the pair-bias projection (their
        # reference adds the bias pre-scaled); openfold3 pre-scales q by 1/sqrt(d) and
        # adds the pair bias UNSCALED (triangular_attention.py + primitives/attention.py
        # _prep_qkv). The folded form makes the OF3 tri_att softmax sqrt(d)~5.7x too
        # peaky -- the root cause behind the OF3 MSA pair_stack z-track degradation
        # (fp32 A/B vs the reference golden: 0.892 folded vs 1.000000 unscaled).
        self._bias_scale = self.scale if scale_pair_bias else 1.0
        # nlp_concat_heads pads each head's channel dim up to a 32-tile boundary, so at
        # a sub-tile head_dim it
        # yields n_heads*32 channels while the gate g carries n_heads*head_dim -- a
        # shape mismatch that throws "Invalid subtile broadcast type" in gate_and_
        # project's multiply_. The tile-aligned head_dim=32 path (MSA / Boltz-2 /
        # Protenix) is unaffected and stays on the original nlp_concat_heads path. The
        # sub-tile path pads the qkv head_dim up to 32 (zeros, so the real head_dim
        # channels are unchanged) and slices + manual-concats the SDPA output back to
        # n_heads*head_dim -- mirrors the AttentionPairBias sub-tile handling.
        head_dim_padding = -head_dim % 32
        self.padded_head_dim = head_dim + head_dim_padding
        self.subtile = head_dim_padding != 0
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.bias_weight = ttnn.multiply_(self.torch_to_tt("linear.weight"), self._bias_scale)
        qkv_weight = torch.cat(
            [
                self.weights["linear_q.weight"],
                self.weights["linear_k.weight"],
                self.weights["linear_v.weight"],
            ],
            dim=0,
        )
        if self.subtile:
            qkv_weight = qkv_weight.reshape(3 * self.n_heads, head_dim, -1)
            qkv_weight = torch.nn.functional.pad(
                qkv_weight, (0, 0, 0, head_dim_padding), mode="constant", value=0
            )
            qkv_weight = qkv_weight.reshape(3 * self.n_heads * self.padded_head_dim, -1)
        self.qkv_weight = ttnn.from_torch(
            qkv_weight.t(),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=_dtype(),
        )
        self.g_weight = self.torch_to_tt("linear_g.weight", dtype=_dtype())

    def __call__(self, x: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        S = x.shape[0]
        need_chunk = S > SEQ_LEN_MORE_CHUNKING and (self.affinity or not _FAST_MODE or _IS_SMALL_GRID)
        if need_chunk:
            # Large-sequence path: never materialise the full layer_norm output.
            # layer_norm is row-local, so norming a row block is bit-identical to
            # slicing the full normed tensor, and for the ending variant a row block of
            # the transposed pair tensor is a column strip of the input followed by the
            # same (1,0,2) transpose -- pure reordering, bit-exact. The triangle bias is
            # built from the same per-block norms and concat'ed along its row axis, so
            # it arrives identical to the full-tensor computation. This drops the
            # chunked path from 3 full pair tensors live (z + normed x + accumulated
            # parts) to 2, plus the n_heads-wide bias.
            chunk = TRIANGLE_ATT_CHUNK_SIZE_FAST if _FAST_MODE else TRIANGLE_ATT_CHUNK_SIZE
            # Byte-cap the row chunk so the fused qkv projection (rows x pad32(S) x 3c
            # bf16) stays a size a fragmented 12 GiB WH part can still supply. The
            # allocator needs size/12 contiguous in every bank, and after the trunk's
            # MSA-chunk churn a >~2 GiB request fails even with GiBs nominally free
            # (measured: od_9i3p refused 512x1920x1152x2 = 2.16 GiB at 4.6 GiB used).
            # 1.5 GiB asks 128 MiB per bank. All row-local ops: the cap changes chunk
            # boundaries only, not what any row computes.
            _qkv_cap = (1536 * 2 ** 20) // (-(-S // 32) * 32 * x.shape[2] * 3 * 2)
            chunk = min(chunk, max(32, _qkv_cap // 32 * 32))

            def normed_rows(s, e):
                blk = x[:, s:e, :] if self.ending else x[s:e, :, :]
                if self.ending:
                    blk = _pair_transpose(blk, _transpose_memory_config(blk))
                return ttnn.layer_norm(
                    blk,
                    weight=self.layer_norm_weight,
                    bias=self.layer_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            bias_parts = []
            for s in range(0, S, chunk):
                e = min(s + chunk, S)
                xc = normed_rows(s, e)
                b = _pair_proj_linear(
                    xc, self.bias_weight, self.compute_kernel_config, ttnn.bfloat16
                )
                ttnn.deallocate(xc)
                # No explicit deallocate of b/bp: unsqueeze shares b's buffer, and an
                # explicit ttnn.deallocate force-frees the buffer even while the view is
                # still referenced, so the permute would read recycled memory (measured:
                # bias garbage at PCC 0.87 with the deallocates, bit-exact without).
                # Rebinding on the next iteration frees the buffer via refcount.
                bp = ttnn.unsqueeze(b, 0)
                bias_parts.append(ttnn.permute(bp, (0, 3, 1, 2)))
            triangle_bias = ttnn.concat(bias_parts, dim=2)
            for bp in bias_parts:
                ttnn.deallocate(bp)
            dram_peak(f"tri_att({'end' if self.ending else 'start'}) bias built [z={'x'.join(str(d) for d in x.shape)}]")
        else:
            if self.ending:
                x = _pair_transpose(x, _transpose_memory_config(x))
            # Explicit DRAM: for the ending variant x is the L1 transpose result
            # (_transpose_memory_config) and ttnn would otherwise inherit L1 here and again
            # for the qkv projection, whose 157 MB does not fit.
            x = ttnn.layer_norm(
                x,
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            triangle_bias = _pair_proj_linear(
                x, self.bias_weight, self.compute_kernel_config, ttnn.bfloat16
            )
            triangle_bias = ttnn.unsqueeze(triangle_bias, 0)
            triangle_bias = ttnn.permute(triangle_bias, (0, 3, 1, 2))
            dram_peak(f"tri_att({'end' if self.ending else 'start'}) bias built [z={'x'.join(str(d) for d in x.shape)}]")

        def attend(qkv_in, bias, keep_heads=False):
            if isinstance(qkv_in, tuple):
                # already head-major, written there by the projection itself
                q, k, v = qkv_in
                return _attend_heads(q, k, v, bias, keep_heads)
            qkv_in = ttnn.unsqueeze(qkv_in, 1)
            # The head split follows the projection: when the projection kept its result in L1
            # (see _tri_att_qkv_l1_config) reading it back out to DRAM here would hand back the
            # whole win, and the chunked path's qkv is in DRAM so this is a no-op for it.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_in, num_heads=self.n_heads, num_kv_heads=self.n_heads,
                transpose_k_heads=False, memory_config=qkv_in.memory_config(),
            )
            ttnn.deallocate(qkv_in)
            return _attend_heads(q, k, v, bias, keep_heads)

        def _attend_heads(q, k, v, bias, keep_heads=False):
            if _FP32_SOFTMAX or self.fp32_softmax:
                o = _fp32_softmax_attention(
                    q, k, v, bias,
                    scale_inv=self.scale ** -1,
                    compute_kernel_config=self.compute_kernel_config,
                    out_dtype=_dtype(),
                    bias_scale_inv=1.0 / self._bias_scale,
                )
            else:
                o = _tri_att_sdpa(q, k, v, bias, self.scale**-1)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if keep_heads:
                # The tail reads [batch, head, seq, 32] directly; see tt_bio/triatt_qkv.py
                return o
            if self.subtile:
                # Slice off the zero-padded head channels, then manual head-concat
                # (head-major) to produce [1, S, n_heads*head_dim] -- nlp_concat_heads
                # would re-pad to n_heads*32 here. Same concat order as the tile-aligned
                # path, so numerically identical for the real head_dim channels.
                o = o[:, :, :, : self.head_dim]
                o = ttnn.permute(o, (0, 1, 3, 2))
                o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
                o = ttnn.permute(o, (0, 2, 1))
            else:
                o_heads = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(o)
                o = ttnn.squeeze(o_heads, 1)
            return o

        def gate_and_project(o_in: ttnn.Tensor, g_in: ttnn.Tensor) -> ttnn.Tensor:
            head_major = len(g_in.shape) == 4
            o_in = ttnn.multiply_(o_in, g_in, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_in)
            if head_major:
                x_out = _triatt_qkv.out_proj(
                    o_in, self.o_weight, self.compute_kernel_config, _dtype())
            else:
                x_out = _pair_proj_linear(
                    o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True
                )
            ttnn.deallocate(o_in)
            return x_out

        if need_chunk:
            if not self.affinity and attn_mask is not None:
                triangle_bias = ttnn.add(triangle_bias, attn_mask)
            # Assemble the row blocks on the host when the full result is large enough
            # that the concat's full-size allocation would risk a fragmented-DRAM
            # refusal (CONCAT_HOST_BYTES); the loop then holds one block on device.
            host_acc = _host_concat(x)
            parts = []
            for s in range(0, S, chunk):
                end = min(s + chunk, S)
                x_chunk = normed_rows(s, end)
                qkv_cfg_chunk = _qkv_mm_config(x_chunk, self.qkv_weight)
                qkv_chunk = _triatt_qkv.qkv_heads(
                    x_chunk, self.qkv_weight, self.compute_kernel_config,
                    self.n_heads, self.head_dim, _dtype(), qkv_cfg_chunk,
                )
                if qkv_chunk is None:
                    qkv_chunk = ttnn.experimental.minimal_matmul(
                        input_tensor=x_chunk,
                        weight_tensor=self.qkv_weight,
                        compute_kernel_config=self.compute_kernel_config,
                        dtype=_dtype(),
                        config=qkv_cfg_chunk,
                    )
                g_cfg_chunk = _qkv_mm_config(x_chunk, self.g_weight)
                g_chunk = None
                if isinstance(qkv_chunk, tuple):
                    g_chunk = _triatt_qkv.gate_proj(
                        x_chunk, self.g_weight, self.o_weight, self.compute_kernel_config,
                        self.n_heads, self.head_dim, _dtype(), g_cfg_chunk,
                    )
                if g_chunk is None:
                    g_chunk = ttnn.experimental.minimal_matmul(
                        input_tensor=x_chunk,
                        weight_tensor=self.g_weight,
                        compute_kernel_config=self.compute_kernel_config,
                        dtype=_dtype(),
                        config=g_cfg_chunk,
                    )
                ttnn.deallocate(x_chunk)
                if self.affinity:
                    bias = ttnn.add(triangle_bias, attn_mask[s:end, :, :])
                    o_chunk = attend(qkv_chunk, bias, len(g_chunk.shape) == 4)
                    ttnn.deallocate(bias)
                else:
                    o_chunk = attend(qkv_chunk, triangle_bias, len(g_chunk.shape) == 4)
                if not isinstance(qkv_chunk, tuple):   # the triple is freed inside attend
                    ttnn.deallocate(qkv_chunk)
                # The projection keeps its L1 output -- same program config, same numerics --
                # and the RESULT is moved off L1 before the next chunk runs. Nothing reads it
                # until the concat after the loop, so holding it in L1 buys nothing and costs
                # the next chunk its circular buffers. Measured on the 8x9 Galaxy at
                # 640/768/1024 aa: the first chunk's output ([512, S, 64] bf16 = 285/342/456
                # tiles per core = 583,680/700,416/933,888 B) sits exactly on top of the free
                # L1, and the tail chunk's qkv projection needs a 1,152,288 B static CB region
                # that no longer fits under it -- "L1 buffer allocated at 915456 and static
                # circular buffer region ends at 1152288". Every size 640-1024 aa threw there;
                # none reached the trunk. The same tensor is 323,584 B per core on 130
                # Blackhole cores and clears the CB region with 23,264 B to spare, which is why
                # it never showed up there -- and Blackhole does not take this branch below
                # 1536 aa in any case.
                #
                # Producing the projection straight into DRAM instead (l1_out=False) also fixes
                # the crash and is one write cheaper, but it takes a different program config
                # and so folds the contraction differently: measured at 640 aa --fast on one
                # pinned tree, plDDT 0.796168 -> 0.791786. Above 608 tokens --fast ALREADY
                # WORKED, so that is a real accuracy change on a live path, and it was 2.3 %
                # slower besides (252.044 -> 257.719 s). A copy is bit-exact, so it is the one
                # that ships. `host_acc` already frees the device buffer on its own path.
                out_chunk = gate_and_project(o_chunk, g_chunk)
                # Spill ONLY when holding this block would make the next chunk throw. Nothing
                # here is numerically inert -- `_pair_proj_linear` accepts or refuses its L1
                # output according to what the allocator has left, and refusing takes a
                # different program config, so relieving L1 pressure changes which config LATER
                # chunks pick. Measured at 640 aa --fast, one pinned tree, one card, against an
                # A/A floor of 2.5e-5: plDDT 0.796168/0.796143 unmodified, 0.791786 producing
                # into DRAM, 0.791229 copying unconditionally. So the relief has to fire exactly
                # where the fold would otherwise produce no output at all, and nowhere else.
                #
                # `_FAST_MODE` is NOT that condition and was tried: it is True only around the
                # trunk (protenix.py wraps `self.trunk(...)` in set_fast_mode) and deliberately
                # False for the confidence and diffusion stages, so a --fast fold still runs
                # this loop with the flag down and still moved (0.792931 on the same card).
                #
                # The condition is the residency itself, in bytes per core, bracketed by
                # measurement on the 8x9 Galaxy: --fast holds 290,496 B at 640 aa and 464,576 B
                # at 1024 aa and folds; the default arm holds 583,680 B at 640 aa, 700,416 at
                # 768 and 933,888 at 1024 and throws at all three, because the next chunk's qkv
                # projection needs a 1,152,288 B static CB region that no longer fits under it.
                # The threshold sits between the largest measured fit and the smallest measured
                # throw, referenced to this part's own L1 so a roomier part keeps more.
                #
                # Only when it really came back in L1. `_pair_proj_linear` declines the L1
                # output whenever the allocator refuses and returns a DRAM tensor instead, and
                # for a DRAM source `to_memory_config(..., DRAM)` hands back the same buffer --
                # so an unconditional deallocate here frees the tensor just appended to `parts`
                # and the concat after the loop dies with "Buffer is not allocated". Same trap
                # the bias loop above documents for `unsqueeze`.
                if (not host_acc
                        and out_chunk.memory_config().buffer_type == ttnn.BufferType.L1
                        and _chunk_l1_per_core(out_chunk) >= TRIATT_CHUNK_L1_SPILL_BYTES):
                    out_dram = ttnn.to_memory_config(out_chunk, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(out_chunk)
                    out_chunk = out_dram
                _acc_append(parts, out_chunk, host_acc)
            dram_peak(f"tri_att({'end' if self.ending else 'start'}) row loop done [z={'x'.join(str(d) for d in x.shape)}]")
            # x here is the reshaped (unpermuted) input -- for the starting variant it can
            # alias the caller's pair tensor, so it must NOT be deallocated.
            ttnn.deallocate(triangle_bias)
            if host_acc:
                h = torch.cat(parts, dim=0)
                # The ending variant's back-transpose rides the host assembly (pure
                # data movement, bit-identical) so no second full-size device tensor
                # is allocated for it here.
                if self.ending:
                    h = h.permute(1, 0, 2)
                x = ttnn.from_torch(h.contiguous(), layout=ttnn.TILE_LAYOUT,
                                    device=get_device(), dtype=ttnn.bfloat16)
                return ttnn.reshape(x, (1, *x.shape))
            x = ttnn.concat(parts, dim=0)
            del parts
        else:
            qkv_cfg = _qkv_l1_config(x, self.qkv_weight, _dtype())
            # When the head-major projection takes the call, `qkv` is already the (q, k, v)
            # triple and no head split follows. It declines an L1 projection outright.
            qkv = None if qkv_cfg is not None else _triatt_qkv.qkv_heads(
                x, self.qkv_weight, self.compute_kernel_config,
                self.n_heads, self.head_dim, _dtype(), _qkv_mm_config(x, self.qkv_weight),
            )
            if qkv is None:
                if qkv_cfg is not None:
                    qkv = ttnn.linear(
                        x,
                        self.qkv_weight,
                        compute_kernel_config=self.compute_kernel_config,
                        dtype=_dtype(),
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        program_config=qkv_cfg,
                    )
                else:
                    qkv = ttnn.experimental.minimal_matmul(
                        input_tensor=x,
                        weight_tensor=self.qkv_weight,
                        compute_kernel_config=self.compute_kernel_config,
                        dtype=_dtype(),
                        config=_qkv_mm_config(x, self.qkv_weight),
                    )
            g = None
            if isinstance(qkv, tuple):
                g = _triatt_qkv.gate_proj(
                    x, self.g_weight, self.o_weight, self.compute_kernel_config,
                    self.n_heads, self.head_dim, _dtype(), _qkv_mm_config(x, self.g_weight),
                )
            if g is None:
                g = ttnn.experimental.minimal_matmul(
                    input_tensor=x,
                    weight_tensor=self.g_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=_dtype(),
                    config=_qkv_mm_config(x, self.g_weight),
                )
            ttnn.deallocate(x)
            if attn_mask is not None:
                triangle_bias = ttnn.add(triangle_bias, attn_mask)
            o = attend(qkv, triangle_bias, len(g.shape) == 4)
            if not isinstance(qkv, tuple):        # the triple is freed inside attend
                ttnn.deallocate(qkv)
            ttnn.deallocate(triangle_bias)
            x = gate_and_project(o, g)
        if self.ending:
            x = _pair_transpose(x, _transpose_memory_config(x))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        compute_pair_bias: bool,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        dtype: ttnn.DataType | None = None,
        fp32_raw_matmul_attention: bool = False,
        scale_pair_bias: bool = True,
        fp32_softmax: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.fp32_softmax = fp32_softmax
        self.head_dim = head_dim
        self.dtype = dtype if dtype is not None else _dtype(ttnn.bfloat16)
        self.fp32_raw_matmul_attention = fp32_raw_matmul_attention
        self.n_heads = n_heads
        self.compute_pair_bias = compute_pair_bias
        self.atom_level = atom_level
        # Set by DiffusionTransformerLayer for the token-level diffusion DiT only. S6 reads it,
        # so no trunk / MSA / template / confidence site can be rerouted by that flag.
        self.token_dit = False
        if atom_level:
            self.q_weight = self.torch_to_tt("proj_q.weight", dtype=self.dtype)
            self.q_bias = self.torch_to_tt("proj_q.bias", dtype=self.dtype)
            kv_weight = torch.cat([self.weights["proj_k.weight"], self.weights["proj_v.weight"]], dim=0)
            self.kv_weight = ttnn.from_torch(
                kv_weight.t(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            qkv_weight = torch.cat(
                [self.weights["proj_q.weight"], self.weights["proj_k.weight"], self.weights["proj_v.weight"]],
                dim=0,
            )
            head_dim_padding = -head_dim % 32
            padded_head_dim = head_dim + head_dim_padding
            qkv_weight = qkv_weight.reshape(3 * self.n_heads, head_dim, -1)
            qkv_weight = torch.nn.functional.pad(qkv_weight, (0, 0, 0, head_dim_padding), mode='constant', value=0)
            qkv_weight = qkv_weight.reshape(3 * self.n_heads * padded_head_dim, -1)
            self.qkv_weight = ttnn.from_torch(
                qkv_weight.t(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=self.dtype,
            )
            q_bias = self.weights["proj_q.bias"]
            q_bias = q_bias.reshape(self.n_heads, head_dim)
            q_bias = torch.nn.functional.pad(q_bias, (0, head_dim_padding), mode='constant', value=0)
            q_bias = q_bias.reshape(self.n_heads * padded_head_dim)
            qkv_bias = torch.cat([q_bias, torch.zeros(2 * self.n_heads * padded_head_dim, dtype=q_bias.dtype, device=q_bias.device)])
            self.qkv_bias = ttnn.from_torch(
                qkv_bias,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=self.dtype,
            )
        self.g_weight = self.torch_to_tt("proj_g.weight", dtype=self.dtype)
        if compute_pair_bias:
            self.z_norm_weight = self.torch_to_tt("proj_z.0.weight", dtype=self.dtype)
            self.z_norm_bias = self.torch_to_tt("proj_z.0.bias", dtype=self.dtype)
            # Boltz/Protenix scale the pair bias by sqrt(head_dim); openfold3 adds it
            # unscaled (q is pre-scaled by 1/sqrt(d) in the reference Attention). OF3
            # construction sites pass scale_pair_bias=False.
            self._bias_scale = self.head_dim**0.5 if scale_pair_bias else 1.0
            self.z_weight = ttnn.multiply_(
                self.torch_to_tt("proj_z.1.weight", dtype=self.dtype),
                self._bias_scale,
            )
        self.o_weight = self.torch_to_tt("proj_o.weight", dtype=self.dtype)

    def compute_bias(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """Project the (LN'd) pair tensor z -> per-head additive attention bias
        (1, n_heads, S, S). This is a pure function of z (no per-query dependence), so
        for a fixed z (e.g. the diffusion trunk pair_z, constant across all sampling
        steps) it can be computed ONCE and replayed via __call__(bias_precomputed=True),
        instead of recomputing this NxNxc_z layer_norm+linear every call. Uses the same
        (head_dim**0.5-scaled) z_weight as the inline path, so the result is identical."""
        z = ttnn.layer_norm(
            z, weight=self.z_norm_weight, bias=self.z_norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.linear(
            z, self.z_weight, compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN,
        )
        return ttnn.permute(z, (0, 3, 1, 2))

    def _attention(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        bias: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if (_FP32_SOFTMAX or self.fp32_softmax) and self.dtype != ttnn.float32:
            # Gate on: fp32 softmax reduction, bf16 operands/storage (reference recipe).
            #
            # Do not reroute this to the fused SDPA to skip the re-materialisation traffic. It is
            # worth 1.37x on the openfold3 512 aa fold (107.489 -> 78.205 s) and it changes the
            # answer: all-atom Kabsch RMSD 27.347 A against this path on a 0.000 A A/A floor, and
            # plDDT 0.547851 -> 0.439598. Flipping only the MSA and template stacks is worth 1.05x
            # and still moves the structure 7.611 A. The compute kernel config cannot rescue it:
            # sdpa_generic keeps the exponentiated scores in a bf16 circular buffer, so
            # fp32_dest_acc never reaches them. Measured: perf/other512/ab_of3_sites_512.json.
            return _fp32_softmax_attention(
                q, k, v, bias,
                scale_inv=self.head_dim ** -0.5,
                compute_kernel_config=self.compute_kernel_config,
                out_dtype=_dtype(),
                bias_scale_inv=1.0 / self._bias_scale,
            )
        if self.dtype != ttnn.float32:
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
            )

        # SDPA accepts bf16/bf8/bf4 only. Keep the fp32 transformer in fp32
        # storage and cross this one hardware boundary in native bf16: the
        # attention reduction is the precision-insensitive part (softmax over
        # a large logits range), and fp32 SDPA is op-blocked on Wormhole, so a
        # pure-fp32 matmul-attention wedges the device. The linears, residuals,
        # and layernorm around it stay fp32.
        q_bf16 = ttnn.typecast(q, ttnn.bfloat16, memory_config=q.memory_config())
        k_bf16 = ttnn.typecast(k, ttnn.bfloat16, memory_config=k.memory_config())
        v_bf16 = ttnn.typecast(v, ttnn.bfloat16, memory_config=v.memory_config())
        bias_bf16 = ttnn.typecast(
            bias, ttnn.bfloat16, memory_config=bias.memory_config()
        )
        out_bf16 = ttnn.transformer.scaled_dot_product_attention(
            q_bf16,
            k_bf16,
            v_bf16,
            attn_mask=bias_bf16,
            is_causal=False,
            scale=self.head_dim**-0.5,
            program_config=_sdpa_program_config_for_lengths(
                q_bf16.shape[2], k_bf16.shape[2]
            ),
        )
        for tensor in (q_bf16, k_bf16, v_bf16, bias_bf16):
            ttnn.deallocate(tensor)
        out = ttnn.typecast(
            out_bf16, self.dtype, memory_config=out_bf16.memory_config()
        )
        ttnn.deallocate(out_bf16)
        return out

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
        bias_precomputed: bool = False,
    ) -> ttnn.Tensor:
        if not self.atom_level:
            qkv = ttnn.linear(
                s,
                self.qkv_weight,
                bias=self.qkv_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            qkv = ttnn.unsqueeze(qkv, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_heads,
                transpose_k_heads=False,
            )
            ttnn.deallocate(qkv)
            # bias_precomputed: z is ALREADY the (1,n_heads,S,S) bias from compute_bias() -> skip recompute
            if self.compute_pair_bias and not bias_precomputed:
                # The z->bias projection below reads this whole tensor (48.82 MB at 298 aa) to
                # write one tile of width, so it is bound by its SOURCE, not by its own write.
                # Handing it an L1-resident normed z removes the norm's DRAM write and the
                # projection's DRAM read at once: 450.3 -> 137.0 us on the projection.
                z, in_l1 = (_l1_layer_norm(z, 1.5, weight=self.z_norm_weight,
                                           bias=self.z_norm_bias, epsilon=1e-5,
                                           compute_kernel_config=self.compute_kernel_config)
                            if _PAIR_BIAS_L1_NORM else
                            (ttnn.layer_norm(z, weight=self.z_norm_weight, bias=self.z_norm_bias,
                                             epsilon=1e-5,
                                             compute_kernel_config=self.compute_kernel_config),
                             False))
                zb = _narrow_proj_linear(z, self.z_weight, self.compute_kernel_config, z.dtype,
                                         l1_out=in_l1)
                if zb is None:
                    zb = ttnn.linear(
                        z,
                        self.z_weight,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN,
                    )
                # The normed pair tensor is dead as soon as the projection has read it, and at
                # 1.5x headroom it holds most of every L1 bank. Freeing it HERE rather than at the
                # rebind below is what matters: the bias permute then allocates in the space it
                # vacates instead of underneath it, so the q@k^T matmul four lines down can still
                # place its circular buffers. Without this the whole [385, 506] token band throws
                # `Statically allocated circular buffers ... clash with L1 buffers`.
                ttnn.deallocate(z)
                z = ttnn.permute(zb, (0, 3, 1, 2))
                ttnn.deallocate(zb)
            if self.dtype == ttnn.float32 and self.fp32_raw_matmul_attention:
                # ttnn SDPA rejects fp32 inputs (bf16/bf8 only), so the Protenix fp32 DiT
                # path computes attention as raw matmul. SDPA scales its additive mask
                # along with QK, so z_weight carries sqrt(head_dim) compensation. Undo
                # that compensation before adding z after the explicit QK scale.
                if self.compute_pair_bias:
                    z = ttnn.multiply(z, self.head_dim ** -0.5)
                if seq_mask is not None:
                    z = ttnn.add_(z, seq_mask)
                kt = ttnn.permute(k, (0, 1, 3, 2))
                sc = batched_matmul(q, kt,
                                    compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(kt)
                sc = ttnn.multiply(sc, self.head_dim ** -0.5)
                sc = ttnn.add(sc, z)
                attn = ttnn.softmax(sc, dim=-1)
                o = batched_matmul(attn, v,
                                   compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(attn)
                ttnn.deallocate(sc)
            elif (self.token_dit and _B2_TOKEN_DIT_SDPA and z is not None
                  and seq_mask is None):
                # S6. The bias is the token DiT's own rollout-invariant `bias_token`, already
                # DRAM-resident, so nothing is spilled to reach the fused kernel. SDPA scales its
                # additive mask along with QK, so scale=head_dim**-0.5 reproduces the unfused
                # chain's (q@k^T + z) * head_dim**-0.5 exactly in exact arithmetic; what differs
                # is the bf16 exponentiated-score buffer, hence the accuracy gate on this arm.
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=z,
                    is_causal=False,
                    scale=self.head_dim**-0.5,
                    program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
                )
            else:
                if seq_mask is not None:
                    z = ttnn.add_(z, seq_mask)
                # Unfused attention: the fused ttnn SDPA kernel systematically flattens
                # near-degenerate attention distributions (observed on 7XI5s repeat-
                # protein logits: output std shrunk ~16%, s-track diverged over trunk
                # cycles). matmul + ttnn.softmax + matmul on the same q/k/v/bias is
                # PCC-clean (0.99993 vs 0.98128). SDPA scales its additive mask along
                # with QK, so (q@k^T + z) * head_dim**-0.5 reproduces it exactly.
                kt = ttnn.transpose(k, -2, -1)
                logits = batched_matmul(q, kt, compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(kt)
                if z is not None:
                    logits = ttnn.add_(logits, z)
                logits = ttnn.multiply_(logits, self.head_dim**-0.5)
                probs = ttnn.softmax(logits, dim=-1,
                                     compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(logits)
                o = batched_matmul(probs, v, compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(probs)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            o = o[:, :, :, :self.head_dim]
            o = ttnn.permute(o, (0, 1, 3, 2))
            o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
            o = ttnn.permute(o, (0, 2, 1))
        else:
            s = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG, dtype=_dtype())
            B, K, W, D_S = s.shape
            s_kv = ttnn.reshape(s, (B, 2 * K, W // 2, -1))
            s_kv = ttnn.permute(s_kv, (0, 2, 3, 1))
            s_kv = ttnn.matmul(
                s_kv,
                keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            s_kv = ttnn.permute(s_kv, (0, 3, 1, 2))
            s_kv = ttnn.reshape(s_kv, (B, K, -1, D_S))

            q = ttnn.linear(
                s,
                self.q_weight,
                bias=self.q_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                dtype=_dtype(),
            )
            kv = ttnn.linear(
                s_kv,
                self.kv_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                dtype=_dtype(),
            )

            if _ATOM_PAD_IN_TILE:
                q = ttnn.pad(q, [[0, 0], [0, 0], [0, ATOM_DIM - ATOM_WINDOW], [0, 0]], 0.0)
            else:
                q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
                q = ttnn.pad(q, [[0, 0], [0, 0], [0, ATOM_DIM - ATOM_WINDOW], [0, 0]], 0.0)
                q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, dtype=_dtype())
            q = ttnn.reshape(q, (B * K, 1, ATOM_DIM, -1))
            kv = ttnn.reshape(kv, (B * K, 1, ATOM_DIM, -1))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(q, kv, num_heads=self.n_heads, num_kv_heads=self.n_heads, transpose_k_heads=False)
            _, H, S, D_Q = q.shape
            q = ttnn.reshape(q, (B, K * H, S, D_Q))
            k = ttnn.reshape(k, (B, K * H, S, D_Q))
            v = ttnn.reshape(v, (B, K * H, S, D_Q))
            q = q[:, :, :ATOM_WINDOW, :]
            z = ttnn.reshape(z, (1, -1, z.shape[2], z.shape[3]))
            o = self._attention(q, k, v, z)
            o = ttnn.reshape(o, (B * K, H, W, D_Q))
            o = ttnn.experimental.nlp_concat_heads(o)
            o = ttnn.squeeze(o, 1)
            o = ttnn.reshape(o, (B, K, W, D_S))
        g = ttnn.linear(
            s,
            self.g_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        if _FAST_MODE:
            o = ttnn.typecast(o, ttnn.bfloat16)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], dtype=self.dtype)
        ttnn.deallocate(g)
        x = ttnn.linear(
            o, self.o_weight, compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(o)
        return x


class Transition(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        dtype: ttnn.DataType | None = None,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.dtype = dtype
        weight_dtype = dtype if dtype is not None else ttnn.bfloat16
        self.norm_weight = self.torch_to_tt("norm.weight", dtype=weight_dtype)
        self.norm_bias = self.torch_to_tt("norm.bias", dtype=weight_dtype)
        self.fc1_weight = self.torch_to_tt("fc1.weight", dtype=weight_dtype)
        self.fc2_weight = self.torch_to_tt("fc2.weight", dtype=weight_dtype)
        self.fc3_weight = self.torch_to_tt("fc3.weight", dtype=weight_dtype)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def swiglu(x):
            dtype = self.dtype if self.dtype is not None else _dtype()
            x_norm = ttnn.layer_norm(
                x,
                weight=self.norm_weight,
                bias=self.norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation=None if _UNFUSED_SILU else "silu",
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=dtype,
                core_grid=CORE_GRID_MAIN,
            )
            if _UNFUSED_SILU:
                x_1 = ttnn.silu(x_1, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=x_1)
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=dtype,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(x_norm)
            x = ttnn.multiply_(x_1, x_2)
            ttnn.deallocate(x_2)
            x_dram = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=dtype,
                core_grid=CORE_GRID_MAIN,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            return x_dram
        if len(x.shape) < 4:
            batch_chunking_threshold = (
                SEQ_LEN_MORE_CHUNKING
                if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
                else TRANSITION_BATCH_CHUNKING_THRESHOLD
            )
            if x.shape[1] > batch_chunking_threshold:
                return ttnn.concat([swiglu(x[b:b+1, :, :]) for b in range(x.shape[0])], dim=0)
            return swiglu(x)

        H, W = x.shape[1], x.shape[2]
        transition_h_chunk_size = TRANSITION_H_CHUNK_SIZE_FAST if _FAST_MODE else TRANSITION_H_CHUNK_SIZE
        if not _FAST_MODE and W <= TRANSITION_H_CHUNK_BIG_MAX_W and x.shape[-1] <= 256:
            transition_h_chunk_size = TRANSITION_H_CHUNK_SIZE_BIG
        # Per-chunk swiglu L1 use ~ h_chunk * W * channel. The chunk size is tuned for the
        # reference W=1024, c=128; scale the row-chunk so memory stays bounded for wider pair
        # channels / longer sequences -- but only shrink when actually needed (so c=128 and
        # Protenix-v2's c=256 at W<=512 keep the full chunk; only large W reduces it). No
        # Boltz-2 regression (c=128, W<=1024 -> factor 1.0). On a small grid (Wormhole, ~55%
        # of Blackhole's L1) the c=128 per-chunk CB still fits (Boltz-2/esmfold2 fold to 1024),
        # but a WIDER channel overflows L1 (Protenix-v2 c=256 clashed at W=512). So shrink the
        # budget ONLY in proportion to the channel's excess over 128: c=256 -> half (h_chunk
        # 16->8, fits), c<=128 -> UNCHANGED (no Boltz-2/esmfold2 regression). Blackhole keeps
        # the full budget for every channel (no small-grid path).
        transition_w_chunking_threshold = (
            SEQ_LEN_MORE_CHUNKING
            if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
            else TRANSITION_W_CHUNKING_THRESHOLD
        )
        # Size the row chunk against the width a swiglu actually sees: when W exceeds
        # the chunking threshold the W loop below never feeds swiglu more than
        # TRANSITION_W_CHUNK_SIZE columns, so dividing by the full W over-shrinks the
        # row chunk by W/w_eff (3-4x at structural scale) and degenerates to one row
        # per chunk -- thousands of tiny live buffers that fragment DRAM. The min(1.0)
        # clamp means this only ever raises the chunk size where it had been shrunk,
        # so W<=threshold shapes (every normal target) are byte-identical to before.
        w_eff = min(W, TRANSITION_W_CHUNK_SIZE) if W > transition_w_chunking_threshold else W
        _base_h = transition_h_chunk_size
        _ref = 1024 * 128
        if _IS_SMALL_GRID:
            _ref = _ref * 128 // max(128, x.shape[-1])
        transition_h_chunk_size = max(1, int(transition_h_chunk_size * min(1.0, _ref / (w_eff * x.shape[-1]))))

        # ...and raise it back where the compounded ratio above over-shrinks. That ratio divides by
        # the channel TWICE -- once here and again in the small-grid `_ref * 128 // c` -- so the
        # small-grid budget falls as 1/c^2 where Blackhole's falls as 1/c. At OpenDDE's c_z=384 that
        # costs a factor of three: the chunk is 3 against Blackhole's 10, i.e. 171 row blocks for a
        # 512-token pair tensor instead of 52. MEASURED on the Galaxy, real Transition module, 2 warm
        # + 5 timed, spread under 1.4 % (perf/wh-opendde/wh_transition_chunk.py):
        #
        #   W=512  h=3 (shipped) 72.36 ms | h=6 65.53 = 1.1035x | h=7 69.65 | h=8 68.32 | h=9 67.01
        #   W=320  h=5 (shipped) 27.50    | h=6 26.42 | h=10 25.49 = 1.0786x
        #
        # The wall is NOT monotonic in the height -- 7, 8 and 9 all fit at W=512 and are all slower
        # than 6 -- so this lands on the measured optimum rather than on the largest chunk that fits,
        # and the per-core cap below still has the last word. Fold-level A/B at 512 aa: 189.398 ->
        # 185.899 s, 1.0188x, against an A/A floor of 0.235 s, CIF digest identical on both arms.
        #
        # Scoped to 256 < c <= 384 on purpose. c=128 is already unshrunk and its shipped h=16
        # measured fastest with no clash at any height; c=256 measures 1.0738x at W=512 but is
        # protenix-v2's channel and the cap above is its crash fix, so it is left exactly alone.
        # The W/H guard keeps this to the regime where every arm was torch.equal: above the
        # 608-token thresholds the loop also W-chunks and the ragged tail re-rounds with the row
        # height (max abs diff 0.015625, bf16).
        _c = int(x.shape[-1])
        if (_IS_SMALL_GRID and SMALL_GRID_TRANSITION_ELEMS
                and 256 < _c <= SMALL_GRID_TRANSITION_MAX_C
                and W <= transition_w_chunking_threshold and H <= SEQ_LEN_MORE_CHUNKING):
            transition_h_chunk_size = max(transition_h_chunk_size,
                                          min(_base_h, SMALL_GRID_TRANSITION_ELEMS // (w_eff * _c)))
        if _IS_SMALL_GRID:
            # Cap the row chunk by measured per-core L1. The budget above scales by per-core L1
            # and by the channel excess; neither term sees that the Galaxy has 45% fewer cores
            # than the 130-core grid these constants were fitted on, so the same nominal chunk
            # lands as ~1.8x the per-core bytes and clashes. Live per chunk: x_norm (c) plus
            # x_1 and x_2 (hidden each), bf16. Every extent is TILE-PADDED: a (1,H,W,c) tile
            # tensor rounds W and c up to 32, so the logical extents understate the real
            # footprint by up to 32/W. At 298 aa, W pads 298 -> 320 and the same chunk costs
            # 409,600 B/core, not the 381,440 the logical arithmetic reports -- across the
            # measured throw edge (perf/wh-protenix/wh_transition_h.py: fits <= 393,216,
            # throws >= 409,600), which is why 298 aa still clashed in the confidence
            # Pairformer after the cap landed. Tile-aligned W is unaffected by construction.
            _gx, _gy = COMPUTE_GRID_MAIN
            _hid = int(self.fc1_weight.shape[-1])
            _tile = lambda v: -(-int(v) // 32) * 32
            _cap = max(1, int(TRANSITION_L1_CHUNK_BYTES_PER_CORE * _gx * _gy
                              // (2 * _tile(w_eff) * (_tile(x.shape[-1]) + 2 * _tile(_hid)))))
            transition_h_chunk_size = min(transition_h_chunk_size, _cap)
        if H > SEQ_LEN_MORE_CHUNKING:
            # Large-sequence path: slice row blocks lazily inside the loop. ttnn.chunk
            # would materialise a full second copy of the pair tensor up front, and the
            # list comprehension accumulates a full set of outputs before concat adds a
            # third; here the peak is input + accumulated outputs + one block. swiglu is
            # row-local, so block boundaries do not change a single output byte.
            # Host-assemble the row blocks when the full result is large enough that
            # the concat's full-size allocation would risk a fragmented-DRAM refusal
            # (CONCAT_HOST_BYTES). Guarded on the swiglu output dtype being bf16.
            host_acc = _host_concat(x) and (self.dtype or _dtype()) == ttnn.bfloat16
            parts = []
            for s in range(0, H, transition_h_chunk_size):
                c = x[:, s:min(s + transition_h_chunk_size, H)]
                if W <= transition_w_chunking_threshold:
                    _acc_append(parts, swiglu(c), host_acc)
                    ttnn.deallocate(c)
                else:
                    w_parts = []
                    for w in range(0, W, TRANSITION_W_CHUNK_SIZE):
                        cw = c[:, :, w:min(w + TRANSITION_W_CHUNK_SIZE, W), :]
                        w_parts.append(swiglu(cw))
                        ttnn.deallocate(cw)
                    ttnn.deallocate(c)
                    _acc_append(parts, ttnn.concat(w_parts, dim=2), host_acc)
                    for wp in w_parts:
                        ttnn.deallocate(wp)
            dram_peak(f"transition4d loop done (lazy, h={transition_h_chunk_size}) [z={'x'.join(str(d) for d in x.shape)}]")
            return _acc_concat(parts, 1, host_acc)
        chunks = ttnn.chunk(x, -(-H // transition_h_chunk_size), dim=1)
        dram_peak(f"transition4d chunked (eager, h={transition_h_chunk_size}) [z={'x'.join(str(d) for d in x.shape)}]")
        if W <= transition_w_chunking_threshold:
            return ttnn.concat([swiglu(c) for c in chunks], dim=1)
        return ttnn.concat([
            ttnn.concat([swiglu(c[:, :, w:min(w+TRANSITION_W_CHUNK_SIZE, W), :]) for w in range(0, W, TRANSITION_W_CHUNK_SIZE)], dim=2)
            for c in chunks
        ], dim=1)


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int | None,
        att_n_heads: int | None,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
        scale_pair_bias: bool = True,
        fp32_softmax: bool = False,
        gated_move: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.transform_s = transform_s
        self.triangle_multiplication_start = TriangleMultiplication(
            False, self.scope("tri_mul_out"), compute_kernel_config, gated_move=gated_move
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, self.scope("tri_mul_in"), compute_kernel_config, gated_move=gated_move
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            self.scope("tri_att_start", "mha."),
            compute_kernel_config,
            affinity=affinity,
            scale_pair_bias=scale_pair_bias,
            fp32_softmax=fp32_softmax,
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            self.scope("tri_att_end", "mha."),
            compute_kernel_config,
            affinity=affinity,
            scale_pair_bias=scale_pair_bias,
            fp32_softmax=fp32_softmax,
        )
        self.transition_z = Transition(
            self.scope("transition_z"), compute_kernel_config
        )
        if transform_s:
            self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight")
            self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias")
            self.attention_pair_bias = AttentionPairBias(
                att_head_dim,
                att_n_heads,
                True,
                False,
                self.scope("attention"),
                compute_kernel_config,
                scale_pair_bias=scale_pair_bias,
                fp32_softmax=fp32_softmax,
            )
            self.transition_s = Transition(
                self.scope("transition_s"), compute_kernel_config
            )

    def __call__(
        self, s: ttnn.Tensor | None, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
        attn_mask_start: ttnn.Tensor | None = None, attn_mask_end: ttnn.Tensor | None = None,
        extra_attn_bias: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        z_update = self.triangle_multiplication_start(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_multiplication_end(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_attention_start(z, attn_mask_start)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_attention_end(z, attn_mask_end)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.transition_z(z)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        if self.transform_s:
            s_norm = ttnn.layer_norm(
                s,
                weight=self.pre_norm_s_weight,
                bias=self.pre_norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            s_update = self.attention_pair_bias(
                s_norm,
                z,
                seq_mask=extra_attn_bias if extra_attn_bias is not None else attn_mask_start,
            )
            ttnn.deallocate(s_norm)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)

            s_update = self.transition_s(s)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)
        return s, z


class Pairformer(Module):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int | None,
        att_n_heads: int | None,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
        scale_pair_bias: bool = True,
        fp32_softmax: bool = False,
        gated_move: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairformerLayer(
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                transform_s,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
                affinity=affinity,
                scale_pair_bias=scale_pair_bias,
                fp32_softmax=fp32_softmax,
                gated_move=gated_move,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self, s: ttnn.Tensor | None, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
        attn_mask_start: ttnn.Tensor | None = None, attn_mask_end: ttnn.Tensor | None = None,
        extra_attn_bias: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        # Tagged so the pair term (z copies) can be read off a real trace instead of assumed:
        # the MSA trunk's peak is floor + k*m_feat + pair_copies*z, and only a measurement
        # separates the two. No-op unless TT_BIO_DRAM_PEAK is set.
        dram_peak(f"pairformer enter [z={'x'.join(str(d) for d in z.shape)}]")
        for i, block in enumerate(self.blocks):
            s, z = block(s, z, mask, attn_mask_start, attn_mask_end, extra_attn_bias)
            dram_peak(f"pairformer block {i} done")
        return s, z


# ---------------------------------------------------------------------------
# fp32-on-device pairformer for the Boltz-2 affinity trunk. Affinity model only
# (Boltz2.affinity_trunk_fp32); the structure model keeps the bf16 Pairformer.
#
# Composed ttnn fp32 ops, reference-exact against tt_bio/reference.py's
# PairformerLayer. Deliberately separate from the shipped bf16 Pairformer: that
# module's chunking, fused kernels and tuned program configs are bf16-sized
# (_pair_proj_minimal_matmul refuses non-bf16 operands, _pair_proj_program_config
# prices 2 bytes/element), and the affinity targets pad to L<=192, so none of
# the large-sequence machinery is needed. fp32 op costs at L=192 were measured
# in perf/boltz2-affinity-device-fp32-trunk/screen_fp32_ops.py (p150a): a fp32
# block is ~15 ms device-busy vs ~11 ms bf16 vs ~101 ms host fp32.
# ---------------------------------------------------------------------------


class Fp32TriangleMultiplication(Module):
    """TriangleMultiplicationOutgoing/Incoming in fp32, reference-exact."""

    def __init__(
        self,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        fp32 = ttnn.float32
        self.in_norm_weight = self.torch_to_tt("norm_in.weight", dtype=fp32)
        self.in_norm_bias = self.torch_to_tt("norm_in.bias", dtype=fp32)
        self.out_norm_weight = self.torch_to_tt("norm_out.weight", dtype=fp32)
        self.out_norm_bias = self.torch_to_tt("norm_out.bias", dtype=fp32)
        self.p_in_weight = self.torch_to_tt("p_in.weight", dtype=fp32)
        self.g_in_weight = self.torch_to_tt("g_in.weight", dtype=fp32)
        self.p_out_weight = self.torch_to_tt("p_out.weight", dtype=fp32)
        self.g_out_weight = self.torch_to_tt("g_out.weight", dtype=fp32)

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        ckc = self.compute_kernel_config
        fp32 = ttnn.float32
        x_in = ttnn.layer_norm(
            x, weight=self.in_norm_weight, bias=self.in_norm_bias,
            epsilon=1e-5, compute_kernel_config=ckc,
        )
        p = ttnn.linear(x_in, self.p_in_weight, compute_kernel_config=ckc, dtype=fp32)
        g = ttnn.linear(x_in, self.g_in_weight, compute_kernel_config=ckc, dtype=fp32)
        gp = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        if mask is not None:
            # mask is already [1, N, N, 1] fp32
            gp = ttnn.multiply_(gp, mask)
        a, b = ttnn.chunk(gp, chunks=2, dim=-1)
        ttnn.deallocate(gp)
        # outgoing: einsum("bikd,bjkd->bijd"); incoming: einsum("bkid,bkjd->bijd"),
        # each as permute -> batched matmul over the channel axis -> permute back.
        perm_a = (0, 3) + ((2, 1) if self.ending else (1, 2))
        perm_b = (0, 3) + ((1, 2) if self.ending else (2, 1))
        ap = ttnn.permute(a, perm_a)
        ttnn.deallocate(a)
        bp = ttnn.permute(b, perm_b)
        ttnn.deallocate(b)
        xc = batched_matmul(ap, bp, compute_kernel_config=ckc)
        ttnn.deallocate(ap)
        ttnn.deallocate(bp)
        xc = ttnn.permute(xc, (0, 2, 3, 1))
        xn = ttnn.layer_norm(
            xc, weight=self.out_norm_weight, bias=self.out_norm_bias,
            epsilon=1e-5, compute_kernel_config=ckc,
        )
        ttnn.deallocate(xc)
        out = ttnn.linear(xn, self.p_out_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(xn)
        g_out = ttnn.linear(x_in, self.g_out_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(x_in)
        out = ttnn.multiply_(
            out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        ttnn.deallocate(g_out)
        return out


class Fp32TriangleAttention(Module):
    """TriangleAttentionStartingNode/EndingNode in fp32, reference-exact.

    Raw-matmul attention: ttnn's fused SDPA rejects fp32 operands (bf16/bf8
    only). q is pre-scaled by 1/sqrt(head_dim) exactly as the reference's
    _prep_qkv, the pair bias is projected from the LN'd input, and the pair mask
    rides as an additive [I, 1, 1, J] bias (1e9, the reference's inf).
    """

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        fp32 = ttnn.float32
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight", dtype=fp32)
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias", dtype=fp32)
        self.bias_weight = self.torch_to_tt("linear.weight", dtype=fp32)
        self.q_weight = self.torch_to_tt("linear_q.weight", dtype=fp32)
        self.k_weight = self.torch_to_tt("linear_k.weight", dtype=fp32)
        self.v_weight = self.torch_to_tt("linear_v.weight", dtype=fp32)
        self.g_weight = self.torch_to_tt("linear_g.weight", dtype=fp32)
        self.o_weight = self.torch_to_tt("linear_o.weight", dtype=fp32)

    def __call__(self, x: ttnn.Tensor, mask_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        ckc = self.compute_kernel_config
        fp32 = ttnn.float32
        # Drop the unit batch dim; the reshape aliases the caller's pair tensor
        # for the starting variant, so it is never deallocated here.
        x = ttnn.reshape(x, tuple(x.shape)[1:])  # [I, J, C]
        if self.ending:
            xt = ttnn.transpose(x, 0, 1)
            xn = ttnn.layer_norm(
                xt, weight=self.layer_norm_weight, bias=self.layer_norm_bias,
                epsilon=1e-5, compute_kernel_config=ckc,
            )
            ttnn.deallocate(xt)
        else:
            xn = ttnn.layer_norm(
                x, weight=self.layer_norm_weight, bias=self.layer_norm_bias,
                epsilon=1e-5, compute_kernel_config=ckc,
            )
        I, J, C = (int(d) for d in xn.shape)
        H, hd = self.n_heads, self.head_dim
        # Pair bias from the normed input: [I,J,H] -> [1,H,I,J].
        bias = ttnn.linear(xn, self.bias_weight, compute_kernel_config=ckc, dtype=fp32)
        bias = ttnn.permute(bias, (2, 0, 1))
        bias = ttnn.unsqueeze(bias, 0)

        def heads(t: ttnn.Tensor) -> ttnn.Tensor:
            # [I,J,H*hd] -> [I,H,J,hd]; both reshape dims are tile-aligned.
            t = ttnn.reshape(t, (I, J, H, hd))
            return ttnn.permute(t, (0, 2, 1, 3))

        q = heads(ttnn.linear(xn, self.q_weight, compute_kernel_config=ckc, dtype=fp32))
        k = heads(ttnn.linear(xn, self.k_weight, compute_kernel_config=ckc, dtype=fp32))
        v = heads(ttnn.linear(xn, self.v_weight, compute_kernel_config=ckc, dtype=fp32))
        g = ttnn.linear(xn, self.g_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(xn)
        q = ttnn.multiply_(q, hd ** -0.5)
        kt = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        sc = batched_matmul(q, kt, compute_kernel_config=ckc)  # [I,H,J,J]
        ttnn.deallocate(q)
        ttnn.deallocate(kt)
        sc = ttnn.add_(sc, bias)  # [1,H,I,J] broadcasts over the row (batch) dim
        ttnn.deallocate(bias)
        if mask_bias is not None:
            # [I,1,1,J]: depends on (row, key), so it rides the scores, not the
            # bias; broadcasts over heads and queries.
            sc = ttnn.add_(sc, mask_bias)
        probs = ttnn.softmax(sc, dim=-1, compute_kernel_config=ckc)
        ttnn.deallocate(sc)
        o = batched_matmul(probs, v, compute_kernel_config=ckc)  # [I,H,J,hd]
        ttnn.deallocate(probs)
        ttnn.deallocate(v)
        # Merge heads via the padding-free (H, hd) merge: [I,H,J,hd] ->
        # [I,H,hd,J] -> [I,H*hd,J] -> [I,J,H*hd].
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (I, H * hd, J))
        o = ttnn.permute(o, (0, 2, 1))
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        out = ttnn.linear(o, self.o_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(o)
        if self.ending:
            out = ttnn.transpose(out, 0, 1)
        return ttnn.reshape(out, (1, I, J, C))


class Fp32Transition(Module):
    """Transition (SwiGLU MLP) in fp32, reference-exact, no chunking."""

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        fp32 = ttnn.float32
        self.norm_weight = self.torch_to_tt("norm.weight", dtype=fp32)
        self.norm_bias = self.torch_to_tt("norm.bias", dtype=fp32)
        self.fc1_weight = self.torch_to_tt("fc1.weight", dtype=fp32)
        self.fc2_weight = self.torch_to_tt("fc2.weight", dtype=fp32)
        self.fc3_weight = self.torch_to_tt("fc3.weight", dtype=fp32)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ckc = self.compute_kernel_config
        fp32 = ttnn.float32
        xn = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias,
            epsilon=1e-5, compute_kernel_config=ckc,
        )
        x1 = ttnn.linear(
            xn, self.fc1_weight, activation="silu",
            compute_kernel_config=ckc, dtype=fp32,
        )
        x2 = ttnn.linear(xn, self.fc2_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(xn)
        x1 = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        out = ttnn.linear(x1, self.fc3_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(x1)
        return out


class Fp32AttentionPairBias(Module):
    """AttentionPairBias (single-track attention with pair bias) in fp32.

    head_dim 24 is sub-tile: the q/k/v projection columns are zero-padded to 32
    per head (exact: padded channels contribute nothing to q.k and are sliced
    off before the head merge), the same trick the bf16 module uses. Attention
    is raw matmul + softmax in fp32 (fused SDPA rejects fp32 operands).
    """

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        pad = -head_dim % 32
        self.padded_head_dim = head_dim + pad
        fp32 = ttnn.float32

        def pad_head(w: torch.Tensor) -> torch.Tensor:
            # [H*hd, C] -> [H*padded_hd, C], zero rows in the padded channels.
            w = w.reshape(self.n_heads, head_dim, -1)
            w = torch.nn.functional.pad(w, (0, 0, 0, pad))
            return w.reshape(self.n_heads * self.padded_head_dim, -1)

        def pad_bias(b: torch.Tensor) -> torch.Tensor:
            b = b.reshape(self.n_heads, head_dim)
            b = torch.nn.functional.pad(b, (0, pad))
            return b.reshape(self.n_heads * self.padded_head_dim)

        self.q_weight = self.torch_to_tt(
            "proj_q.weight", transform=lambda w: pad_head(w).t(), dtype=fp32
        )
        self.q_bias = self.torch_to_tt(
            "proj_q.bias", transform=pad_bias, dtype=fp32
        )
        self.k_weight = self.torch_to_tt(
            "proj_k.weight", transform=lambda w: pad_head(w).t(), dtype=fp32
        )
        self.v_weight = self.torch_to_tt(
            "proj_v.weight", transform=lambda w: pad_head(w).t(), dtype=fp32
        )
        self.g_weight = self.torch_to_tt("proj_g.weight", dtype=fp32)
        self.o_weight = self.torch_to_tt("proj_o.weight", dtype=fp32)
        self.z_norm_weight = self.torch_to_tt("proj_z.0.weight", dtype=fp32)
        self.z_norm_bias = self.torch_to_tt("proj_z.0.bias", dtype=fp32)
        self.z_weight = self.torch_to_tt("proj_z.1.weight", dtype=fp32)

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        seq_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        ckc = self.compute_kernel_config
        fp32 = ttnn.float32
        B, L = int(s.shape[0]), int(s.shape[1])
        H, hd, phd = self.n_heads, self.head_dim, self.padded_head_dim
        zn = ttnn.layer_norm(
            z, weight=self.z_norm_weight, bias=self.z_norm_bias,
            epsilon=1e-5, compute_kernel_config=ckc,
        )
        bias = ttnn.linear(zn, self.z_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(zn)
        bias = ttnn.permute(bias, (0, 3, 1, 2))  # [B,H,L,L]
        if seq_mask is not None:
            bias = ttnn.add_(bias, seq_mask)  # [B,1,1,L] additive key mask

        def qkv(w: ttnn.Tensor, b: ttnn.Tensor | None = None) -> ttnn.Tensor:
            t = ttnn.linear(s, w, bias=b, compute_kernel_config=ckc, dtype=fp32)
            t = ttnn.reshape(t, (B, L, H, phd))
            return ttnn.permute(t, (0, 2, 1, 3))  # [B,H,L,phd]

        q = qkv(self.q_weight, self.q_bias)
        k = qkv(self.k_weight)
        v = qkv(self.v_weight)
        kt = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        sc = batched_matmul(q, kt, compute_kernel_config=ckc)  # [B,H,L,L]
        ttnn.deallocate(q)
        ttnn.deallocate(kt)
        sc = ttnn.multiply_(sc, hd ** -0.5)
        sc = ttnn.add_(sc, bias)
        ttnn.deallocate(bias)
        probs = ttnn.softmax(sc, dim=-1, compute_kernel_config=ckc)
        ttnn.deallocate(sc)
        o = batched_matmul(probs, v, compute_kernel_config=ckc)  # [B,H,L,phd]
        ttnn.deallocate(probs)
        ttnn.deallocate(v)
        # Head merge with the sub-tile slice, the same sequence the bf16 module
        # uses: [B,H,L,phd] -> [B,H,L,hd] -> [B,H,hd,L] -> [B,H*hd,L] -> [B,L,H*hd].
        o = o[:, :, :, :hd]
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (B, H * hd, L))
        o = ttnn.permute(o, (0, 2, 1))
        g = ttnn.linear(s, self.g_weight, compute_kernel_config=ckc, dtype=fp32)
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        out = ttnn.linear(o, self.o_weight, compute_kernel_config=ckc, dtype=fp32)
        ttnn.deallocate(o)
        return out


class Fp32PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.transform_s = transform_s
        self.triangle_multiplication_start = Fp32TriangleMultiplication(
            False, self.scope("tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = Fp32TriangleMultiplication(
            True, self.scope("tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = Fp32TriangleAttention(
            tri_att_head_dim, tri_att_n_heads, False,
            self.scope("tri_att_start", "mha."), compute_kernel_config,
        )
        self.triangle_attention_end = Fp32TriangleAttention(
            tri_att_head_dim, tri_att_n_heads, True,
            self.scope("tri_att_end", "mha."), compute_kernel_config,
        )
        self.transition_z = Fp32Transition(
            self.scope("transition_z"), compute_kernel_config
        )
        if transform_s:
            self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight", dtype=ttnn.float32)
            self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias", dtype=ttnn.float32)
            self.attention_pair_bias = Fp32AttentionPairBias(
                att_head_dim, att_n_heads,
                self.scope("attention"), compute_kernel_config,
            )
            self.transition_s = Fp32Transition(
                self.scope("transition_s"), compute_kernel_config
            )

    def __call__(
        self,
        s: ttnn.Tensor | None,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        tri_attn_mask: ttnn.Tensor | None = None,
        s_attn_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        z_update = self.triangle_multiplication_start(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        z_update = self.triangle_multiplication_end(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        z_update = self.triangle_attention_start(z, tri_attn_mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        z_update = self.triangle_attention_end(z, tri_attn_mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        z_update = self.transition_z(z)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        if self.transform_s:
            s_norm = ttnn.layer_norm(
                s, weight=self.pre_norm_s_weight, bias=self.pre_norm_s_bias,
                epsilon=1e-5, compute_kernel_config=self.compute_kernel_config,
            )
            s_update = self.attention_pair_bias(s_norm, z, seq_mask=s_attn_mask)
            ttnn.deallocate(s_norm)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)
            s_update = self.transition_s(s)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)
        return s, z


class Fp32Pairformer(Module):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            Fp32PairformerLayer(
                tri_att_head_dim, tri_att_n_heads, att_head_dim, att_n_heads,
                transform_s, self.scope(f"layers.{i}"), compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self,
        s: ttnn.Tensor | None,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        tri_attn_mask: ttnn.Tensor | None = None,
        s_attn_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask, tri_attn_mask, s_attn_mask)
        return s, z


class MiniTriangularUpdate(Module):
    """Bi-directional triangular multiplicative update (BoltzGen Miniformer).

    Equivalent to PyTorch reference (boltzgen/.../triangular.py:MiniTriangularUpdate):

        x = norm_in(x)
        x = p_in(x) * sigmoid(g_in(x))        # (B, N, N, D)
        x = x * mask.unsqueeze(-1)
        a1, b1, a2, b2 = chunk(x, 4, dim=-1)  # 4 x (B, N, N, D/4)
        x1 = einsum("bikd,bjkd->bijd", a1, b1)  # outgoing-style
        x2 = einsum("bkid,bkjd->bijd", a2, b2)  # incoming-style
        x = cat([x1, x2], -1)                 # (B, N, N, D/2)
        x = norm_out(x)
        return p_out(x) * sigmoid(g_out(x))   # (B, N, N, D)

    Each einsum decomposes to a permute-matmul-permute, reusing the same
    permutation pattern as TriangleMultiplication (outgoing=False / incoming=True).
    """

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_in_weight = self.torch_to_tt("norm_in.weight")
        self.norm_in_bias = self.torch_to_tt("norm_in.bias")
        self.p_in_weight = self.torch_to_tt("p_in.weight")
        self.g_in_weight = self.torch_to_tt("g_in.weight")
        self.norm_out_weight = self.torch_to_tt("norm_out.weight")
        self.norm_out_bias = self.torch_to_tt("norm_out.bias")
        self.p_out_weight = self.torch_to_tt("p_out.weight")
        self.g_out_weight = self.torch_to_tt("g_out.weight")

    @staticmethod
    def _matmul_einsum(
        a: ttnn.Tensor,
        b: ttnn.Tensor,
        ending: bool,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """Compute the einsum bikd,bjkd->bijd (ending=False, outgoing) or
        bkid,bkjd->bijd (ending=True, incoming) via permute-matmul-permute."""
        a_perm = (0, 3) + ((2, 1) if ending else (1, 2))
        b_perm = (0, 3) + ((1, 2) if ending else (2, 1))
        ap = ttnn.permute(a, a_perm, memory_config=memory_config)
        bp = ttnn.permute(b, b_perm, memory_config=memory_config)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        out = ttnn.matmul(
            ap,
            bp,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(ap)
        ttnn.deallocate(bp)
        return ttnn.permute(out, (0, 2, 3, 1), memory_config=memory_config)

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        seq_len = x.shape[1]
        memory_config = _triangle_mul_memory_config(seq_len)

        x_norm = ttnn.layer_norm(
            x,
            weight=self.norm_in_weight,
            bias=self.norm_in_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        p = ttnn.linear(
            x_norm,
            self.p_in_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        g = ttnn.linear(
            x_norm,
            self.g_in_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x_norm)
        x_gated = ttnn.multiply_(
            p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        ttnn.deallocate(g)
        if mask is not None:
            x_gated = ttnn.multiply_(x_gated, ttnn.unsqueeze(mask, -1))

        a1, b1, a2, b2 = ttnn.chunk(x_gated, chunks=4, dim=-1)
        ttnn.deallocate(x_gated)

        x1 = self._matmul_einsum(
            a1, b1, ending=False,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        x2 = self._matmul_einsum(
            a2, b2, ending=True,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        x = ttnn.concat([x1, x2], dim=-1)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)

        x = ttnn.layer_norm(
            x,
            weight=self.norm_out_weight,
            bias=self.norm_out_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        p_out = ttnn.linear(
            x,
            self.p_out_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        g_out = ttnn.linear(
            x,
            self.g_out_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x)
        return ttnn.multiply_(
            p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )


class MiniformerLayer(Module):
    """One Miniformer block: triangular + attention on s + transitions."""

    def __init__(
        self,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.triangular = MiniTriangularUpdate(
            self.scope("triangular"), compute_kernel_config
        )
        self.transition_z = Transition(self.scope("transition_z"), compute_kernel_config)
        self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight")
        self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias")
        self.attention_pair_bias = AttentionPairBias(
            att_head_dim,
            att_n_heads,
            True,
            False,
            self.scope("attention"),
            compute_kernel_config,
        )
        self.transition_s = Transition(self.scope("transition_s"), compute_kernel_config)

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        z_update = self.triangular(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.transition_z(z)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        s_norm = ttnn.layer_norm(
            s,
            weight=self.pre_norm_s_weight,
            bias=self.pre_norm_s_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_update = self.attention_pair_bias(s_norm, z, seq_mask=seq_mask)
        ttnn.deallocate(s_norm)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)

        s_update = self.transition_s(s)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)
        return s, z


class Miniformer(Module):
    def __init__(
        self,
        n_blocks: int,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            MiniformerLayer(
                att_head_dim,
                att_n_heads,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask, seq_mask)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        # L6's memo: the (s_scale, s_bias) pair and the identity of the `s` it was computed from.
        # Kept apart so a reset can free the pair without touching the caller's `s`.
        self._s_memo = None
        self._s_memo_src = None
        self.s_norm_weight = self.torch_to_tt("s_norm.weight", dtype=dtype)
        self.s_scale_weight = self.torch_to_tt("s_scale.weight", dtype=dtype)
        self.s_scale_bias = self.torch_to_tt("s_scale.bias", dtype=dtype)
        self.s_bias_weight = self.torch_to_tt("s_bias.weight", dtype=dtype)

    def s_terms(self, s: ttnn.Tensor, large_seq_len: bool = False):
        """``(s_scale, s_bias)``: the conditioning half, a pure function of ``s``.

        Split out so a caller whose ``s`` is a loop invariant computes it once instead of
        once per call. The diffusion rollout is exactly that case: 401 atom-transformer
        calls per fold, 9 AdaLNs each, and ``s`` is the atom conditioning ``cl``, which
        does not depend on the noise level or the noisy coordinates."""
        memo = self.atom_level and _B2_ADALN_S_MEMO
        if memo and self._s_memo is not None and self._s_memo_src is s:
            return self._s_memo
        # The memo keys on the CALLER's `s` and is taken before the L1 conversion below. Keying
        # on the converted copy instead both misses every time (the copy is fresh per call) and
        # pins 12 x 1.83 MB of L1 for the whole rollout, which throws `Statically allocated
        # circular buffers ... clash with L1 buffers` in the confidence stack downstream.
        s_src = s
        memory_config = _adaln_memory_config(self.atom_level, large_seq_len)
        if self.atom_level:
            s = ttnn.to_memory_config(s, memory_config=memory_config)
        s = ttnn.layer_norm(
            s,
            weight=self.s_norm_weight,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_scale = ttnn.linear(
            s,
            self.s_scale_weight,
            bias=self.s_scale_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            #core_grid=ttnn.CoreGrid(y=10, x=11), CAUSES ACCURACY ISSUE
        )
        s_bias = ttnn.linear(
            s,
            self.s_bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            #core_grid=ttnn.CoreGrid(y=10, x=11), CAUSES ACCURACY ISSUE
        )
        if memo:
            # DRAM, not the L1 `memory_config` above: 24 pairs of 1.83 MB retained for the whole
            # rollout would hold ~44 MB of L1 and clash with a later op's circular buffers.
            # A memory config does not change values, so the pair stays bit-identical.
            s_scale = ttnn.to_memory_config(s_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s_bias = ttnn.to_memory_config(s_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._s_memo = (s_scale, s_bias)
            self._s_memo_src = s_src
            # Return the memo OBJECT, not a fresh tuple: `__call__` decides ownership by identity
            # against it, and a fresh tuple makes the storing call deallocate what it just stored.
            return self._s_memo
        return s_scale, s_bias

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, large_seq_len: bool = False,
                 s_terms=None) -> ttnn.Tensor:
        memory_config = _adaln_memory_config(self.atom_level, large_seq_len)
        if self.atom_level:
            a = ttnn.to_memory_config(a, memory_config=memory_config)
        a = ttnn.layer_norm(
            a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config
        )
        own = s_terms is None
        if own:
            s_terms = self.s_terms(s, large_seq_len)
            # A memoised pair belongs to the memo and must survive this call.
            own = self._s_memo is not s_terms
        s_scale, s_bias = s_terms
        a = ttnn.multiply_(a, s_scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        a = ttnn.add_(a, s_bias)
        if own:                     # a cached pair belongs to the caller
            ttnn.deallocate(s_scale)
            ttnn.deallocate(s_bias)
        a = ttnn.to_memory_config(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, self.scope("adaln"), compute_kernel_config
        )
        swish_chunk, gates_chunk = torch.chunk(self.weights["swish_gate.0.weight"], chunks=2, dim=0)
        self.swish_weight, self.gates_weight = [
            ttnn.from_torch(chunk.t(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=_dtype(ttnn.bfloat16))
            for chunk in [swish_chunk, gates_chunk]
        ]
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(
        self, a: ttnn.Tensor, s: ttnn.Tensor, large_seq_len: bool = False
    ) -> ttnn.Tensor:
        a = self.adaln(a, s, large_seq_len=large_seq_len)
        a_swish = ttnn.linear(
            a,
            self.swish_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        gates = ttnn.linear(
            a,
            self.gates_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        a_swish = ttnn.multiply_(gates, a_swish, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        a_b = ttnn.linear(
            a,
            self.a_to_b_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(a)
        b = ttnn.multiply_(a_swish, a_b)
        ttnn.deallocate(a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        b_a = ttnn.linear(
            b,
            self.b_to_a_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(b)
        a = ttnn.multiply_(s, b_a, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(b_a)
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.s_o = None
        self.adaln = AdaLN(
            atom_level, self.scope("adaln"), compute_kernel_config
        )
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            compute_pair_bias=False,
            atom_level=atom_level,
            state_dict=self.scope("pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.attn_pair_bias.token_dit = not atom_level
        self.output_projection_weight = self.torch_to_tt(
            "output_projection_linear.weight"
        )
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            atom_level,
            self.scope("transition"),
            compute_kernel_config,
        )

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        b = self.adaln(a, s, large_seq_len=large_seq_len)
        if not self.atom_level:
            b = self.attn_pair_bias(b, z)
        else:
            b = self.attn_pair_bias(b, z, keys_indexing)
        if self.s_o is None:
            s_o = ttnn.linear(
                s,
                self.output_projection_weight,
                bias=self.output_projection_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                activation="sigmoid",
            )
            if self.atom_level:
                self.s_o = s_o
        else:
            s_o = self.s_o
        b = ttnn.multiply(s_o, b)
        a = ttnn.add(a, b)
        a_t = self.transition(a, s, large_seq_len=large_seq_len)
        a = ttnn.add(a, a_t)
        return a


class DiffusionTransformer(Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                atom_level,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        if isinstance(z, (list, tuple)):
            # L7: the head-ranges were cut once per fold (AtomDiffusion._hoist_layer_bias),
            # because z is constant across the whole denoise rollout.
            for layer, z_layer in zip(self.layers, z):
                a = layer(a, s, z_layer, keys_indexing, large_seq_len=large_seq_len)
            return a
        dim = z.shape[1] // len(self.layers)
        for i, layer in enumerate(self.layers):
            a = layer(
                a,
                s,
                z[:, i * dim : (i + 1) * dim, :, :],
                keys_indexing,
                large_seq_len=large_seq_len,
            )
        return a


class PairWeightedAveraging(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.m_norm_weight = self.torch_to_tt("norm_m.weight")
        self.m_norm_bias = self.torch_to_tt("norm_m.bias")
        self.z_norm_weight = self.torch_to_tt("norm_z.weight")
        self.z_norm_bias = self.torch_to_tt("norm_z.bias")
        self.m_weight = self.torch_to_tt("proj_m.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_weight = self.torch_to_tt("proj_z.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])
        z = ttnn.reshape(z, tuple(z.shape)[1:])
        m = ttnn.layer_norm(
            m,
            weight=self.m_norm_weight,
            bias=self.m_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        # One layer_norm, `n_heads` projections of it: every head reads the whole normed pair
        # tensor to write one tile of width, so all eight are source-bound and one L1-resident
        # copy serves all of them. 3572.2 -> 991.0 us on the eight-head region, `torch.equal`.
        z, z_in_l1 = (_l1_layer_norm(z, 1.5, weight=self.z_norm_weight, bias=self.z_norm_bias,
                                     epsilon=1e-5,
                                     compute_kernel_config=self.compute_kernel_config)
                      if _PWA_L1_NORM else
                      (ttnn.layer_norm(z, weight=self.z_norm_weight, bias=self.z_norm_bias,
                                       epsilon=1e-5,
                                       compute_kernel_config=self.compute_kernel_config), False))
        o_out = None
        for i in range(self.n_heads):
            zw = self.z_weight[:, i : i + 1]
            b = _narrow_proj_linear(z, zw, self.compute_kernel_config, z.dtype, l1_out=z_in_l1)
            if b is None:
                b = ttnn.linear(
                    z,
                    zw,
                    compute_kernel_config=self.compute_kernel_config,
                    core_grid=CORE_GRID_MAIN,
                )
            b = ttnn.permute(b, (2, 0, 1))
            if attn_mask is not None:
                b = ttnn.add_(b, ttnn.reshape(attn_mask, (1, 1, attn_mask.shape[-1])))
            w = ttnn.softmax(
                b,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            v = ttnn.linear(
                m,
                self.m_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            v = ttnn.permute(v, (0, 2, 1))
            o = ttnn.matmul(
                v,
                w,
                transpose_b=True,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(v)
            ttnn.deallocate(w)
            o = ttnn.permute(o, (0, 2, 1))
            g = ttnn.linear(
                m,
                self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g)
            o = ttnn.linear(
                o,
                self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            o_out = o if o_out is None else ttnn.add(o_out, o)
        o_out = ttnn.reshape(o_out, (1, *o_out.shape))
        return o_out


class OuterProductMean(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        scale_bias: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        # scale_bias: divide the proj_o bias by the row count too. The AF3/OF3
        # reference divides the WHOLE linear_out output (raw outer + bias) by the
        # pair norm; the default (Boltz/Protenix convention here) scales only the
        # raw outer product, adding the bias full-strength. For OF3 the unscaled
        # bias is a structured per-channel constant that the pairformer amplifies.
        self.scale_bias = scale_bias
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.a_weight = self.torch_to_tt("proj_a.weight")
        self.b_weight = self.torch_to_tt("proj_b.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.o_bias = self.torch_to_tt("proj_o.bias")

    def __call__(self, x: ttnn.Tensor, msa_mask: ttnn.Tensor | None = None, n_msa: int | None = None) -> ttnn.Tensor:
        # `x` may arrive as a LIST of depth chunks. The MSA trunk keeps its representation chunked
        # so it never has to exist contiguously: materialising it costs a full extra copy at the
        # join, which is what made a 1.78 GiB m_feat OOM on a 12 GiB part even WITH chunking. This
        # path consumes the chunks directly, and since only the c=32 projections are concatenated
        # the join costs 1/4 of what joining the c_m tensor would.
        x_chunks = x if isinstance(x, list) else None
        if x_chunks is None:
            x = ttnn.reshape(x, tuple(x.shape)[1:])

        def project_ab(xc, maskc):
            """layer_norm + the two c=32 projections. Every op here is per MSA row (norm over
            channels, linear over channels, mask multiply per row), so this may be applied to a
            slice of the depth axis and concatenated without changing a single number."""
            mc = ttnn.layer_norm(
                xc,
                weight=self.norm_weight,
                bias=self.norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            ac = ttnn.linear(
                mc,
                self.a_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            bc = ttnn.linear(
                mc,
                self.b_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(mc)
            if maskc is not None:
                ac = ttnn.multiply_(ac, maskc)
            return ac, bc

        # The layer_norm above is out-of-place, so unchunked it materialises a SECOND full-depth
        # c_m tensor beside `x` (2.494 GiB each for a 682-token/14860-deep target) purely to feed
        # two c=32 projections that are 1/4 the size. Chunking the depth axis keeps only the
        # projections at full depth. Measured: this is the allocation a deep-MSA target dies on
        # once the MSA-update path is chunked (refused 669,532,160 B = 32 channels x 2 B).
        #
        # For a chunk-list input the matmul's contraction along S (the depth axis) is accumulated
        # per chunk instead; every other path below keeps the single full-depth matmul. See
        # `depth_parts` and `z_rows`.
        depth_parts = None
        if x_chunks is not None:
            if msa_mask is not None:
                raise NotImplementedError(
                    "OuterProductMean: chunk-list input with an msa_mask is not wired up; the "
                    "trunk that uses the list path passes mask=None.")
            # Never materialise contiguous a/b at all.
            #
            # Joining them cannot be made to fit, and not for want of a better free order:
            # project_ab derives ac AND bc from one layer_norm, so both part lists are already
            # complete before any join could run. The FIRST concat therefore needs
            # a_parts + b_parts + a live at once -- ~1.96 GiB at 768 tokens x depth 14208, which
            # is the 698351616 B allocation 9lof dies on -- with nothing yet freeable. Freeing
            # per-side (tried, bit-exact, kept below for the non-list branch) relieves only the
            # SECOND concat, which was never the binding one. Two passes over project_ab do not
            # help either: the second still needs a + b_parts + b, and pays layer_norm twice.
            #
            # So the join is removed. `sum_c (a_c @ b_c^T)` is the same sum over the same depth
            # rows, and peak drops to a_parts + b_parts plus one z accumulator.
            #
            # This is NOT bit-exact, deliberately: it reassociates a bf16 accumulation that used
            # to be one matmul over full depth. The same question was settled for Transition --
            # scripts/abag_xm/probe_transition_vs_torch.py measured chunked and whole equally
            # close to an fp32 reference, one mantissa step apart -- but it does move the last
            # bit, so the acceptance test here is DockQ against the reference fold, NOT an md5.
            depth_parts = []
            S = 0
            for c in x_chunks:
                ac, bc = project_ab(ttnn.reshape(c, tuple(c.shape)[1:]), None)
                Sc, I, C = ac.shape
                _, J, D = bc.shape
                S += Sc
                acp = ttnn.permute(ac, (1, 2, 0))           # (I, C, Sc)
                ttnn.deallocate(ac)
                bcp = ttnn.permute(bc, (2, 1, 0))           # (D, J, Sc)
                ttnn.deallocate(bc)
                bcp = ttnn.to_layout(bcp, ttnn.ROW_MAJOR_LAYOUT)
                bcp = ttnn.reshape(bcp, (-1, Sc))           # (D*J, Sc)
                bcp = ttnn.to_layout(bcp, ttnn.TILE_LAYOUT)
                depth_parts.append((acp, bcp, Sc))
            a = b = None
        elif x.shape[0] * x.shape[1] * x.shape[2] * 2 <= OPM_ROW_CHUNK_BUDGET_BYTES:
            a, b = project_ab(x, msa_mask)
        else:
            a_parts, b_parts = [], []
            for s in range(0, x.shape[0], MSA_CHUNK_SIZE):
                e = min(s + MSA_CHUNK_SIZE, x.shape[0])
                ac, bc = project_ab(x[s:e], None if msa_mask is None else msa_mask[s:e])
                a_parts.append(ac)
                b_parts.append(bc)
            # Same per-side free as the chunk-list branch above, for the same reason.
            a = ttnn.concat(a_parts, dim=0)
            for p in a_parts:
                ttnn.deallocate(p)
            b = ttnn.concat(b_parts, dim=0)
            for p in b_parts:
                ttnn.deallocate(p)
        if depth_parts is None:
            S, I, C = a.shape
            _, J, D = b.shape
            a = ttnn.permute(a, (1, 2, 0))  # (I, C, S)
            b = ttnn.permute(b, (2, 1, 0))
            b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
            b = ttnn.reshape(b, (-1, S))
            b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
            if I > SEQ_LEN_MORE_CHUNKING:
                # Compact large tensors before OPM matmuls to reduce DRAM fragmentation.
                a = ttnn.reallocate(a)
                b = ttnn.reallocate(b)

        def z_rows(i0, i1):
            """`z = a b^T` contracted over the full depth, for token rows [i0, i1).

            One matmul when a/b are contiguous; a running sum over depth chunks when they are
            not. Both compute the same contraction over the same rows -- the chunked form just
            reassociates it, so it lands within a bf16 step rather than bit-exactly.
            """
            rows = i1 - i0

            def rows_of(t):
                """Slice the token axis, but not when the slice is the whole tensor -- a ttnn
                slice copies, and `a` is 0.65 GiB at the sizes this path exists for."""
                return t if i0 == 0 and i1 == t.shape[0] else t[i0:i1, :, :]

            if depth_parts is None:
                a_flat = ttnn.reshape(rows_of(a), (rows * C, S))
                z = ttnn.matmul(a_flat, b, transpose_b=True,
                                compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(a_flat)
                return z
            z = None
            for acp, bcp, Sc in depth_parts:
                a_flat = ttnn.reshape(rows_of(acp), (rows * C, Sc))
                zp = ttnn.matmul(a_flat, bcp, transpose_b=True,
                                 compute_kernel_config=self.compute_kernel_config)
                ttnn.deallocate(a_flat)
                if z is None:
                    z = zp
                else:
                    # In place: z is (rows*C, D*J) -- ~400 MB at rows=256, J=768 -- so an
                    # out-of-place add would hold three of them at the peak.
                    ttnn.add_(z, zp)
                    ttnn.deallocate(zp)
            return z

        def outer_product_mean(i0, i1):
            rows = i1 - i0
            z = z_rows(i0, i1)
            z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
            z = ttnn.reshape(z, (rows, C * D, J))
            z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
            z = ttnn.permute(z, (0, 2, 1))
            scale = 1 / (n_msa if n_msa is not None else S)
            z = ttnn.multiply_(z, scale)
            o_bias = self.o_bias
            if self.scale_bias:
                o_bias = ttnn.multiply(self.o_bias, scale)
            out = ttnn.linear(
                z,
                self.o_weight,
                bias=o_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            if self.scale_bias:
                ttnn.deallocate(o_bias)
            ttnn.deallocate(z)
            return out

        if I > SEQ_LEN_MORE_CHUNKING:
            # Row block sized so the per-block matmul result stays under OPM_Z_BUDGET_BYTES. That
            # result is (rows*C, D*J), so at a fixed row count it grows with J -- the constant 256
            # is fine at 285 tokens and is what 9i3p (992 padded) dies on. Never below one tile.
            per_row = C * D * J * 2
            rows_blk = max(32, min(OPM_CHUNK_SIZE,
                                   (OPM_Z_BUDGET_BYTES // max(per_row, 1)) // 32 * 32))
            z_acc = None
            for i in range(0, I, rows_blk):
                part = outer_product_mean(i, min(i + rows_blk, I))
                if z_acc is None:
                    z_acc = part
                else:
                    z_old = z_acc
                    z_acc = ttnn.concat([z_old, part], dim=0)
                    ttnn.deallocate(z_old)
                    ttnn.deallocate(part)
            z = z_acc
        else:
            z = outer_product_mean(0, I)
        if depth_parts is None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        else:
            for acp, bcp, _ in depth_parts:
                ttnn.deallocate(acp)
                ttnn.deallocate(bcp)
        z = ttnn.reshape(z, (1, *z.shape))
        return z


class MSALayer(Module):
    def __init__(
        self,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_transition = Transition(
            self.scope("msa_transition"), compute_kernel_config
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            head_dim=avg_head_dim,
            n_heads=avg_n_heads,
            state_dict=self.scope("pair_weighted_averaging"),
            compute_kernel_config=compute_kernel_config,
        )
        self.outer_product_mean = OuterProductMean(
            state_dict=self.scope("outer_product_mean"),
            compute_kernel_config=compute_kernel_config,
        )
        self.pairformer_layer = PairformerLayer(
            tri_att_head_dim,
            tri_att_n_heads,
            None,
            None,
            False,
            self.scope("pairformer_layer"),
            compute_kernel_config,
        )

    def __call__(
        self,
        z: ttnn.Tensor,
        m: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        attn_mask: ttnn.Tensor | None,
        msa_mask: ttnn.Tensor | None,
        n_msa: int | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        S = m.shape[2]
        if S > SEQ_LEN_MORE_CHUNKING:
            z = ttnn.reallocate(z)
            # Collect the row chunks and join them once, AFTER the source is freed. Joining
            # pairwise inside the loop kept three copies of the MSA representation live at the
            # last step -- `m`, the accumulator, and the concat's output. Freeing `m` first costs
            # nothing: every slice has already been taken by then. Bit-exact: concat copies rows,
            # so one N-way join writes the same bytes in the same order as N-1 pairwise ones, and
            # 9d72's 15 structure md5s reproduce exactly with this active.
            # Measured, so it is not oversold: this buys nothing on any target the AbAg-XM panel
            # is blocked on. 9i3p fails at the identical byte count with and without it, because
            # what blocks the large targets is the pair representation, not MSA occupancy.
            parts = []
            N = m.shape[1]
            for s in range(0, N, MSA_CHUNK_SIZE):
                mc = m[:, s:min(s + MSA_CHUNK_SIZE, N), :]
                mc = ttnn.add_(mc, self.pair_weighted_averaging(mc, z, attn_mask))
                mc = ttnn.add_(mc, self.msa_transition(mc))
                parts.append(mc)
                dram_peak("msalayer chunked: row loop")
            ttnn.deallocate(m)
            if len(parts) == 1:
                m = parts[0]
            else:
                m = ttnn.concat(parts, dim=1)
                for p in parts:
                    ttnn.deallocate(p)
            m = ttnn.reallocate(m)
            dram_peak("msalayer chunked: rows joined")
            z = ttnn.add_(z, self.outer_product_mean(m, msa_mask, n_msa))
            dram_peak("msalayer chunked: opm")
        else:
            m = ttnn.add_(m, self.pair_weighted_averaging(m, z, attn_mask))
            dram_peak("msalayer whole: pwa")
            m = ttnn.add_(m, self.msa_transition(m))
            dram_peak("msalayer whole: transition")
            z = ttnn.add_(z, self.outer_product_mean(m, msa_mask, n_msa))
            dram_peak("msalayer whole: opm")

        z = self.pairformer_layer(
            None, z, mask=mask, attn_mask_start=attn_mask, attn_mask_end=attn_mask,
        )[1]
        dram_peak("msalayer: pairformer layer")

        return z, m


class MSA(Module):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.s_weight = self.torch_to_tt("s_proj.weight")
        self.msa_weight = self.torch_to_tt("msa_proj.weight")
        self.blocks = [
            MSALayer(
                avg_head_dim,
                avg_n_heads,
                tri_att_head_dim,
                tri_att_n_heads,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self,
        z: ttnn.Tensor,
        m: ttnn.Tensor,
        emb: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        attn_mask: ttnn.Tensor | None,
        msa_mask: ttnn.Tensor | None,
        n_msa: int | None,
    ) -> ttnn.Tensor:
        # The MSA trunk is the campaign's DRAM limiter: `m` is (1, depth, tokens, c_m) and
        # deep-MSA targets carry several full copies of it at once. Tag the floor (weights +
        # z + the one-hot input, before the c_m projection exists) so a measured peak can be
        # decomposed into floor + k*m_feat rather than guessed at. No-op unless
        # TT_BIO_DRAM_PEAK is set.
        dram_peak(f"msa floor [depth={m.shape[1]} tokens={m.shape[2]}]")
        m = ttnn.linear(
            m,
            self.msa_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        m = ttnn.add_(
            m,
            ttnn.linear(
                emb,
                self.s_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            ),
        )
        dram_peak(f"msa m_feat projected [c_m={m.shape[-1]}]")
        for i, block in enumerate(self.blocks):
            z, m = block(z, m, mask, attn_mask, msa_mask, n_msa)
            dram_peak(f"msa block {i} done")
        return z


class Diffusion(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self._s_conditioned = None
        self._c_reshaped = None
        self.conditioner_norm_weight = self.torch_to_tt(
            "single_conditioner.norm_single.weight"
        )
        self.conditioner_norm_bias = self.torch_to_tt(
            "single_conditioner.norm_single.bias"
        )
        self.conditioner_embed_weight = self.torch_to_tt(
            "single_conditioner.single_embed.weight"
        )
        self.conditioner_embed_bias = self.torch_to_tt(
            "single_conditioner.single_embed.bias"
        )
        self.conditioner_fourier_embed_weight = self.torch_to_tt(
            "single_conditioner.fourier_embed.proj.weight"
        )
        self.conditioner_fourier_embed_bias = self.torch_to_tt(
            "single_conditioner.fourier_embed.proj.bias"
        )
        self.conditioner_norm_fourier_weight = self.torch_to_tt(
            "single_conditioner.norm_fourier.weight"
        )
        self.conditioner_norm_fourier_bias = self.torch_to_tt(
            "single_conditioner.norm_fourier.bias"
        )
        self.conditioner_fourier_single_weight = self.torch_to_tt(
            "single_conditioner.fourier_to_single.weight"
        )
        self.conditioner_transition_0 = Transition(
            self.scope("single_conditioner.transitions.0"),
            compute_kernel_config,
        )
        self.conditioner_transition_1 = Transition(
            self.scope("single_conditioner.transitions.1"),
            compute_kernel_config,
        )
        self.r_to_q_weight = self.torch_to_tt(
            "atom_attention_encoder.r_to_q_trans.weight"
        )
        self.encoder = DiffusionTransformer(
            n_layers=ATOM_N_LAYERS,
            dim=ATOM_DIM,
            n_heads=ATOM_N_HEADS,
            atom_level=True,
            state_dict=self.scope("atom_attention_encoder.atom_encoder.diffusion_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.atom_to_token_weight = self.torch_to_tt(
            "atom_attention_encoder.atom_to_token_trans.0.weight"
        )
        self.s_to_a_norm_weight = self.torch_to_tt("s_to_a_linear.0.weight")
        self.s_to_a_norm_bias = self.torch_to_tt("s_to_a_linear.0.bias")
        self.s_to_a_linear_weight = self.torch_to_tt("s_to_a_linear.1.weight")
        self.token_transformer_fp32 = _DIFFUSION_FP32_DEVICE
        token_dtype = ttnn.float32 if self.token_transformer_fp32 else None
        with device_dtype_override(token_dtype):
            self.token_transformer = DiffusionTransformer(
                n_layers=TOKEN_N_LAYERS,
                dim=TOKEN_DIM,
                n_heads=TOKEN_N_HEADS,
                atom_level=False,
                state_dict=self.scope("token_transformer"),
                compute_kernel_config=compute_kernel_config,
            )
        self.a_norm_weight = self.torch_to_tt("a_norm.weight")
        self.a_norm_bias = self.torch_to_tt("a_norm.bias")
        self.a_to_q_weight = self.torch_to_tt(
            "atom_attention_decoder.a_to_q_trans.weight"
        )
        self.decoder = DiffusionTransformer(
            n_layers=ATOM_N_LAYERS,
            dim=ATOM_DIM,
            n_heads=ATOM_N_HEADS,
            atom_level=True,
            state_dict=self.scope("atom_attention_decoder.atom_decoder.diffusion_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.feat_to_pos_norm_weight = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.0.weight"
        )
        self.feat_to_pos_norm_bias = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.0.bias"
        )
        self.feat_to_pos_linear_weight = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.1.weight"
        )

    def __call__(
        self,
        r: ttnn.Tensor,
        times: ttnn.Tensor,
        s_inputs: ttnn.Tensor,
        s_trunk: ttnn.Tensor,
        q: ttnn.Tensor,
        c: ttnn.Tensor,
        bias_encoder: ttnn.Tensor,
        bias_token: ttnn.Tensor,
        bias_decoder: ttnn.Tensor,
        keys_indexing: ttnn.Tensor,
        atom_to_token: ttnn.Tensor,
        atom_to_token_normed: ttnn.Tensor,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        B, N, D = q.shape
        NW = N // ATOM_WINDOW
        if self._s_conditioned is None:
            s = ttnn.concat([s_trunk, s_inputs], dim=-1)
            s = ttnn.layer_norm(
                s,
                weight=self.conditioner_norm_weight,
                bias=self.conditioner_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            self._s_conditioned = ttnn.linear(
                s,
                self.conditioner_embed_weight,
                bias=self.conditioner_embed_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(s)
            self._c_reshaped = ttnn.reshape(c, (B, NW, ATOM_WINDOW, -1))
        r_to_q = ttnn.linear(
            r,
            self.r_to_q_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        q = ttnn.add(q, r_to_q)
        ttnn.deallocate(r_to_q)
        q = ttnn.reshape(q, (B, NW, ATOM_WINDOW, -1))
        q = self.encoder(
            q,
            self._c_reshaped,
            bias_encoder,
            keys_indexing,
            large_seq_len=large_seq_len,
        )
        q = ttnn.reshape(q, (B, NW * ATOM_WINDOW, D))
        a = ttnn.linear(
            q,
            self.atom_to_token_weight,
            compute_kernel_config=self.compute_kernel_config,
            activation="relu",
            core_grid=CORE_GRID_MAIN,
        )
        a = ttnn.matmul(
            a,
            atom_to_token_normed,
            transpose_a=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.permute(a, (0, 2, 1))
        times = ttnn.unsqueeze(times, 1)
        fourier = ttnn.linear(
            times,
            self.conditioner_fourier_embed_weight,
            bias=self.conditioner_fourier_embed_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        fourier = ttnn.multiply(fourier, 2 * pi)
        fourier = ttnn.cos(fourier)
        fourier = ttnn.layer_norm(
            fourier,
            weight=self.conditioner_norm_fourier_weight,
            bias=self.conditioner_norm_fourier_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        fourier = ttnn.linear(
            fourier,
            self.conditioner_fourier_single_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        fourier = ttnn.unsqueeze(fourier, 1)
        s = ttnn.add(self._s_conditioned, fourier)
        ttnn.deallocate(fourier)
        s_update = self.conditioner_transition_0(s)
        s = ttnn.add(s, s_update)
        ttnn.deallocate(s_update)
        s_update = self.conditioner_transition_1(s)
        s = ttnn.add(s, s_update)
        ttnn.deallocate(s_update)
        s_to_a = ttnn.layer_norm(
            s,
            weight=self.s_to_a_norm_weight,
            bias=self.s_to_a_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_to_a = ttnn.linear(
            s_to_a,
            self.s_to_a_linear_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        a = ttnn.add(a, s_to_a)
        ttnn.deallocate(s_to_a)
        if self.token_transformer_fp32:
            a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=a.memory_config())
            s_fp32 = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
            bias_fp32 = ttnn.typecast(
                bias_token, ttnn.float32, memory_config=bias_token.memory_config()
            )
            a_fp32 = self.token_transformer(a_fp32, s_fp32, bias_fp32)
            a = ttnn.typecast(
                a_fp32, ttnn.bfloat16, memory_config=a_fp32.memory_config()
            )
            for tensor in (a_fp32, s_fp32, bias_fp32):
                ttnn.deallocate(tensor)
        else:
            a = self.token_transformer(a, s, bias_token)
        ttnn.deallocate(s)
        a = ttnn.layer_norm(
            a,
            weight=self.a_norm_weight,
            bias=self.a_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        a_to_q = ttnn.linear(
            a,
            self.a_to_q_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        # Keep explicit 3D axis reorder for this batched path; operand swapping
        # changed semantics under stricter matmul batch validation.
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        a_to_q = ttnn.matmul(
            a_to_q,
            atom_to_token,
            transpose_b=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        q = ttnn.add(q, a_to_q)
        ttnn.deallocate(a_to_q)
        q = ttnn.reshape(q, (B, NW, ATOM_WINDOW, -1))
        q = self.decoder(
            q,
            self._c_reshaped,
            bias_decoder,
            keys_indexing,
            large_seq_len=large_seq_len,
        )
        q = ttnn.reshape(q, (B, NW * ATOM_WINDOW, D))
        r_update = ttnn.layer_norm(
            q,
            weight=self.feat_to_pos_norm_weight,
            bias=self.feat_to_pos_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        r_update = ttnn.linear(
            r_update,
            self.feat_to_pos_linear_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(q)
        return r_update


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        self.tt_device = get_device()
        self._runtime_cache = {}
        self._first_forward_pass = True
        kernel_cls = (
            ttnn.types.WormholeComputeKernelConfig
            if self.tt_device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig
        )
        self.compute_kernel_config = kernel_cls(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=self.tt_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)

    def _cache_set(self, key: str, value):
        self._runtime_cache[key] = value
        return value

    def _cache_get(self, key: str, default=None):
        return self._runtime_cache.get(key, default)

    def _cache_has_all(self, keys: tuple[str, ...]) -> bool:
        return all(key in self._runtime_cache for key in keys)

    def _deallocate_tensor_like(self, value):
        if value is None:
            return
        # Runtime caches may be a single TT tensor or small containers of TT tensors.
        if isinstance(value, (list, tuple)):
            for item in value:
                self._deallocate_tensor_like(item)
            return
        try:
            if isinstance(value, ttnn.Tensor):
                ttnn.deallocate(value)
        except Exception:
            # Best effort cleanup: stale/already-freed buffers should not break reset.
            pass

    def _clear_cached_attrs(self, obj, attr_names):
        for attr in attr_names:
            value = getattr(obj, attr, None)
            self._deallocate_tensor_like(value)
            setattr(obj, attr, None)

    def _clear_runtime_cache(self):
        for value in self._runtime_cache.values():
            self._deallocate_tensor_like(value)
        self._runtime_cache.clear()

    def _load_from_state_dict(self, state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs):
        self.module = self._create_module(WeightScope.wrap(state_dict).child(prefix[:-1]))

    def _create_module(self, weights: WeightScope):
        raise NotImplementedError

    def reset_static_cache(self):
        """Reset cached static data so it is recomputed on the next forward pass.

        Call between proteins when input dimensions change.
        """
        self._clear_runtime_cache()
        self._first_forward_pass = True


class PairformerModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        affinity: bool = False,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads
        self.transform_s = transform_s
        self.affinity = affinity

    def _create_module(self, weights: WeightScope):
        return Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.transform_s,
            weights,
            self.compute_kernel_config,
            affinity=self.affinity,
        )

    def forward(
        self,
        s: torch.Tensor | None,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        use_kernels: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        seq_len = z.shape[1]
        pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "attn_mask_start_tt", "attn_mask_end_tt")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
            if s is not None:
                s = torch.nn.functional.pad(s, (0, 0, 0, pad))

        # Compute masks (once, reused across forward calls)
        if self._first_forward_pass:
            if self.affinity:
                # Affinity: cross-chain pair_mask, separate start/end additive masks
                if pad:
                    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
                self._cache_set("mask_tt", self._from_torch(pair_mask))
                self._cache_set("attn_mask_start_tt", self._from_torch(pair_mask.permute(1, 0, 2).unsqueeze(2) * 1e9 - 1e9))
                self._cache_set("attn_mask_end_tt", self._from_torch(pair_mask.permute(2, 0, 1).unsqueeze(2) * 1e9 - 1e9))
            elif mask is not None or pad:
                # Non-affinity: 1D mask → additive [1,1,1,S], pair_mask [1,S,S] for TriangleMul
                mask_1d = mask if mask is not None else z.new_ones(1, seq_len)
                if pad:
                    mask_1d = torch.nn.functional.pad(mask_1d, (0, pad))
                    if pair_mask is not None:
                        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
                self._cache_set("mask_tt", self._from_torch(pair_mask if pair_mask is not None else mask_1d))
                attn_mask = self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9)
                self._cache_set("attn_mask_start_tt", attn_mask)
                self._cache_set("attn_mask_end_tt", attn_mask)
            else:
                self._cache_set("mask_tt", None)
                self._cache_set("attn_mask_start_tt", None)
                self._cache_set("attn_mask_end_tt", None)
            self._first_forward_pass = False

        s_out, z_out = self.module(
            self._from_torch(s) if s is not None else None,
            self._from_torch(z),
            self._cache_get("mask_tt"),
            self._cache_get("attn_mask_start_tt"),
            self._cache_get("attn_mask_end_tt"),
        )

        s_result = self._to_torch(s_out)[:, :seq_len, :] if s_out is not None else None
        z_result = self._to_torch(z_out)[:, :seq_len, :seq_len, :]
        return s_result, z_result


class Fp32PairformerModule(TorchWrapper):
    """fp32-on-device pairformer trunk for the affinity model.

    Same call signature as
    PairformerModule; activations and weights stay fp32 end to end, so the z
    that feeds the affinity head carries no bf16 storage rounding. The host
    recycle loop in Boltz2.forward is reused unchanged: s/z arrive as host
    torch fp32, ride the device for the 64 blocks, and return to host fp32.
    """

    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads
        self.transform_s = transform_s

    def _create_module(self, weights: WeightScope):
        return Fp32Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.transform_s,
            weights,
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor | None,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        use_kernels: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        seq_len = z.shape[1]
        pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "tri_attn_mask_tt", "attn_mask_tt")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
            if s is not None:
                s = torch.nn.functional.pad(s, (0, 0, 0, pad))

        if self._first_forward_pass:
            mask_1d = mask if mask is not None else z.new_ones(1, seq_len)
            if pad:
                mask_1d = torch.nn.functional.pad(mask_1d, (0, pad))
                if pair_mask is not None:
                    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
            if pair_mask is None:
                pair_mask = mask_1d[:, :, None] * mask_1d[:, None, :]
            # tri-mul mask [1,S,S,1]; tri-attn additive pair mask [S,1,1,S]
            # (1e9, the reference's inf); s-attn additive key mask [1,1,1,S]
            # (1e6, the reference AttentionPairBias's inf).
            self._cache_set(
                "mask_tt",
                self._from_torch(pair_mask.unsqueeze(-1), ttnn.float32),
            )
            tri_mask = (pair_mask - 1.0) * 1e9  # [1,S,S], 0 on real pairs
            self._cache_set(
                "tri_attn_mask_tt",
                self._from_torch(
                    tri_mask.reshape(seq_len + pad, 1, 1, seq_len + pad), ttnn.float32
                ),
            )
            self._cache_set(
                "attn_mask_tt",
                self._from_torch(
                    (1 - mask_1d).reshape(1, 1, 1, -1) * -1e6, ttnn.float32
                ),
            )
            self._first_forward_pass = False

        s_tt = self._from_torch(s, ttnn.float32) if s is not None else None
        z_tt = self._from_torch(z, ttnn.float32)
        s_out, z_out = self.module(
            s_tt,
            z_tt,
            self._cache_get("mask_tt"),
            self._cache_get("tri_attn_mask_tt"),
            self._cache_get("attn_mask_tt"),
        )
        # The layers' first residual add_ writes into the uploaded tensors, so
        # s_out/z_out alias s_tt/z_tt: deallocate the outputs only.
        z_result = self._to_torch(z_out)[:, :seq_len, :seq_len, :]
        ttnn.deallocate(z_out)
        s_result = None
        if s_out is not None:
            s_result = self._to_torch(s_out)[:, :seq_len, :]
            ttnn.deallocate(s_out)
        return s_result, z_result


class MiniformerModule(TorchWrapper):
    """Public wrapper for BoltzGen's Miniformer (design-stage pairformer).

    Same interface as PairformerModule.forward(s, z, mask, pair_mask, ...) but
    drives the lighter Miniformer stack: one MiniTriangularUpdate per layer
    instead of 4 triangular ops.
    """

    def __init__(self, n_blocks: int, att_head_dim: int, att_n_heads: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads

    def _create_module(self, weights: WeightScope):
        return Miniformer(
            self.n_blocks,
            self.att_head_dim,
            self.att_n_heads,
            weights,
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        use_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = z.shape[1]
        pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "seq_mask_tt")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
            s = torch.nn.functional.pad(s, (0, 0, 0, pad))

        if self._first_forward_pass:
            mask_1d = mask if mask is not None else z.new_ones(1, seq_len)
            if pad:
                mask_1d = torch.nn.functional.pad(mask_1d, (0, pad))
                if pair_mask is not None:
                    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
            # 2D pair-mask if provided, otherwise the 1D token mask (Miniformer
            # masks the bi-directional update by token, not by pair).
            self._cache_set(
                "mask_tt",
                self._from_torch(pair_mask if pair_mask is not None else mask_1d),
            )
            self._cache_set(
                "seq_mask_tt",
                self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9),
            )
            self._first_forward_pass = False

        s_out, z_out = self.module(
            self._from_torch(s),
            self._from_torch(z),
            self._cache_get("mask_tt"),
            self._cache_get("seq_mask_tt"),
        )

        s_result = self._to_torch(s_out)[:, :seq_len, :]
        z_result = self._to_torch(z_out)[:, :seq_len, :seq_len, :]
        return s_result, z_result


class DiffusionModule(TorchWrapper):
    def __init__(self):
        super().__init__()

    def _create_module(self, weights: WeightScope):
        return Diffusion(weights, self.compute_kernel_config)

    def _populate_diffusion_cache(
        self,
        r_batch: int,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        bias_encoder: torch.Tensor,
        bias_token: torch.Tensor,
        bias_decoder: torch.Tensor,
        keys_indexing: torch.Tensor,
        mask: torch.Tensor,
        atom_to_token: torch.Tensor,
    ) -> tuple[int, int, int]:
        """Hoist the per-step-INVARIANT diffusion conditioning onto the device once.

        Everything except ``r`` and ``times`` is constant across all sampling
        steps, so it is uploaded / reshaped / masked once and read from the
        runtime cache every step. Idempotent. Returns ``(seq_len, N, N_padded)``
        so callers can slice the output and pick the chunked kernel path.
        """
        cond_key = s_inputs  # the entry object; padding below replaces the name
        cond_ref = self._cache_get("cond_ref")
        if cond_ref is not None and cond_ref is not cond_key:
            # A new fold's conditioning arrived while the cache still holds the
            # previous fold's. The staged device tensors, and any trace captured
            # over their buffers, are valid only for the conditioning they were
            # staged from; shape alone cannot tell "next step of the same fold"
            # from "first step of a new one", but tensor identity can.
            self.reset_static_cache()
        B, N, _ = q.shape
        NW = N // ATOM_WINDOW

        seq_len = s_inputs.shape[1]
        token_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        padded_seq = seq_len + token_pad
        N_padded = padded_seq * MAX_ATOMS_PER_TOKEN
        if N > N_padded:
            # The protein-derived bucket (Trp=14 atoms/token) under-sizes targets with
            # nucleic-acid tokens (up to 23 atoms) or large modified residues. Extend to
            # the next window multiple that covers the real atom count — the padding is
            # masked out, and the diffusion trace keys on N_padded, so this costs one
            # recompile per new shape rather than a failure. Protein-only inputs never
            # take this branch, so their shapes (and compiled caches) are unchanged.
            N_padded = -(-N // ATOM_WINDOW) * ATOM_WINDOW
        atom_pad = N_padded - N
        NW_padded = N_padded // ATOM_WINDOW
        K_padded = B * NW_padded

        required_cache_keys = (
            "s_inputs",
            "s_trunk",
            "q",
            "c",
            "keys_indexing",
            "bias_encoder",
            "bias_token",
            "bias_decoder",
            "atom_to_token",
            "atom_to_token_normed",
            "atom_pad",
        )
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        # Compute all static data once (everything except r and times is constant across diffusion steps)
        if self._first_forward_pass:
            if token_pad:
                s_inputs = torch.nn.functional.pad(s_inputs, (0, 0, 0, token_pad))
                s_trunk = torch.nn.functional.pad(s_trunk, (0, 0, 0, token_pad))
            self._cache_set("s_inputs", self._from_torch(s_inputs))
            self._cache_set("s_trunk", self._from_torch(s_trunk))

            q_pt = q if r_batch == q.shape[0] else torch.repeat_interleave(q, r_batch, dim=0)
            c_pt = c if r_batch == c.shape[0] else torch.repeat_interleave(c, r_batch, dim=0)
            if atom_pad:
                q_pt = torch.nn.functional.pad(q_pt, (0, 0, 0, atom_pad))
                c_pt = torch.nn.functional.pad(c_pt, (0, 0, 0, atom_pad))
            self._cache_set("q", self._from_torch(q_pt))
            self._cache_set("c", self._from_torch(c_pt))

            if atom_pad:
                ki_pad_rows = 2 * NW_padded - keys_indexing.shape[0]
                ki_pad_cols = 8 * NW_padded - keys_indexing.shape[1]
                keys_indexing = torch.nn.functional.pad(keys_indexing, (0, ki_pad_cols, 0, ki_pad_rows))
            keys_indexing_tt = self._from_torch(keys_indexing, dtype=ttnn.bfloat4_b)
            self._cache_set("keys_indexing", keys_indexing_tt)

            if atom_pad:
                mask = torch.nn.functional.pad(mask, (0, atom_pad))
            mask = self._from_torch(mask)
            mask = ttnn.reshape(mask, (2 * K_padded, ATOM_WINDOW // 2, -1))
            # transpose_a swaps only the last two dims and cannot replace this 3D axis reorder.
            mask = ttnn.permute(mask, (1, 2, 0))
            mask = ttnn.matmul(
                mask,
                keys_indexing_tt,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            mask = ttnn.permute(mask, (2, 0, 1))
            mask = ttnn.reshape(mask, (K_padded, 1, 1, -1))
            # Additive mask: 0 → valid, -1e9 → padded (bfloat16 for -1e9 precision)
            mask = (-1 * mask + 1) * -1e9

            def prepare_atom_bias(bias_pt):
                if atom_pad:
                    bias_pt = torch.nn.functional.pad(bias_pt, (0, 0, 0, 0, 0, 0, 0, NW_padded - NW))
                bias = self._from_torch(bias_pt)
                bias = ttnn.reshape(bias, (B * NW_padded, ATOM_WINDOW, ATOM_DIM, -1))
                bias = ttnn.permute(bias, (0, 3, 1, 2))
                bias = ttnn.add_(bias, mask)
                return ttnn.multiply_(bias, ATOM_WINDOW ** 0.5)

            self._cache_set("bias_encoder", self._hoist_layer_bias(
                prepare_atom_bias(bias_encoder), self.module.encoder))
            self._cache_set("bias_decoder", self._hoist_layer_bias(
                prepare_atom_bias(bias_decoder), self.module.decoder))

            if token_pad:
                bias_token = torch.nn.functional.pad(bias_token, (0, 0, 0, token_pad, 0, token_pad))
            bias = self._from_torch(bias_token)
            bias = ttnn.multiply_(
                bias, (TOKEN_DIM / TOKEN_N_HEADS) ** 0.5
            )
            bias_token_tt = ttnn.permute(bias, (0, 3, 1, 2))
            if token_pad:
                # Fuse additive padding mask into token bias (bfloat16 for -1e9)
                seq_mask = torch.zeros(1, 1, 1, padded_seq)
                seq_mask[..., seq_len:] = -1e9
                bias_token_tt = ttnn.add_(bias_token_tt, self._from_torch(seq_mask))
            self._cache_set("bias_token", self._hoist_layer_bias(
                bias_token_tt,
                None if self.module.token_transformer_fp32 else self.module.token_transformer))

            if atom_pad or token_pad:
                atom_to_token = torch.nn.functional.pad(atom_to_token, (0, token_pad, 0, atom_pad))
            atom_to_token_tt = self._from_torch(atom_to_token)
            self._cache_set("atom_to_token", atom_to_token_tt)
            atom_to_token_normed_tt = ttnn.multiply(
                atom_to_token_tt,
                ttnn.reciprocal(
                    ttnn.sum(atom_to_token_tt, dim=1, keepdim=True) + 1e-6
                ),
            )
            self._cache_set("atom_to_token_normed", atom_to_token_normed_tt)

            self._cache_set("atom_pad", atom_pad)
            self._cache_set("cond_ref", cond_key)
            self._first_forward_pass = False
        return seq_len, N, N_padded

    def _hoist_layer_bias(self, bias: ttnn.Tensor, transformer):
        """L7: cut the per-layer head-ranges once, here, instead of once per denoise step.

        ``DiffusionTransformer.__call__`` slices its layer's head-range out of the shared
        attention bias on every call, and the bias is a rollout invariant. At 512 aa that is
        6000 identical slices per fold, measured in-fold as 449.2 ms of the 8279.3 ms the stage
        spends inside the transformer. Bit-exact: the same deterministic op on the same input.
        Memory-neutral: the parts partition the whole and the source is freed.

        ``transformer=None`` disables the hoist for that bias (the fp32 token path typecasts the
        whole tensor and has no list form).
        """
        if not _B2_BIAS_SLICE_HOIST or transformer is None:
            return bias
        n_layers = len(transformer.layers)
        dim = bias.shape[1] // n_layers
        parts = [bias[:, i * dim : (i + 1) * dim, :, :] for i in range(n_layers)]
        ttnn.deallocate(bias)
        return parts

    def _run_diffusion_device(
        self, r_dev: ttnn.Tensor, times_dev: ttnn.Tensor, large_seq_len: bool
    ) -> ttnn.Tensor:
        """Pure on-device DiT step over the cached conditioning. ``r_dev`` and
        ``times_dev`` are the only per-step-varying inputs; everything else is
        read from the runtime cache populated by ``_populate_diffusion_cache``.
        Returns the ``r_update`` device tensor (not sliced)."""
        return self.module(
            r_dev,
            times_dev,
            self._cache_get("s_inputs"),
            self._cache_get("s_trunk"),
            self._cache_get("q"),
            self._cache_get("c"),
            self._cache_get("bias_encoder"),
            self._cache_get("bias_token"),
            self._cache_get("bias_decoder"),
            self._cache_get("keys_indexing"),
            self._cache_get("atom_to_token"),
            self._cache_get("atom_to_token_normed"),
            large_seq_len=large_seq_len,
        )

    def forward(
        self,
        r: torch.Tensor,
        times: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        bias_encoder: torch.Tensor,
        bias_token: torch.Tensor,
        bias_decoder: torch.Tensor,
        keys_indexing: torch.Tensor,
        mask: torch.Tensor,
        atom_to_token: torch.Tensor,
    ) -> torch.Tensor:
        seq_len, N, _ = self._populate_diffusion_cache(
            r.shape[0], s_inputs, s_trunk, q, c,
            bias_encoder, bias_token, bias_decoder,
            keys_indexing, mask, atom_to_token)
        atom_pad_cached = self._cache_get("atom_pad", 0)
        if atom_pad_cached:
            r = torch.nn.functional.pad(r, (0, 0, 0, atom_pad_cached))
        out = self._run_diffusion_device(
            self._from_torch(r), self._from_torch(times), seq_len > SEQ_LEN_MORE_CHUNKING)
        return self._to_torch(out)[:, :N, :]

    # ---------------------------------------------------------------------------
    # ttnn TRACE of the per-step DiT device stream (opt-in via
    # Boltz.__init__(diffusion_trace=True) / TTScoreModelAdapter.forward(trace=True)). The BoltzGen diffusion loop is
    # shape-stable across all sampling steps (only the scalar ``times`` and the
    # ``r`` coords change; the schedule phases are host-side scalars), so the
    # device graph captured once per design replays every step — collapsing the
    # per-step host dispatch. Mirrors Protenix's DiffusionModule._capture_trace /
    # denoise_traced (tt_bio/protenix.py): stage the two varying inputs into
    # persistent device buffers, warm the lazy caches, begin/end capture, then
    # copy + execute_trace each step. Lossless by construction — the replayed
    # graph is the exact captured program with new input buffer contents, so the
    # output is bit-identical to the untraced ``forward`` (unlike the proven-lossy
    # distinct-structure batching, see the boltzgen-batch-threshold-dead-end memo).
    # ---------------------------------------------------------------------------
    def _host_tt(self, x: torch.Tensor) -> ttnn.Tensor:
        """Host-resident tiled bfloat16 tensor for copy_host_to_device_tensor staging."""
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def _release_trace(self):
        tr = getattr(self, "_diff_trace", None)
        if tr is not None:
            try:
                ttnn.release_trace(self.tt_device, tr["tid"])
            except Exception:
                pass
            self._diff_trace = None

    def _capture_diff_trace(
        self, r_padded: torch.Tensor, times: torch.Tensor,
        large_seq_len: bool, B: int, N_padded: int,
    ) -> dict:
        in_r = self._from_torch(r_padded)        # persistent input buffer
        in_times = self._from_torch(times)
        _ = self._run_diffusion_device(in_r, in_times, large_seq_len)   # warmup / compile
        _ = self._run_diffusion_device(in_r, in_times, large_seq_len)   # 2nd warmup: populate lazy _s_conditioned/_c_reshaped + per-layer s_o
        ttnn.synchronize_device(self.tt_device)
        tid = ttnn.begin_trace_capture(self.tt_device, cq_id=0)
        out = self._run_diffusion_device(in_r, in_times, large_seq_len)  # record
        ttnn.end_trace_capture(self.tt_device, tid, cq_id=0)
        self._diff_trace = {"tid": tid, "in_r": in_r, "in_times": in_times,
                            "out": out, "B": B, "N_padded": N_padded}
        return self._diff_trace

    def forward_traced(
        self,
        r: torch.Tensor,
        times: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        bias_encoder: torch.Tensor,
        bias_token: torch.Tensor,
        bias_decoder: torch.Tensor,
        keys_indexing: torch.Tensor,
        mask: torch.Tensor,
        atom_to_token: torch.Tensor,
    ) -> torch.Tensor:
        """Traced equivalent of ``forward``. Captures the per-step DiT device
        graph once per (B, N_padded) and replays it each step with the new
        ``r`` / ``times`` staged into the captured input buffers. Requires a
        device opened with a trace region (get_device(trace_region_size=1<<30)
        or TT_BIO_TRACE_REGION_SIZE)."""
        import tt_bio.tenstorrent as _TTd
        if _TTd.trace_region_size() <= 0:
            raise ValueError(
                "forward_traced needs a device opened with a trace region; "
                "call get_device(trace_region_size=1 << 30) (or set "
                "TT_BIO_TRACE_REGION_SIZE) before tracing.")
        seq_len, N, N_padded = self._populate_diffusion_cache(
            r.shape[0], s_inputs, s_trunk, q, c,
            bias_encoder, bias_token, bias_decoder,
            keys_indexing, mask, atom_to_token)
        atom_pad_cached = self._cache_get("atom_pad", 0)
        if atom_pad_cached:
            r = torch.nn.functional.pad(r, (0, 0, 0, atom_pad_cached))
        large = seq_len > SEQ_LEN_MORE_CHUNKING
        B = r.shape[0]
        tr = getattr(self, "_diff_trace", None)
        if tr is None or tr["B"] != B or tr["N_padded"] != N_padded:
            if tr is not None:
                self._release_trace()
            tr = self._capture_diff_trace(r, times, large, B, N_padded)
        ttnn.copy_host_to_device_tensor(self._host_tt(r), tr["in_r"])
        ttnn.copy_host_to_device_tensor(self._host_tt(times), tr["in_times"])
        ttnn.execute_trace(self.tt_device, tr["tid"], cq_id=0, blocking=False)
        result = torch.Tensor(ttnn.to_torch(tr["out"])).to(torch.float32)
        return result[:, :N, :]

    def reset_static_cache(self):
        super().reset_static_cache()
        self._release_trace()
        if self.module is not None:
            self._clear_cached_attrs(self.module, ("_s_conditioned", "_c_reshaped"))
            for layer in self.module.encoder.layers + self.module.decoder.layers:
                self._clear_cached_attrs(layer, ("s_o",))
                for adaln in (layer.adaln, layer.transition.adaln):
                    # The pair is owned by the memo; `_s_memo_src` is the caller's `s`
                    # (`_c_reshaped`, freed just above) so it is dropped, never deallocated.
                    self._clear_cached_attrs(adaln, ("_s_memo",))
                    adaln._s_memo_src = None


class MSAModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_head_dim = avg_head_dim
        self.avg_n_heads = avg_n_heads
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads

    def _create_module(self, weights: WeightScope):
        return MSA(
            self.n_blocks,
            self.avg_head_dim,
            self.avg_n_heads,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            weights,
            self.compute_kernel_config,
        )

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        feats: dict[str, torch.Tensor],
        use_kernels: bool = False,
    ) -> torch.Tensor:
        m = torch.cat(
            [
                torch.nn.functional.one_hot(feats["msa"], num_classes=33),
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
                feats["msa_paired"].unsqueeze(-1),
            ],
            dim=-1,
        )

        seq_len = z.shape[1]
        n_msa = m.shape[1]
        seq_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        msa_pad = (-n_msa) % MSA_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "attn_mask_tt", "msa_mask_tt", "n_msa")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if seq_pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, seq_pad, 0, seq_pad))
            emb = torch.nn.functional.pad(emb, (0, 0, 0, seq_pad))
        if seq_pad or msa_pad:
            m = torch.nn.functional.pad(m, (0, 0, 0, seq_pad, 0, msa_pad))

        # Compute masks (once, reused across forward calls)
        if self._first_forward_pass:
            if seq_pad:
                padded_seq = seq_len + seq_pad
                mask_1d = z.new_ones(1, padded_seq)
                mask_1d[:, seq_len:] = 0.0
                # 2D mask for TriangleMultiplication (row + column masking)
                self._cache_set("mask_tt", self._from_torch(mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)))
                # 4D additive mask for TriangleAttention (bfloat16 for -1e9)
                self._cache_set("attn_mask_tt", self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9))
            else:
                self._cache_set("mask_tt", None)
                self._cache_set("attn_mask_tt", None)
            if msa_pad:
                padded_msa = n_msa + msa_pad
                msa_mask = z.new_zeros(padded_msa, 1, 1)
                msa_mask[:n_msa] = 1.0
                self._cache_set("msa_mask_tt", self._from_torch(msa_mask))
                self._cache_set("n_msa", n_msa)
            else:
                self._cache_set("msa_mask_tt", None)
                self._cache_set("n_msa", None)
            self._first_forward_pass = False

        # Hoisted (same left-to-right order as the call that follows) so the uploads can be
        # tagged individually: the MSA one-hot `m` is the trunk's first large device tensor
        # and is what a deep-MSA target actually OOMs on, which is invisible if the whole
        # upload happens inside the argument list. Tags are no-ops unless TT_BIO_DRAM_PEAK.
        dram_peak(f"msamodule pre-upload [m={tuple(m.shape)} {m.dtype}]")
        z_tt = self._from_torch(z)
        dram_peak("msamodule uploaded z")
        m_tt = self._from_torch(m)
        dram_peak(f"msamodule uploaded m [{tuple(m.shape)} {m.dtype}]")
        emb_tt = self._from_torch(emb)
        dram_peak("msamodule uploaded emb")
        z_out = self._to_torch(
            self.module(
                z_tt,
                m_tt,
                emb_tt,
                self._cache_get("mask_tt"),
                self._cache_get("attn_mask_tt"),
                self._cache_get("msa_mask_tt"),
                self._cache_get("n_msa"),
            )
        )

        z_out = z_out[:, :seq_len, :seq_len, :]
        return z_out


class TrunkRecycle:
    """Device-resident trunk recycle glue.

    Computes, entirely on the TT device::

        s = s_init + s_recycle(s_norm(s))
        z = z_init + z_recycle(z_norm(z))

    mirroring the host torch ops in ``Boltz2.forward`` (the recycling loop).
    ``s_norm``/``z_norm`` are ``nn.LayerNorm`` (weight + bias, eps 1e-5);
    ``s_recycle``/``z_recycle`` are ``nn.Linear(.., bias=False)``. The ttnn
    weights are built directly from the already-loaded torch modules so this
    needs no separate state-dict load.
    """

    def __init__(self, s_norm, z_norm, s_recycle, z_recycle, compute_kernel_config):
        self.compute_kernel_config = compute_kernel_config
        device = get_device()

        def w(tensor, transpose=False):
            t = tensor.detach()
            if transpose:
                t = t.t().contiguous()
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.s_norm_weight = w(s_norm.weight)
        self.s_norm_bias = w(s_norm.bias)
        self.z_norm_weight = w(z_norm.weight)
        self.z_norm_bias = w(z_norm.bias)
        # nn.Linear stores weight as [out, in]; ttnn.linear wants [in, out].
        self.s_recycle_weight = w(s_recycle.weight, transpose=True)
        self.z_recycle_weight = w(z_recycle.weight, transpose=True)

    def _branch(self, x, norm_weight, norm_bias, recycle_weight, init):
        x_norm = ttnn.layer_norm(
            x,
            weight=norm_weight,
            bias=norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_rec = ttnn.linear(
            x_norm,
            recycle_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x_norm)
        out = ttnn.add(init, x_rec)
        ttnn.deallocate(x_rec)
        return out

    def __call__(self, s, z, s_init, z_init):
        s_out = self._branch(s, self.s_norm_weight, self.s_norm_bias, self.s_recycle_weight, s_init)
        z_out = self._branch(z, self.z_norm_weight, self.z_norm_bias, self.z_recycle_weight, z_init)
        return s_out, z_out


class TemplateRecycle:
    """Device-resident template injection for the Boltz-2 trunk.

    Mirrors ``TemplateV2Module.forward`` but runs the per-recycling-iteration,
    z-dependent ops fully on the TT device, reusing the template module's inner
    ttnn ``Pairformer``. The z-INDEPENDENT template geometry (``a_tij``) is
    constant across iterations, so it is computed once on host (``precompute``),
    padded to the trunk's padded seq length, and uploaded. This removes the
    per-iteration host round-trip the torch template module would otherwise incur.

    Per call:  u = u_proj(relu( sum_t v_t / num_templates ))
      where    v_t = v_norm( w_t + pairformer(w_t) ),  w_t = z_proj(z_norm(z)) + a_tij[t]
    """

    def __init__(self, template_module, compute_kernel_config):
        self.tmpl = template_module                          # torch module (template_features)
        self.pairformer = template_module.pairformer.module  # inner device-resident pairformer
        self.compute_kernel_config = compute_kernel_config
        device = get_device()

        def w(t, transpose=False):
            t = t.detach()
            if transpose:
                t = t.t().contiguous()
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.z_norm_w = w(template_module.z_norm.weight)
        self.z_norm_b = w(template_module.z_norm.bias)
        self.v_norm_w = w(template_module.v_norm.weight)
        self.v_norm_b = w(template_module.v_norm.bias)
        # nn.Linear weight is [out, in]; ttnn.linear wants [in, out].
        self.z_proj_w = w(template_module.z_proj.weight, transpose=True)  # token_z -> template_dim
        self.u_proj_w = w(template_module.u_proj.weight, transpose=True)  # template_dim -> token_z

    def precompute(self, feats, pair_mask_unpad, seq_len, seq_pad):
        """Host once-per-protein: a_tij (padded, uploaded per present template) plus
        the padding-only masks the template pairformer uses (it is called mask-free)."""
        device = get_device()
        a_tij, template_mask, num_templates, _, _, T = self.tmpl.template_features(
            feats, pair_mask_unpad
        )
        if seq_pad:
            a_tij = torch.nn.functional.pad(a_tij, (0, 0, 0, seq_pad, 0, seq_pad))
        present = [t for t in range(T) if bool(template_mask[0, t] > 0)]
        a_tij_tt = [
            ttnn.from_torch(a_tij[:, t].contiguous(), layout=ttnn.TILE_LAYOUT,
                            device=device, dtype=ttnn.bfloat16)
            for t in present
        ]
        # template pairformer is called without a mask -> padding-only masks (mirror
        # PairformerModule.forward's no-mask branch); None when no padding.
        if seq_pad:
            mask_1d = a_tij.new_ones(1, seq_len + seq_pad)
            mask_1d[:, seq_len:] = 0.0
            mask_tt = ttnn.from_torch(mask_1d, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            attn_tt = ttnn.from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9,
                                      layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        else:
            mask_tt = attn_tt = None
        return {"a_tij_tt": a_tij_tt, "num_templates": float(num_templates[0]),
                "mask_tt": mask_tt, "attn_tt": attn_tt}

    def __call__(self, z, tmpl):
        """z [1,P,P,token_z] -> template delta u [1,P,P,token_z], fully on device."""
        ckc = self.compute_kernel_config
        z_n = ttnn.layer_norm(z, weight=self.z_norm_w, bias=self.z_norm_b,
                              epsilon=1e-5, compute_kernel_config=ckc)
        z_p = ttnn.linear(z_n, self.z_proj_w, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(z_n)
        mask_tt, attn_tt = tmpl["mask_tt"], tmpl["attn_tt"]
        u_acc = None
        for a_tij_tt in tmpl["a_tij_tt"]:
            v = ttnn.add(z_p, a_tij_tt)
            _, z_out = self.pairformer(None, v, mask_tt, attn_tt, attn_tt)
            v2 = ttnn.add(v, z_out)
            ttnn.deallocate(v)
            ttnn.deallocate(z_out)
            v2 = ttnn.layer_norm(v2, weight=self.v_norm_w, bias=self.v_norm_b,
                                 epsilon=1e-5, compute_kernel_config=ckc)
            if u_acc is None:
                u_acc = v2
            else:
                new = ttnn.add(u_acc, v2)
                ttnn.deallocate(u_acc)
                ttnn.deallocate(v2)
                u_acc = new
        ttnn.deallocate(z_p)
        u = ttnn.multiply(u_acc, 1.0 / tmpl["num_templates"])
        ttnn.deallocate(u_acc)
        u = ttnn.relu(u)
        u = ttnn.linear(u, self.u_proj_w, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        return u


class TokenDistanceRecycle:
    """Device-resident token-distance injection for BoltzGen's trunk.

    Mirrors ``TokenDistanceModule.forward`` (``tt_bio/boltzgen/model/modules/trunk.py``).
    ``a_ij`` (the distogram + relative-position distance features, projected by
    ``a_proj``) depends only on static per-protein geometry (``center_coords``,
    ``relative_position_encoding``), not the recycled ``z`` -- exactly like
    ``TemplateRecycle``'s ``a_tij``. It is computed once on host (``precompute``)
    and the per-iteration z-dependent path (``z_proj``/pairformer/``v_norm``/
    ``u_proj``) runs fully on device, reusing the module's inner ttnn Pairformer.

    Per call:  u = u_proj(relu( v_norm( v + pairformer(v) ) )),  v = z_proj(z_norm(z)) + a_ij
    """

    def __init__(self, token_distance_module, compute_kernel_config):
        self.mod = token_distance_module
        self.pairformer = token_distance_module.pairformer.module
        self.compute_kernel_config = compute_kernel_config
        device = get_device()

        def w(t, transpose=False):
            t = t.detach()
            if transpose:
                t = t.t().contiguous()
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.z_norm_w = w(token_distance_module.z_norm.weight)
        self.z_norm_b = w(token_distance_module.z_norm.bias)
        self.v_norm_w = w(token_distance_module.v_norm.weight)
        self.v_norm_b = w(token_distance_module.v_norm.bias)
        self.z_proj_w = w(token_distance_module.z_proj.weight, transpose=True)
        self.u_proj_w = w(token_distance_module.u_proj.weight, transpose=True)

    def precompute(self, feats, relative_position_encoding, seq_len, seq_pad):
        """Host once-per-protein: a_ij = a_proj(distance features), padded + uploaded,
        plus the padding-only mask/attn-bias its inner pairformer uses (mirrors
        TemplateRecycle.precompute: called with mask=None, so only the tile-padding
        is masked, not the real per-token mask)."""
        device = get_device()
        mod = self.mod
        token_distance_mask = feats["token_distance_mask"]
        token_coords = feats["center_coords"]
        with torch.autocast(device_type="cuda", enabled=False):
            dists = torch.cdist(token_coords, token_coords)
            boundaries = torch.linspace(mod.min_dist, mod.max_dist, mod.num_bins - 1).to(dists.device)
            distogram = (dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = torch.nn.functional.one_hot(distogram, num_classes=mod.num_bins)
            if mod.use_token_distance_feats:
                dist_features = mod.token_distance_encoder(relative_position_encoding, feats)
                a_ij = torch.cat([distogram, dist_features], dim=-1)
            else:
                a_ij = distogram
            a_ij = a_ij * token_distance_mask.unsqueeze(-1)
            a_ij = mod.a_proj(a_ij.float())
        if seq_pad:
            a_ij = torch.nn.functional.pad(a_ij, (0, 0, 0, seq_pad, 0, seq_pad))
        a_ij_tt = ttnn.from_torch(a_ij, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        if seq_pad:
            mask_1d = a_ij.new_ones(1, seq_len + seq_pad)
            mask_1d[:, seq_len:] = 0.0
            mask_tt = ttnn.from_torch(mask_1d, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            attn_tt = ttnn.from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9,
                                      layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        else:
            mask_tt = attn_tt = None
        return {"a_ij_tt": a_ij_tt, "mask_tt": mask_tt, "attn_tt": attn_tt}

    def __call__(self, z, td):
        """z [1,P,P,token_z] -> token-distance delta u [1,P,P,token_z], fully on device."""
        ckc = self.compute_kernel_config
        z_n = ttnn.layer_norm(z, weight=self.z_norm_w, bias=self.z_norm_b,
                              epsilon=1e-5, compute_kernel_config=ckc)
        z_p = ttnn.linear(z_n, self.z_proj_w, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(z_n)
        v = ttnn.add(z_p, td["a_ij_tt"])
        ttnn.deallocate(z_p)
        _, v_pf = self.pairformer(None, v, td["mask_tt"], td["attn_tt"], td["attn_tt"])
        v2 = ttnn.add(v, v_pf)
        ttnn.deallocate(v)
        ttnn.deallocate(v_pf)
        v2 = ttnn.layer_norm(v2, weight=self.v_norm_w, bias=self.v_norm_b,
                             epsilon=1e-5, compute_kernel_config=ckc)
        u = ttnn.relu(v2)
        ttnn.deallocate(v2)
        u = ttnn.linear(u, self.u_proj_w, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        return u


class TrunkModule(TorchWrapper):
    """Device-resident Boltz2 trunk (recycling) loop.

    Replaces the host-side recycling loop in ``Boltz2.forward`` for the simplest
    case (no templates). The whole loop runs on the TT device: ``s``/``z`` are
    uploaded once as zeros, all per-protein constants (``s_init``/``z_init``/
    ``s_inputs``, the MSA feature tensor and every mask) are built on host once
    and uploaded once, and only the final ``s``/``z`` come back to torch. This
    removes the per-iteration host round-trips (4x from_torch/to_torch of the
    full padded z) that previously defeated on-device residency.

    It reuses the *inner* (already device-resident) ``MSA`` and ``Pairformer``
    modules owned by the existing ``MSAModule`` / ``PairformerModule`` wrappers,
    plus a ``TrunkRecycle`` for the glue. The mask / MSA-feature construction
    below mirrors ``MSAModule.forward`` and ``PairformerModule.forward`` exactly.
    """

    def __init__(self, recycle: TrunkRecycle, msa_inner: "MSA", pairformer_inner: "Pairformer",
                 template_recycle: "TemplateRecycle" = None,
                 token_distance_recycle: "TokenDistanceRecycle" = None,
                 template_module_torch=None, use_kernels: bool = False):
        super().__init__()
        self.recycle = recycle
        self.msa = msa_inner
        self.pairformer = pairformer_inner
        # Optional device-resident template injection. When set AND the input carries
        # templates, z = z + template(z) runs fully on device each recycling iteration
        # (no host round-trip), reusing the template's inner ttnn Pairformer; the
        # z-independent a_tij geometry is hoisted (computed once). See TemplateRecycle.
        self.template_recycle = template_recycle
        # Optional device-resident token-distance injection (BoltzGen only): when
        # set, z = z + token_distance(z) runs fully on device each recycling
        # iteration, reusing the module's inner ttnn Pairformer; the z-independent
        # a_ij geometry is hoisted (computed once). See TokenDistanceRecycle.
        self.token_distance_recycle = token_distance_recycle
        # BoltzGen's TemplateModule computes its per-template geometry (frame
        # rotation/translation, visibility, CB/CA distances) inline rather than
        # through a factored template_features() helper like Boltz2's
        # TemplateV2Module, and is called unconditionally every iteration
        # (no has_templates gate) -- porting it to a device-resident recycle is
        # out of scope for this pass. When set, it is called as-is (unchanged,
        # torch/ttnn hybrid module) via one host round-trip per iteration,
        # bracketed by the fully-resident recycle/token-distance/msa/pairformer
        # path. This still collapses 4 host<->device crossings/iteration to 2.
        self.template_module_torch = template_module_torch
        self.use_kernels = use_kernels

    def _build_static(self, s_inputs, s_init, z_init, feats, relative_position_encoding=None):
        """Build + upload (once per protein) all loop-invariant device tensors.

        Returns a dict cached in ``self._runtime_cache`` and reused across the
        recycling iterations.
        """
        seq_len = z_init.shape[1]
        seq_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        padded_seq = seq_len + seq_pad

        # ---- MSA feature tensor (host), mirrors MSAModule.forward ----
        m = torch.cat(
            [
                torch.nn.functional.one_hot(feats["msa"], num_classes=33),
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
                feats["msa_paired"].unsqueeze(-1),
            ],
            dim=-1,
        )
        n_msa = m.shape[1]
        msa_pad = (-n_msa) % MSA_PAD_MULTIPLE

        # ---- pad the per-protein constants ----
        pad = torch.nn.functional.pad
        s_init_p = pad(s_init, (0, 0, 0, seq_pad)) if seq_pad else s_init
        z_init_p = pad(z_init, (0, 0, 0, seq_pad, 0, seq_pad)) if seq_pad else z_init
        s_inputs_p = pad(s_inputs, (0, 0, 0, seq_pad)) if seq_pad else s_inputs
        m_p = pad(m, (0, 0, 0, seq_pad, 0, msa_pad)) if (seq_pad or msa_pad) else m

        # ---- Pairformer masks (mirror PairformerModule.forward, non-affinity) ----
        token_mask = feats["token_pad_mask"].float()
        pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
        pair_mask_unpad = pair_mask  # unpadded [B, seq_len, seq_len] for the template module
        mask_1d_pf = token_mask
        if seq_pad:
            mask_1d_pf = pad(mask_1d_pf, (0, seq_pad))
            pair_mask = pad(pair_mask, (0, seq_pad, 0, seq_pad))
        pf_mask_tt = self._from_torch(pair_mask)
        pf_attn_tt = self._from_torch((1 - mask_1d_pf).unsqueeze(1).unsqueeze(1) * -1e9)

        # ---- MSA masks (mirror MSAModule.forward: derived from padding only) ----
        if seq_pad:
            mask_1d_msa = z_init.new_ones(1, padded_seq)
            mask_1d_msa[:, seq_len:] = 0.0
            msa_mask_tt = self._from_torch(mask_1d_msa.unsqueeze(-1) * mask_1d_msa.unsqueeze(1))
            msa_attn_tt = self._from_torch((1 - mask_1d_msa).unsqueeze(1).unsqueeze(1) * -1e9)
        else:
            msa_mask_tt = None
            msa_attn_tt = None
        if msa_pad:
            padded_msa = n_msa + msa_pad
            msa_row = z_init.new_zeros(padded_msa, 1, 1)
            msa_row[:n_msa] = 1.0
            msa_rowmask_tt = self._from_torch(msa_row)
            n_msa_arg = n_msa
        else:
            msa_rowmask_tt = None
            n_msa_arg = None

        # ---- templates (device-resident injection, only if input carries them) ----
        tm = feats.get("template_mask")
        has_templates = (
            self.template_recycle is not None
            and tm is not None
            and bool(tm.any().item())
        )
        tmpl_static = (
            self.template_recycle.precompute(feats, pair_mask_unpad, seq_len, seq_pad)
            if has_templates else None
        )

        # ---- the template module reduces to zero when the input carries no template ----
        # BoltzGen's TemplateModule ends in
        #     u = (v * template_mask).sum(dim=1) / num_templates ; u = u_proj(relu(u))
        # (trunk.py:376-381) and `u_proj` has `bias=False`, so an all-zero `template_mask` gives a
        # structurally exact zero delta -- proven from the source, not observed on one fixture, and
        # MEASURED absmax 0.0 on 4/4 recycles at bg_R3. `_apply_template_host` then reduces to its
        # only other effect: it slices the pair tensor to seq_len and re-pads with zeros, and the
        # padded region really is dirty when it arrives (8.65e6 non-zero elements, absmax 17.25
        # MEASURED at bg_R3), so the call is NOT a no-op and cannot simply be skipped. Multiplying
        # by a keep mask on device reproduces it exactly and costs one program.
        tmpl_noop = (
            self.template_module_torch is not None
            and not has_templates
            and tm is not None
            and not bool(tm.any().item())
        )
        tmpl_keep_mask = None
        if tmpl_noop and seq_pad:
            keep = z_init.new_zeros(1, padded_seq, padded_seq, 1)
            keep[:, :seq_len, :seq_len, :] = 1.0
            tmpl_keep_mask = self._from_torch(keep)

        # ---- token distances (BoltzGen only, device-resident injection) ----
        has_token_distance = self.token_distance_recycle is not None
        token_distance_static = (
            self.token_distance_recycle.precompute(
                feats, relative_position_encoding, seq_len, seq_pad
            )
            if has_token_distance else None
        )

        static = {
            "seq_len": seq_len,
            "seq_pad": seq_pad,
            "has_templates": has_templates,
            "tmpl_static": tmpl_static,
            "tmpl_noop": tmpl_noop,
            "tmpl_keep_mask": tmpl_keep_mask,
            "has_token_distance": has_token_distance,
            "token_distance_static": token_distance_static,
            "feats": feats,
            "pair_mask_unpad": pair_mask_unpad,
            "s_init_tt": self._from_torch(s_init_p),
            "z_init_tt": self._from_torch(z_init_p),
            "emb_tt": self._from_torch(s_inputs_p),
            "m_tt": self._from_torch(m_p),
            "pf_mask_tt": pf_mask_tt,
            "pf_attn_tt": pf_attn_tt,
            "msa_mask_tt": msa_mask_tt,
            "msa_attn_tt": msa_attn_tt,
            "msa_rowmask_tt": msa_rowmask_tt,
            "n_msa_arg": n_msa_arg,
        }
        for k, v in static.items():
            self._cache_set(k, v)
        return static

    def _apply_template(self, z_rec, st):
        """z_rec = z_rec + template(z_rec), fully on device (no host round-trip)."""
        delta = self.template_recycle(z_rec, st["tmpl_static"])
        z_out = ttnn.add(z_rec, delta)
        ttnn.deallocate(z_rec)
        ttnn.deallocate(delta)
        return z_out

    def _apply_template_noop(self, z_rec, st):
        """`z_rec = z_rec + template(z_rec)` when template(.) is provably zero: re-zero the pad.

        MEASURED at bg_R3 on qb1 card 1, ttnn 0.67.4: the host path costs 3.455 s/design over 4
        recycles (85 MB down, a full host TemplateModule forward, 85 MB up); this costs 513.89 us
        per call, and `torch.equal` holds against the sliced-and-re-padded host result.
        """
        mask = st["tmpl_keep_mask"]
        if mask is None:                       # seq_pad == 0: there is no pad to re-zero
            return z_rec
        z_out = ttnn.multiply(z_rec, mask)
        ttnn.deallocate(z_rec)
        return z_out

    def _apply_token_distance(self, z_rec, st):
        """z_rec = z_rec + token_distance(z_rec), fully on device (no host round-trip)."""
        delta = self.token_distance_recycle(z_rec, st["token_distance_static"])
        z_out = ttnn.add(z_rec, delta)
        ttnn.deallocate(z_rec)
        ttnn.deallocate(delta)
        return z_out

    def _apply_template_host(self, z_rec, st):
        """z_rec = z_rec + template_module(z_rec) via one host round-trip.

        Calls BoltzGen's original torch/ttnn-hybrid TemplateModule unchanged
        (see class docstring) -- unlike the other trunk sub-modules this one
        is not resident, but it still collapses what would otherwise be a
        separate round-trip into a single down/up pair.
        """
        seq_len, seq_pad = st["seq_len"], st["seq_pad"]
        z_torch = self._to_torch(z_rec)[:, :seq_len, :seq_len, :]
        with torch.no_grad():
            delta = self.template_module_torch(
                z_torch, st["feats"], st["pair_mask_unpad"], use_kernels=self.use_kernels
            )
        z_new = z_torch + delta
        if seq_pad:
            z_new = torch.nn.functional.pad(z_new, (0, 0, 0, seq_pad, 0, seq_pad))
        z_out = self._from_torch(z_new)
        ttnn.deallocate(z_rec)
        return z_out

    def _iteration(self, s, z, st):
        """Run one recycling iteration fully on device; returns (s, z)."""
        # s = s_init + s_recycle(s_norm(s)); z = z_init + z_recycle(z_norm(z))
        s_rec, z_rec = self.recycle(s, z, st["s_init_tt"], st["z_init_tt"])
        ttnn.deallocate(s)
        ttnn.deallocate(z)

        # token distances (BoltzGen only, before templates, mirrors host):
        # z_rec = z_rec + token_distance_module(z_rec)
        if st["has_token_distance"]:
            z_rec = self._apply_token_distance(z_rec, st)

        # templates (before MSA, mirrors host): z_rec = z_rec + template_module(z_rec)
        if st["has_templates"]:
            z_rec = self._apply_template(z_rec, st)
        elif self.template_module_torch is not None:
            z_rec = (self._apply_template_noop(z_rec, st)
                     if (st["tmpl_noop"] and _TEMPLATE_NOOP_GATE)
                     else self._apply_template_host(z_rec, st))

        # z = z + msa(z). The inner MSA mutates its z argument in place, so clone
        # z_rec first to preserve it for the residual add (matches the wrapper,
        # which passes a fresh upload each call).
        z_for_msa = ttnn.clone(z_rec)
        z_msa = self.msa(
            z_for_msa,
            st["m_tt"],
            st["emb_tt"],
            st["msa_mask_tt"],
            st["msa_attn_tt"],
            st["msa_rowmask_tt"],
            st["n_msa_arg"],
        )
        z = ttnn.add(z_rec, z_msa)
        ttnn.deallocate(z_rec)
        ttnn.deallocate(z_msa)

        # s, z = pairformer(s, z) -- inner mutates s_rec / z in place and returns them.
        s, z = self.pairformer(s_rec, z, st["pf_mask_tt"], st["pf_attn_tt"], st["pf_attn_tt"])
        return s, z

    def forward(self, s_inputs, s_init, z_init, feats, recycling_steps,
                relative_position_encoding=None, progress_fn=None):
        st = self._build_static(s_inputs, s_init, z_init, feats, relative_position_encoding)
        seq_len = st["seq_len"]

        s = self._from_torch(torch.zeros(list(st["s_init_tt"].shape), dtype=s_init.dtype))
        z = self._from_torch(torch.zeros(list(st["z_init_tt"].shape), dtype=z_init.dtype))
        # Tick the live view once per recycling iteration, mirroring the host
        # fallback loop in Boltz2.forward — the resident loop runs entirely on
        # device, so without this the bar sits at "Trunk 0/N" then jumps to the
        # final iteration when the loop returns.
        for _cyc in range(recycling_steps + 1):
            if progress_fn:
                progress_fn("trunk", step=_cyc, total=recycling_steps + 1)
            s, z = self._iteration(s, z, st)

        s_out = self._to_torch(s)[:, :seq_len, :]
        z_out = self._to_torch(z)[:, :seq_len, :seq_len, :]
        ttnn.deallocate(s)
        ttnn.deallocate(z)
        return s_out, z_out
