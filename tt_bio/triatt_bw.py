"""Triangle attention's backward, with the score block never leaving L1.

`autograd.triangle_attention`'s backward recomputes one score block per chunk and frees it, which
keeps the 191.1 MB score tensor out of DRAM as a *resident* object but still moves it through DRAM
roughly 48 times per call: the matmul writes it, the scale reads and writes it, the bias add reads
and writes it, the softmax reads and writes it, and then the whole softmax backward reads and
writes it again. Measured at the shipped BindCraft 2 shape by `bcx-p10-bytes`, that is 962.6 GB a
round, 49.4 % of everything the round moves, on an op family already running at 84.5 % of the
card's DRAM roof. There is no per-byte speed left to find there; the only lever is fewer bytes.

The shape is what makes fewer bytes possible. Triangle attention's leading axis is the residue
axis itself, so a `[S, S, c]` pair tensor is S independent attention problems of length S, and at
the shipped 288-token axis one problem is

    q = k = v = dO   [288, 32] bf16    18.4 KB        scores   [288, 288] bf16   165.9 KB

with four heads. **N/d = 9**, so the scores are nine times q, and one whole problem plus its
gradients fits in a single core's L1 with room left over. A core here takes one head and a
contiguous group of the leading axis, forms S, P and dS in L1, and never writes any of them. What
crosses DRAM is q, k, v, dO in and dq, dk, dv out, which is 9x less than the score tensor alone.

THE BIAS GRADIENT IS WHY THIS CANNOT BE TWO OPS. The bias is `[1, H, S, S]` broadcast over the
leading axis, so `dbias` is the score gradient summed over that axis -- and dS exists only inside
this kernel. Splitting the gradient computation from the bias reduction would mean materialising
dS, which is the thing being avoided. So each core carries a float32 accumulator in its own L1
for its own group of the leading axis, writes it once at the end, and a single `ttnn.sum` over the
group axis finishes it. Float32 and not bf16: the accumulator sums the core's rows and the reduce
sums the partials on top of that, and a bf16 accumulator would put a rounding floor under the one
gradient this kernel produces by reduction rather than by matmul.

The key axis is never chunked, which is what keeps log-sum-exp bookkeeping out of this file. A
flash kernel chunks keys, so every block sees a partial softmax denominator and has to carry
running row statistics and rescale. Here each core holds every key for its rows, so each softmax
is exact and complete the first time.

THE QUERY AXIS IS CHUNKED, OUTSIDE THE LEADING AXIS, when the whole query does not fit. Whole,
`bias` and the `dbias` accumulator are `[Nt, Nt]` each and alone cross the L1 budget at 512 tokens.
With a chunk of `Qt` query tiles the core fronts `bias[h, chunk]` and seeds `dbias[chunk]`, both
`[Qt, Nt]`, and streams every row of its group through them before moving to the next chunk.
`dbias` rows are indexed by the query row, so they need no cross-chunk reduction. dK and dV do:
they sum over query rows, so with more than one chunk they are float32 running sums in DRAM, each
chunk adding to what the previous one wrote for the same row (`SEM_WRITTEN` orders the read after
the write), and one cast to bf16 at the end. `serving_plan` takes the whole query when it fits and
otherwise the largest chunk that divides `Nt` and fits, which serves every bucket from 288 to 1024
tokens. At a shape where the whole query fits, the program is the pre-loop one, bit for bit.

WORMHOLE SERVES THE WHOLE-QUERY FORM ONLY. The chunked form has been graded against float64 on
Blackhole and has not run on a Wormhole chip, so `serving_plan` declines a chunk there and the
caller keeps the chunked-recompute fallback. Lifting that is deleting one line in `serving_plan`,
after `perf/bcw_dbias/grade.py` has run on Wormhole at 288, 544 and 768.

Driven through `tt_bio.sdpa_generic`'s machinery rather than the wheel's SDPA kernels: the forward
inherits `reader_interleaved.cpp`'s chain-forwarding, multicast, paging and MLA arguments, all of
which are dead at this call, and a backward built on top of them would be harder to read than
three small kernels of its own. What is inherited is the part that earned its place -- the CB
table, the work split, and the forward's own proof that a head's bias can sit permanently fronted
in L1 while the batch streams past it.

Default OFF behind `TT_BIO_TRIATT_BW_FUSED`. Gated in `tt_bio/autograd.py::triangle_attention`,
which keeps its chunked-recompute backward as the fallback for every shape this refuses.

REACHING IT IS A SEPARATE SWITCH, and on its own the flag above measures nothing. This hooks
`autograd.triangle_attention`'s backward, and a taped AF2 round does not call that function:
`tenstorrent.py` computes triangle attention through `_fp32_softmax_attention`. Something has to
put the backward there first, and a per-kernel tape entry for the fused HiFi forward
(`TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi`, see `taped_ttnn._k_tri_att_sdpa_hifi`) does it. With
that route open the kernel serves 108 calls a BindCraft 2 round; without it, zero. `STATS` is
what says which: `bw_calls` counts every entry into the backward before any gate, so `bw_calls`
at zero is a routing problem and `declined` above zero is a shape gate.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import ttnn

from . import sdpa_generic as SG
from .envflags import env_flag, env_int

TILE = 32

# CB bytes one core may hold: its L1 less what tt-metal reserves below the CBs. Blackhole's figure
# is `sdpa_generic`'s, exact on ten measured refusals. Wormhole has 1464 KiB of L1, not 1536, and
# its L1 bank is 1395424 B (a Galaxy's own allocator report, `perf/ceilrfd3/results`). Priced
# against Blackhole's figure, a Wormhole plan at 480 tokens takes a chunk of 5 query tiles that
# tt-metal refuses at program creation, inside a backward, with no fallback; against its own it
# takes 3.
L1_PER_CORE = getattr(SG, "L1_PER_CORE", 1572864)
PROGRAM_RESERVE = getattr(SG, "PROGRAM_RESERVE", 109056)
WH_CB_BUDGET = 1395424


def _wormhole() -> bool:
    from . import tenstorrent
    return tenstorrent.is_wormhole()


def cb_budget(wormhole=None) -> int:
    wormhole = _wormhole() if wormhole is None else wormhole
    return WH_CB_BUDGET if wormhole else L1_PER_CORE - PROGRAM_RESERVE

FUSED = env_flag("TT_BIO_TRIATT_BW_FUSED", False)

STATS = {"served": 0, "declined": 0}


def _kdir() -> Path:
    return Path(__file__).parent / "kernels" / "triatt_bw"


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# Tiles the destination register holds. HALF the usual eight, because this kernel runs with
# fp32_dest_acc_en on, and a subblock sized against eight silently overflows DST and returns
# garbage rather than refusing. It bites unevenly, which is what made it hard to see: at
# Nt=2 and Nt=9 the subblock search picks the same answer either way, so the small test shape
# and the shipped 288 were both right by luck, while Nt=4 chose an 8-tile subblock and read
# 275x off the float64 reference.
DST_TILES_FP32_ACC = 4


def plan(B: int, H: int, N: int, d: int, grid, q_chunk_tiles=None,
         dst_size=DST_TILES_FP32_ACC):
    """Work split and tile geometry. Card-free, so the gate can be tested without a device.

    One core owns one head and a contiguous group of the leading axis. Heads go to the FAST index
    so that the cores sharing a head are adjacent, which is what lets the bias for that head be
    read once per core rather than once per leading-axis row.
    """
    if N % TILE or d % TILE:
        raise ValueError(f"token axis {N} and head dim {d} must both be tile aligned")
    Nt, Dt = N // TILE, d // TILE
    Qt = Nt if q_chunk_tiles is None else min(int(q_chunk_tiles), Nt)
    while Qt > 1 and Nt % Qt:
        Qt -= 1

    gx, gy = grid
    # Never more groups than there are leading-axis rows: a core with no row still allocates its
    # dbias accumulator and still contributes a partial to the reduce, so an empty one is pure
    # cost in both L1 and DRAM.
    groups = max(1, min(B, (gx * gy) // H))
    num_cores = groups * H
    rows_per_core = _div_up(B, groups)

    p = {
        "B": B, "H": H, "N": N, "d": d, "Nt": Nt, "Dt": Dt, "Qt": Qt,
        "groups": groups, "num_cores": num_cores, "rows_per_core": rows_per_core,
        "gx": gx, "gy": gy, "dst_size": dst_size,
    }
    # Priced off the CB table itself rather than a parallel formula. The two drifting apart is how
    # a gate ends up reporting that a config fits while tt-metal refuses it.
    p["l1_bytes"] = cb_bytes(p)
    return p


def fits_l1(p, wormhole=None) -> bool:
    """Whether tt-metal will accept this config's CBs on this board."""
    return p["l1_bytes"] <= cb_budget(wormhole)


def largest_fitting_q_chunk(B, H, N, d, grid, wormhole=None, **kw):
    """The biggest query chunk that divides `Nt` and fits, or None if one tile row does not."""
    Nt = N // TILE
    for qt in range(Nt, 0, -1):
        if Nt % qt:
            continue
        p = plan(B, H, N, d, grid, q_chunk_tiles=qt, **kw)
        if fits_l1(p, wormhole):
            return p
    return None


def serving_plan(B, H, N, d, grid, wormhole=None, **kw):
    """The plan `run` uses: the whole-query form when it fits, else the largest query chunk.

    `wormhole` defaults to the board this process sees; tests pass it to price the other one.
    """
    wormhole = _wormhole() if wormhole is None else wormhole
    p = plan(B, H, N, d, grid, **kw)
    if fits_l1(p, wormhole):
        return p
    if wormhole:
        return None   # the chunked form is ungraded on Wormhole (module docstring)
    return largest_fitting_q_chunk(B, H, N, d, grid, wormhole, **kw)


def eligible(q, k, v, bias, *, taping=False):
    """Whether this call can be served, with the reason recorded rather than guessed.

    Deliberately narrow. Everything it refuses falls through to the chunked-recompute backward,
    which is correct at every shape; there is no configuration where refusing costs a wrong answer.
    """
    if not FUSED:
        return False, "flag off"
    if bias is None:
        return False, "no bias: the chunked path is already cheap without a dbias reduction"
    qs = [int(x) for x in q.shape]
    if len(qs) != 4:
        return False, f"expected [B, H, N, d], got {qs}"
    B, H, N, d = qs
    if [int(x) for x in k.shape] != qs or [int(x) for x in v.shape] != qs:
        return False, "q, k and v must share a shape; cross attention is not this op"
    bs = [int(x) for x in bias.shape]
    if bs != [1, H, N, N]:
        return False, f"bias must be [1, {H}, {N}, {N}] broadcast over the batch, got {bs}"
    if N % TILE or d % TILE:
        return False, f"ragged: N={N} d={d} are not both tile aligned"
    if any(t.dtype != ttnn.bfloat16 for t in (q, k, v, bias)):
        return False, "bf16 operands only"
    return True, "ok"


# CB indices. Laid out so the three score-sized buffers are adjacent and easy to price.
CB_Q, CB_K, CB_V, CB_DO = 0, 1, 2, 3
CB_BIAS = 4                      # [Nt, Nt] for one head, fronted once and indexed, never popped
CB_SCALAR = 5                    # the packed bf16 1.0 the row reductions scale by
CB_ZERO = 6                      # one all-zero tile, what the dbias accumulator is seeded from
CB_SCALE = 7                     # the attention scale, as a broadcast scalar
CB_ONES = 8                      # Nt copies of the column identity: every row sum is a matmul
CB_PREV = 9                      # the previous query chunk's float32 dV, then dK, for one row
CB_TMP = 10                      # this chunk's dV or dK before the previous chunk's is added
CB_P = 24                        # S, then P, in place
CB_DP = 25                       # dP, then dS, in place
CB_T = 26                        # transpose scratch: P^T for dV, dS^T for dK
CB_ROW_A, CB_ROW_B = 27, 28      # row max, then row sum, then its reciprocal
CB_DBIAS = 29                    # float32 accumulator, [Nt, Nt], persistent across the whole group
CB_DONE = 30                     # carries no data: the writer's handshake on the dbias accumulator
CB_DQ, CB_DK, CB_DV = 16, 17, 18
SEM_WRITTEN = 0

# There is deliberately no separate output buffer for `dbias`. The accumulator is read and written
# in place across the whole group and then pushed once, so the writer drains the same L1 the
# compute kernel accumulated into. A second [Nt, Nt] float32 buffer would have cost 331.8 KB to
# hold a copy of data that is already sitting there, and at the shipped shape that is the
# difference between fitting in L1 and not.


def cb_table(p):
    """(buffer index, tiles, page bytes, data format), priced the same way `sdpa_generic` prices.

    EVERY index the kernels name has to appear here. An index that does not is not an error
    anywhere: tt-metal gives it zero length, `cb_reserve_back` on it waits for space that can never
    appear, and the core sits in the watcher's `CRBW` state forever with no message. The three
    single-tile buffers -- the zero seed, the scale scalar and the writer's handshake -- were
    missing from this table for exactly that reason, and `check_cb_coverage` now tests it.
    """
    Nt, Dt, Qt = p["Nt"], p["Dt"], p["Qt"]
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    b16, b32 = 2048, 4096
    # With more than one query chunk dK and dV are float32 partials summed across chunks, single
    # buffered to keep L1 for the chunk, and the previous chunk's partial needs somewhere to land.
    # With one chunk the two extra buffers are never touched and hold one tile each.
    chunked = Qt < Nt
    kv = (Nt * Dt, b32, f32) if chunked else (Nt * Dt * 2, b16, bf16)
    side = Nt * Dt if chunked else 1
    return [
        (CB_Q, Qt * Dt * 2, b16, bf16),
        (CB_K, Nt * Dt * 2, b16, bf16),
        (CB_V, Nt * Dt * 2, b16, bf16),
        (CB_DO, Qt * Dt * 2, b16, bf16),
        (CB_BIAS, Qt * Nt, b16, bf16),
        (CB_SCALAR, 1, b16, bf16),
        (CB_ZERO, 1, b16, bf16),
        (CB_SCALE, 1, b16, bf16),
        (CB_ONES, Nt, b16, bf16),
        (CB_PREV, side, b32, f32),
        (CB_TMP, side, b32, f32),
        (CB_DONE, 1, b16, bf16),
        (CB_P, Qt * Nt, b16, bf16),
        (CB_DP, Qt * Nt, b16, bf16),
        (CB_T, Qt * Nt, b16, bf16),
        (CB_ROW_A, Qt, b16, bf16),
        (CB_ROW_B, Qt, b16, bf16),
        (CB_DBIAS, Qt * Nt, b32, f32),
        (CB_DQ, Qt * Dt * 2, b16, bf16),
        (CB_DK, *kv),
        (CB_DV, *kv),
    ]


def cb_bytes(p) -> int:
    return sum(n * page for _i, n, page, _f in cb_table(p))


# Every CB index the three kernels name, kept beside the table it has to agree with.
KERNEL_CB_INDICES = {CB_Q, CB_K, CB_V, CB_DO, CB_BIAS, CB_SCALAR, CB_ZERO, CB_SCALE, CB_ONES,
                     CB_PREV, CB_TMP,
                     CB_P, CB_DP, CB_T, CB_ROW_A, CB_ROW_B, CB_DBIAS, CB_DONE,
                     CB_DQ, CB_DK, CB_DV}


def check_cb_coverage(p):
    """Raise if the kernels name a circular buffer the host never allocated.

    Card-free, and it is here because the failure mode it catches is silent: an undeclared CB is a
    zero-length one, so the first `cb_reserve_back` against it hangs the core with no diagnostic
    short of attaching the watcher.
    """
    declared = {idx for idx, _n, _pg, _f in cb_table(p)}
    missing = sorted(KERNEL_CB_INDICES - declared)
    extra = sorted(declared - KERNEL_CB_INDICES)
    if missing or extra:
        raise ValueError(f"CB table disagrees with the kernels: missing {missing}, extra {extra}")


def partial_shape(p):
    """The shape of the `dbias` partial tensor the kernel writes and `ttnn.sum` reduces.

    One `[H, N, N]` slab per group of the leading axis. `dim=0` of this is the only reduction the
    host does, and it is the whole reason a core can keep its accumulator in L1.
    """
    return [p["groups"], p["H"], p["N"], p["N"]]


def byte_census(p, elem=2, partial_elem=4):
    """What this program moves through DRAM, per invocation, in bytes.

    Same convention as the campaign's census: operand and result bytes on the padded device shape.
    Stated here rather than in a comment so an A/B can assert against it instead of a paragraph.
    """
    B, H, N, d = p["B"], p["H"], p["N"], p["d"]
    qkv = B * H * N * d * elem
    head_bias = H * N * N * elem
    per_core_bias = N * N * elem
    partial = p["groups"] * H * N * N * partial_elem
    return {
        "read_qkvdo": 4 * qkv,
        "read_bias": p["num_cores"] * per_core_bias,
        "write_dqdkdv": 3 * qkv,
        "write_dbias_partials": partial,
        "reduce_read_partials": partial,
        "reduce_write_dbias": head_bias,
        "total": 4 * qkv + p["num_cores"] * per_core_bias + 3 * qkv
                 + 2 * partial + head_bias,
    }


def chunked_recompute_bytes(p, passes=48, elem=2):
    """What the path this replaces moves, for the same invocation: `passes` over the score tensor.

    `passes` is measured, not assumed -- `bcx-p10-bytes` read 962.6 GB a round over 104
    invocations at the shipped shape, which is 48.4 score-tensor passes per call.
    """
    return int(passes * p["B"] * p["H"] * p["N"] * p["N"] * elem)


def _bf16_bits(x: float) -> int:
    """The top 16 bits of the float32 pattern, which is what a bf16 tile element holds."""
    return (_f32_bits(x) >> 16) & 0xFFFF


def _packed_bf16(x: float) -> int:
    """`pack_two_bfloat16_into_uint32({x, x})`, the form the bcast-scalar generators take."""
    b = _bf16_bits(x)
    return (b << 16) | b


def work_cores(num_cores: int, gx: int):
    """The CoreRangeSet holding exactly the cores that get runtime arguments.

    This has to match the runtime-arg list EXACTLY, and getting it wrong is silent and
    destructive rather than merely wasteful. A kernel placed on the whole grid runs on every core
    in it; a core with no runtime arguments reads them as zero, so its writer takes address 0 as
    its output base and writes into whatever the allocator put at the bottom of DRAM. On this row
    that was the first tensor uploaded, and it looked for most of a pass like the first DRAM
    buffer being unreadable -- the buffer was fine until the program's own stray cores ran over
    it. Cores are assigned `CoreCoord(i % gx, i // gx)`, so the set is whole rows plus a partial.
    """
    full, rem = divmod(num_cores, gx)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def build(device, q, k, v, do, bias, dq, dk, dv, dbias_partial, p, ckc, scale):
    """The ProgramDescriptor for one triangle-attention backward.

    Deliberately its own program rather than an arm of `sdpa_generic.build`. The forward's reader
    carries chain-forwarding, multicast, paging and MLA arguments that are all dead at this call,
    and the backward's work split is a different shape anyway: it owns a group of the leading axis
    rather than a chunk of the query axis, because that is the axis `dbias` reduces over.
    """
    gx, gy = p["gx"], p["gy"]
    Nt, Dt, H = p["Nt"], p["Dt"], p["H"]
    num_cores = p["num_cores"]
    core_grid = work_cores(num_cores, gx)

    if Nt % p["Qt"]:
        raise ValueError(f"query chunk {p['Qt']} does not divide Nt={Nt}")
    check_cb_coverage(p)
    cbs = [ttnn.CBDescriptor(
        total_size=n * page, core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt,
                                                    page_size=page)])
        for idx, n, page, fmt in cb_table(p)]

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    qkv_tb = SG.tile_bytes(q.dtype)
    bias_tb = SG.tile_bytes(bias.dtype)
    part_tb = SG.tile_bytes(dbias_partial.dtype)

    if os.environ.get("TT_BIO_TRIATT_BW_DUMP"):
        for nm, tn in (("q", q), ("k", k), ("v", v), ("do", do), ("bias", bias),
                       ("dq", dq), ("dk", dk), ("dv", dv), ("part", dbias_partial)):
            print(f"  addr {nm:5s} 0x{tn.buffer_address():x}  acc_len {len(acc(tn))}  "
                  f"dtype {tn.dtype} shape {list(tn.padded_shape)}")
    kv_tb = SG.tile_bytes(dk.dtype)
    reader_ct = ([H, Nt, Dt, qkv_tb, bias_tb, p["Qt"], kv_tb, SEM_WRITTEN]
                 + acc(q) + acc(k) + acc(v) + acc(do) + acc(bias) + acc(dk) + acc(dv))
    writer_ct = ([H, Nt, Dt, qkv_tb, part_tb, SG._packed_identity_scalar(), _packed_bf16(scale),
                  p["Qt"], kv_tb, SEM_WRITTEN]
                 + acc(dq) + acc(dk) + acc(dv) + acc(dbias_partial))

    # Subblocks are what keep a result inside DST. A [Nt, Nt] score block and a [Nt, Dt] gradient
    # column have different aspect ratios, so they get different ones.
    sq_h, sq_w = SG.largest_subblock(p["Qt"], Nt, p["dst_size"])
    col_h, col_w = SG.largest_subblock(Nt, Dt, p["dst_size"])
    qcol_h, _ = SG.largest_subblock(p["Qt"], Dt, p["dst_size"])
    compute_ct = [Nt, Dt, H, _f32_bits(scale), sq_h, sq_w, col_h, col_w, p["Qt"], qcol_h]

    kd = _kdir()
    rr, wr, cr = [], [], []
    addrs = (q.buffer_address(), k.buffer_address(), v.buffer_address(), do.buffer_address(),
             bias.buffer_address(), dq.buffer_address(), dk.buffer_address(),
             dv.buffer_address(), dbias_partial.buffer_address())
    for i in range(num_cores):
        core = ttnn.CoreCoord(i % gx, i // gx)
        # Heads on the fast index, so the cores sharing a head are adjacent and each of them reads
        # that head's bias once. Groups on the slow index own contiguous leading-axis rows.
        head, group = i % H, i // H
        r0 = min(group * p["rows_per_core"], p["B"])
        r1 = min(r0 + p["rows_per_core"], p["B"])
        rr.append((core, [addrs[0], addrs[1], addrs[2], addrs[3], addrs[4], head, r0, r1,
                          addrs[6], addrs[7]]))
        wr.append((core, [addrs[5], addrs[6], addrs[7], addrs[8], head, group, r0, r1]))
        cr.append((core, [r0, r1]))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(kd / "dataflow/reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=reader_ct, defines=[], runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(kd / "dataflow/writer.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=writer_ct, defines=[], runtime_args=wr,
            config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(kd / "compute/triatt_bw.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=compute_ct,
            defines=[("REDUCE_GRANULARITY", "1"), ("ADD_BLOCK_GRANULARITY", "1"),
                     ("STATS_GRANULARITY", "1"), ("SUB_EXP_GRANULARITY", "1"),
                     ("MUL_BCAST_GRANULARITY", "1"), ("DHT_GRANULARITY", "1"),
                     ("EXP_APPROX_MODE", "0")]
                    + ([("BW_DUMP", "1")] if os.environ.get("TT_BIO_TRIATT_BW_DUMP") else []),
            runtime_args=cr,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ckc[0], math_approx_mode=False,
                fp32_dest_acc_en=True, dst_full_sync_en=False)),
    ]
    # Counts (chunk, row) writes of dK/dV that have hit the write barrier, so the reader never
    # reads a partial back before it lands. Reset to zero every launch.
    sems = [ttnn.SemaphoreDescriptor(id=SEM_WRITTEN, core_ranges=core_grid, initial_value=0)]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "plan": p, "rt": (rr, wr, cr),
            "addrs": addrs}


def run(device, q, k, v, bias, g, scale, ckc, q_chunk_tiles=None, grid=None):
    """The whole backward for one triangle-attention call. Returns (dq, dk, dv, dbias).

    `q`, `k`, `v`, `bias` and `g` are raw ttnn tensors, not taped ones: the caller owns the tape.
    Outputs are allocated here and written in full by the program -- every leading-axis row is
    owned by exactly one core, and every `(group, head)` slab of the partial is written once -- so
    they are allocated uninitialised rather than zeroed.
    """
    B, H, N, d = (int(x) for x in q.padded_shape)
    if grid is None:
        cg = device.compute_with_storage_grid_size()
        grid = (cg.x, cg.y)
    # `q_chunk_tiles` forces a chunk, which is how the grader reaches the looped kernel at a
    # shape the whole-query form would also fit, and on a board `serving_plan` declines it on.
    # `grid` forces a smaller work split, which is how Wormhole's 8x8 runs on a Blackhole card.
    p = (serving_plan(B, H, N, d, grid) if q_chunk_tiles is None
         else plan(B, H, N, d, grid, q_chunk_tiles=q_chunk_tiles))
    if p is None or not fits_l1(p):
        raise ValueError(f"does not fit L1 at any query chunk: N={N}")

    def like(t, dtype=None):
        return ttnn.empty(t.padded_shape, dtype or t.dtype, ttnn.TILE_LAYOUT, device,
                          ttnn.DRAM_MEMORY_CONFIG)

    # dK and dV sum across query chunks, so with more than one they are float32 running sums.
    kv_dtype = ttnn.float32 if p["Qt"] < p["Nt"] else None
    dq, dk, dv = like(q), like(k, kv_dtype), like(v, kv_dtype)
    part = ttnn.empty(ttnn.Shape(partial_shape(p)), ttnn.float32, ttnn.TILE_LAYOUT, device,
                      ttnn.DRAM_MEMORY_CONFIG)
    e = build(device, q, k, v, g, bias, dq, dk, dv, part, p, ckc, scale)
    ttnn.generic_op([q, k, v, g, bias, dq, dk, dv, part], e["pd"])
    # The one reduction the host does. Every core accumulated its own group of the leading axis in
    # its own L1; this sums the groups.
    dbias = ttnn.sum(part, dim=0, keepdim=True)
    ttnn.deallocate(part)
    if kv_dtype is not None:
        # One rounding to bf16 at the end, the same one the whole-query kernel's pack makes.
        dk, dv = (ttnn.typecast(t, ttnn.bfloat16) for t in (dk, dv))
    STATS["served"] += 1
    return dq, dk, dv, dbias
