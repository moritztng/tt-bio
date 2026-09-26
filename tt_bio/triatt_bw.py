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
dS, which is the thing being avoided. So each core carries a `[S, S]` float32 accumulator in its
own L1 for its own group of the leading axis, writes it once at the end, and a single `ttnn.sum`
over the group axis finishes it. Float32 and not bf16: the accumulator sums a few hundred terms
and the reduce sums the partials on top of that, and a bf16 accumulator would put a rounding floor
under the one gradient this kernel produces by reduction rather than by matmul.

The key axis is never chunked, which is what keeps log-sum-exp bookkeeping out of this file. A
flash kernel chunks keys, so every block sees a partial softmax denominator and has to carry
running row statistics and rescale. Here each core holds every key for its rows, so each softmax
is exact and complete the first time. The QUERY axis may be chunked freely -- `dbias` rows are
indexed by the query row, so a query chunk needs no cross-chunk reduction -- and that is the knob
that carries this to longer token axes, where `bias + P + dS` stops fitting whole.

Driven through `tt_bio.sdpa_generic`'s machinery rather than the wheel's SDPA kernels: the forward
inherits `reader_interleaved.cpp`'s chain-forwarding, multicast, paging and MLA arguments, all of
which are dead at this call, and a backward built on top of them would be harder to read than
three small kernels of its own. What is inherited is the part that earned its place -- the CB
table, the work split, and the forward's own proof that a head's bias can sit permanently fronted
in L1 while the batch streams past it.

Default OFF behind `TT_BIO_TRIATT_BW_FUSED`. Gated in `tt_bio/autograd.py::triangle_attention`,
which keeps its chunked-recompute backward as the fallback for every shape this refuses.
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

# The whole-query form needs `bias`, `P`, `dS` and a transpose scratch live at once, plus a float32
# `dbias` accumulator that does not shrink with the query chunk. Past this many tiles on the token
# axis the whole-query form stops fitting and the caller falls back; `q_chunk_tiles` is what pulls
# it back under the line at a longer axis.
# Blackhole has 1.5 MB of L1 per core; tt-metal reserves the bottom of it for the program itself,
# and `sdpa_generic` already carries the figure the forward was sized against.
L1_PER_CORE = getattr(SG, "L1_PER_CORE", 1499136)
PROGRAM_RESERVE = getattr(SG, "PROGRAM_RESERVE", 0)

FUSED = env_flag("TT_BIO_TRIATT_BW_FUSED", False)

STATS = {"served": 0, "declined": 0}


def _kdir() -> Path:
    return Path(__file__).parent / "kernels" / "triatt_bw"


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def plan(B: int, H: int, N: int, d: int, grid, q_chunk_tiles=None, dst_size=8):
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


def fits_l1(p) -> bool:
    """Whether tt-metal will accept this config's CBs, priced the way `sdpa_generic` prices."""
    return p["l1_bytes"] + PROGRAM_RESERVE <= L1_PER_CORE


def largest_fitting_q_chunk(B, H, N, d, grid, **kw):
    """The biggest query chunk that fits, or None if even one tile row does not.

    The query axis is the only knob that shrinks the score-sized buffers, and shrinking it costs
    nothing in DRAM bytes or in arithmetic: `dbias` rows are indexed by the query row, so query
    chunks do not have to be reduced against each other. It does not touch the key axis, which is
    what would force log-sum-exp bookkeeping.
    """
    Nt = N // TILE
    for qt in range(Nt, 0, -1):
        if Nt % qt:
            continue
        p = plan(B, H, N, d, grid, q_chunk_tiles=qt, **kw)
        if fits_l1(p):
            return p
    return None


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
CB_P = 24                        # S, then P, in place
CB_DP = 25                       # dP, then dS, in place
CB_T = 26                        # transpose scratch: P^T for dV, dS^T for dK
CB_ROW_A, CB_ROW_B = 27, 28      # row max, then row sum, then its reciprocal
CB_DBIAS = 29                    # float32 accumulator, [Nt, Nt], persistent across the whole group
CB_DONE = 30                     # carries no data: the writer's handshake on the dbias accumulator
CB_DQ, CB_DK, CB_DV = 16, 17, 18

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
    return [
        (CB_Q, Nt * Dt * 2, b16, bf16),
        (CB_K, Nt * Dt * 2, b16, bf16),
        (CB_V, Nt * Dt * 2, b16, bf16),
        (CB_DO, Nt * Dt * 2, b16, bf16),
        (CB_BIAS, Nt * Nt, b16, bf16),
        (CB_SCALAR, 1, b16, bf16),
        (CB_ZERO, 1, b16, bf16),
        (CB_SCALE, 1, b16, bf16),
        (CB_ONES, Nt, b16, bf16),
        (CB_DONE, 1, b16, bf16),
        (CB_P, Qt * Nt, b16, bf16),
        (CB_DP, Qt * Nt, b16, bf16),
        (CB_T, Qt * Nt, b16, bf16),
        (CB_ROW_A, Qt, b16, bf16),
        (CB_ROW_B, Qt, b16, bf16),
        (CB_DBIAS, Nt * Nt, b32, f32),
        (CB_DQ, Nt * Dt * 2, b16, bf16),
        (CB_DK, Nt * Dt * 2, b16, bf16),
        (CB_DV, Nt * Dt * 2, b16, bf16),
    ]


def cb_bytes(p) -> int:
    return sum(n * page for _i, n, page, _f in cb_table(p))


# Every CB index the three kernels name, kept beside the table it has to agree with.
KERNEL_CB_INDICES = {CB_Q, CB_K, CB_V, CB_DO, CB_BIAS, CB_SCALAR, CB_ZERO, CB_SCALE, CB_ONES,
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
    reader_ct = [H, Nt, Dt, qkv_tb, bias_tb] + acc(q) + acc(k) + acc(v) + acc(do) + acc(bias)
    writer_ct = ([H, Nt, Dt, qkv_tb, part_tb, SG._packed_identity_scalar(), _packed_bf16(scale)]
                 + acc(dq) + acc(dk) + acc(dv) + acc(dbias_partial))

    # Subblocks are what keep a result inside DST. A [Nt, Nt] score block and a [Nt, Dt] gradient
    # column have different aspect ratios, so they get different ones.
    sq_h, sq_w = SG.largest_subblock(Nt, Nt, p["dst_size"])
    col_h, col_w = SG.largest_subblock(Nt, Dt, p["dst_size"])
    compute_ct = [Nt, Dt, H, _f32_bits(scale), sq_h, sq_w, col_h, col_w]

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
        rr.append((core, [addrs[0], addrs[1], addrs[2], addrs[3], addrs[4], head, r0, r1]))
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
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "plan": p, "rt": (rr, wr, cr),
            "addrs": addrs}
