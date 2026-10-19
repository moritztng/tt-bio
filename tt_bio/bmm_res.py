"""``a [1, C, M, K] @ b [1, C, K, N]`` per channel with ``b[c]`` resident in L1, for the trimul einsum on Wormhole.

The production plan (`tenstorrent._triangle_mul_program_config`, a 2D multicast with a 2x2-tile block per core at
512 tokens) runs the [1, 256, 512, 512] einsum at 3.39 ms on a Wormhole chip at 1000 MHz, 2.5x its ~1.4 ms DRAM floor
(402 MB at 288 GB/s), and neither fidelity nor any ttnn program config moves it (perf/spd/af2_trimul_bmm_sweep.py at 512:
whole-channel plans overflow L1, row-block reuse re-reads b per block). Here each core owns whole channels: it loads
b[c] into L1 once (512 KB bf16 at 512 tokens, two channels deep when L1 allows), streams a[c]'s 32-row tiles through
it and writes out[c], so a, b and out each cross DRAM once. The reader, compute and writer are pair_linear's, keyed
per channel. One fp32 dest pass over all of K.

Where b[c] does not fit (above 704 tokens bf16), BRES_SPLIT (off) splits N: a work unit is (channel, a block of NB
of the NT output tile columns), so b[c] is resident one column block at a time and a[c] is read once per block. At
1024 tokens that is two blocks, a read twice: 8 MB of DRAM traffic a channel against the 6 MB floor. A split call
streams the transposed a in 4-tile groups and drains the output one dest group (compute's OG) at a time, so b's
block fits L1 beside them. The accumulation order over K is unchanged, so a split call returns
exactly what an unsplit one would.

BRES_RT (1; 2 is a lever under test) is compute's row block: RT row tiles of a against each OG-tile group of b, so
every resident b tile is unpacked once per RT rows instead of once per row. RT 2 holds 2 x OG fp32 tiles, more than
half of dest, so it runs dest in full sync; RT + 1 rows of a are in flight where L1 allows, else RT, and the writer drains
RT x OG tiles at a time. Same products, same K order in one fp32 dest pass, so it returns exactly what RT 1 does.

`transpose_a` / `transpose_b` take ttnn.matmul's meaning (the operand is stored [.., K, M] / [.., N, K]), which is
how the trimul's gin_moved chunks arrive: the reader fetches tiles in the transposed order, the matmul unpacker
transposes each b tile, and compute transposes each a tile into a bf16 staging CB before it multiplies (ttnn's in0
transpose). A tile transpose moves values only and bf16 holds every bfp8 value, so neither changes a number.

The trimul einsum calls it under the `trimul_bres` lever (tenstorrent.py); perf/spd/af2_trimul_bmm_sweep.py --res times it.
`matmul` returns the product or None, and None means the caller keeps its own path.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split
from . import mm_generic as MG
from .envflags import env_flag, env_int

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "bmm_res"
A_CB, B_CB, A_RAW_CB, OUT_CB = 0, 1, 2, 16
A_DEPTH = 2            # row tiles of a in flight per core
SPLIT_GROUP = 4        # a split call's transposed-a read group in tiles (compute's own transpose grouping)
BRES_SPLIT = env_flag("TT_BIO_BRES_SPLIT", False)
BRES_RT = env_int("TT_BIO_BRES_RT", 1)
#: L1 a core may spend on CBs: b depth * kt * nt + a A_DEPTH * kt (+ kt bf16 staged for transpose_a) + out 2 * nt tiles.
L1_BUDGET = 1200 * 1024
_CACHE: dict = {}


def _og(nb):
    """compute's OG: output tiles a dest group spans along N."""
    return 4 if nb % 4 == 0 else 2 if nb % 2 == 0 else 1


def _a_rows(rt):
    """Candidate row tiles of a in flight, most first: A_DEPTH, or under BRES_RT one row of the next block
    prefetched (RT + 1) and else none (RT)."""
    return (A_DEPTH,) if rt == 1 else (rt + 1, rt)


def _groups(kt, nb, ns, transpose_a, rt=1):
    """(a read group, output drain) in tiles: whole rows unsplit; split, the transpose group and compute's OG;
    RT > 1, compute's RT x OG dest block."""
    ag = kt if ns == 1 or not transpose_a else SPLIT_GROUP
    if rt > 1:
        return ag, rt * _og(nb)
    return ag, (nb if ns == 1 else _og(nb))


def _rest_bytes(kt, nb, ns, a_bytes, transpose_a, rt=1, rows=A_DEPTH):
    """Every CB but b's: a's row tiles (the raw transposed groups + the bf16 stage) and the output."""
    bf16 = MG.tile_bytes(ttnn.bfloat16)
    ag, w = _groups(kt, nb, ns, transpose_a, rt)
    a_raw = A_DEPTH * ag if transpose_a else rows * ag
    return a_raw * a_bytes + transpose_a * rt * kt * bf16 + 2 * w * bf16


def _plan(kt, nt, a_bytes, b_bytes, budget=L1_BUDGET, transpose_a=False, split=None, rt=1):
    """(column blocks ns, b units in flight 2 or 1, a rows in flight) for the fewest blocks that fit, or (0, 0, 0).
    `split` None takes BRES_SPLIT; False allows only ns = 1, an int forces that ns."""
    split = BRES_SPLIT if split is None else split
    if split is False:
        cands = (1,)
    elif split is True:
        cands = tuple(n for n in range(1, nt + 1) if nt % n == 0)
    else:
        cands = (split,) if nt % split == 0 else ()
    for ns in cands:
        nb = nt // ns
        for depth in (2, 1):
            for rows in _a_rows(rt):
                if depth * kt * nb * b_bytes + _rest_bytes(kt, nb, ns, a_bytes, transpose_a, rt, rows) <= budget:
                    return ns, depth, rows
    return 0, 0, 0


def cb_bytes(kt, nt, a_dtype, b_dtype, transpose_a=False, compute_kernel_config=None, budget=L1_BUDGET,
             rt=None) -> int:
    """Static CB bytes per core a call at this shape plans, or 0 where `matmul` declines on these arguments.

    For a caller deciding whether a tensor it is about to place in L1 can sit beside this call: the budget is a
    constant, not a fit test against live L1, so an L1 buffer above `unreserved - cb_bytes` clashes at program
    creation. At 512 tokens that is 1179648 B (1212416 with transpose_a), which no L1 `a` operand fits beside.
    """
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        return 0
    a_bytes, b_bytes = MG.tile_bytes(a_dtype), MG.tile_bytes(b_dtype)
    rt = BRES_RT if rt is None else rt
    ns, depth, rows = _plan(kt, nt, a_bytes, b_bytes, budget, transpose_a, rt=rt)
    nb = nt // ns if ns else 0
    return depth and depth * kt * nb * b_bytes + _rest_bytes(kt, nb, ns, a_bytes, transpose_a, rt, rows)


def _build(a, b, out, C, mt, kt, nt, ta, tb, ns, depth, fidelity, max_cores, abl=0, rt=1, rows=A_DEPTH):
    g = a.device().compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, C * ns, max_cores)
    nb = nt // ns
    ag, w = _groups(kt, nb, ns, ta, rt)

    def cb(idx, tiles, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
        return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=grid,
                                 format_descriptors=[fmt])

    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == C * ns, (first, C, ns)
    rct = [mt, kt, nt, int(ta), int(tb), nb, ag]
    for t in (a, b):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [mt, nt, nb, w, rt] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 2,
                                   defines=[("BRES_NO_A", "1")] if abl & 4 else [], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=[("BRES_NO_WRITE", "1")] if abl & 2 else [], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=path, core_ranges=grid,
        compile_time_args=[mt, kt, nb, int(ta), int(tb), rt], runtime_args=cp,
        defines=[("BRES_NO_MM", "1")] if abl & 1 else [],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True,
                                            dst_full_sync_en=rt * _og(nb) > 4))
    a_cb = cb(A_CB, rt * kt, ttnn.bfloat16) if ta else cb(A_CB, rows * kt, a.dtype)
    cbs = [a_cb, cb(B_CB, depth * kt * nb, b.dtype), cb(OUT_CB, 2 * w, ttnn.bfloat16)]
    if ta:
        cbs.append(cb(A_RAW_CB, A_DEPTH * ag, a.dtype))
    return [reader, writer, compute], cbs


STATS = [0, 0]   # calls served, calls declined (None returned)


def matmul(a, b, compute_kernel_config=None, transpose_a=False, transpose_b=False, budget=L1_BUDGET, max_cores=0,
           depth=None, abl=0, split=None, rt=None):
    """``a @ b`` over the channel axis as a new bf16 DRAM tensor, or None if the call is outside what the kernel does.

    It sums K in an fp32 dest at the caller's fidelity (HiFi4 without a config); a bf16-dest config declines.
    `max_cores` caps the channel split (0 = the whole grid) and `depth` forces b's channel depth; both are bench
    knobs that move scheduling, never a number, and so are `split` (see `_plan`) and `rt` (BRES_RT). `abl` is a bench-only stage mask that does change the result:
    1 skips the matmuls, 2 the output writes, 4 the reads of a (time attribution, af2_trimul_bmm_sweep --abl).
    """
    out = _matmul(a, b, compute_kernel_config, transpose_a, transpose_b, budget, max_cores, depth, abl, split,
                  BRES_RT if rt is None else rt)
    STATS[out is None] += 1
    return out


def _matmul(a, b, compute_kernel_config, transpose_a, transpose_b, budget, max_cores, depth, abl, split, rt):
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        return None
    fidelity = ttnn.MathFidelity.HiFi4 if compute_kernel_config is None else compute_kernel_config.math_fidelity
    for t in (a, b):
        mc = t.memory_config()
        if (t.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or t.layout != ttnn.TILE_LAYOUT
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or len(t.shape) != 4
                or int(t.shape[0]) != 1):
            return None
    sa, sb = [int(d) for d in a.padded_shape], [int(d) for d in b.padded_shape]
    if (transpose_a or transpose_b) and (sa != [int(d) for d in a.shape] or sb != [int(d) for d in b.shape]):
        return None
    M, K = (sa[3], sa[2]) if transpose_a else (sa[2], sa[3])
    K2, N = (sb[3], sb[2]) if transpose_b else (sb[2], sb[3])
    if sa[1] != sb[1] or K != K2:
        return None
    C, mt, kt, nt = sa[1], M // 32, K // 32, N // 32
    if rt not in (1, 2) or mt % rt:
        rt = 1
    ns, fit, rows = _plan(kt, nt, MG.tile_bytes(a.dtype), MG.tile_bytes(b.dtype), budget, transpose_a, split, rt)
    depth = min(depth or fit, fit)
    if not depth:
        return None
    dev = a.device()
    am, bn = (-1 if transpose_a else -2), (-2 if transpose_b else -1)
    shape = [int(d) for d in a.shape][:-2] + [int(a.shape[am]), int(b.shape[bn])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape),
                                         ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    key = (dev.id(), tuple(sa), tuple(sb), a.dtype, b.dtype, a.memory_config().buffer_type,
           b.memory_config().buffer_type, transpose_a, transpose_b, ns, depth, fidelity, max_cores, abl, rt, rows)
    if key not in _CACHE:
        _CACHE[key] = _build(a, b, out, C, mt, kt, nt, transpose_a, transpose_b, ns, depth, fidelity, max_cores, abl,
                             rt, rows)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    return ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
