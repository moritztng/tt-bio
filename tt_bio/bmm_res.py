"""``a [1, C, M, K] @ b [1, C, K, N]`` per channel with ``b[c]`` resident in L1, for the trimul einsum on Wormhole.

The production plan (`tenstorrent._triangle_mul_program_config`, a 2D multicast with a 2x2-tile block per core at
512 tokens) runs the [1, 256, 512, 512] einsum at 3.39 ms on a Wormhole chip at 1000 MHz, 2.5x its ~1.4 ms DRAM floor
(402 MB at 288 GB/s), and neither fidelity nor any ttnn program config moves it (perf/spd/af2_trimul_bmm_sweep.py at 512:
whole-channel plans overflow L1, row-block reuse re-reads b per block). Here each core owns whole channels: it loads
b[c] into L1 once (512 KB bf16 at 512 tokens, two channels deep when L1 allows), streams a[c]'s 32-row tiles through
it and writes out[c], so a, b and out each cross DRAM once. The reader, compute and writer are pair_linear's, keyed
per channel. One fp32 dest pass over all of K.

Where b[c] does not fit (above 704 tokens bf16), BRES_SPLIT (Wormhole; TT_BIO_BRES_SPLIT=0/1 forces it) splits N: a work
unit is (channel, a block of NB of the NT output tile columns), so b[c] is resident one column block at a time and
a[c] is read once per block. At 1024 tokens that is two blocks, a read twice: 8 MB of DRAM traffic a channel against
the 6 MB floor. A split call streams the transposed a in 4-tile groups and drains the output one dest group (compute's
OG) at a time, so b's block fits L1 beside them. The accumulation order over K is unchanged, so a split call returns
exactly what an unsplit one would.

`transpose_a` / `transpose_b` take ttnn.matmul's meaning (the operand is stored [.., K, M] / [.., N, K]), which is
how the trimul's gin_moved chunks arrive: the reader fetches tiles in the transposed order, the matmul unpacker
transposes each b tile, and compute transposes each a tile into a bf16 staging CB before it multiplies (ttnn's in0
transpose). A tile transpose moves values only and bf16 holds every bfp8 value, so neither changes a number.

BRES_B8 (off) keeps a bf16 b resident as bfp8 where that removes a split: the reader stages b[c] in BG-tile groups
and compute repacks each group into the resident buffer, so at 1024 tokens b[c] (1.06 MB) fits whole and a[c] is
read once (6 MB of DRAM traffic a channel instead of 8). It rounds b to bfp8, as trimul_b8in does at the in-projection.

The trimul einsum calls it under the `trimul_bres` lever (tenstorrent.py); perf/spd/af2_trimul_bmm_sweep.py --res times it.
`matmul` returns the product or None, and None means the caller keeps its own path.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import core_split
from . import mm_generic as MG
from .envflags import env_flag

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "bmm_res"
A_CB, B_CB, A_RAW_CB, B_RAW_CB, OUT_CB = 0, 1, 2, 3, 16
A_DEPTH = 2            # row tiles of a in flight per core
SPLIT_GROUP = 4        # a split call's transposed-a read group in tiles (compute's own transpose grouping)
#: None (unset) splits on Wormhole only. On a Blackhole p150a at 1350 MHz the split einsum lost to ttnn.matmul at 1024
#: tokens: ESMFold2-Fast 41.49 -> 42.00 s normal, 31.82 -> 32.30 s --fast, ESMFold2 --fast 54.09 -> 54.72 s.
BRES_SPLIT = None if os.environ.get("TT_BIO_BRES_SPLIT", "").strip() == "" else env_flag("TT_BIO_BRES_SPLIT", True)
BRES_B8 = env_flag("TT_BIO_BRES_B8", False)
#: Soak knobs for the WH HiFi2 transposed-a hang (perf/spd_m_opendde/bres_soak.py), compute.cpp defines that change no
#: number: BRES_DRAIN drains every Tensix thread at each transpose <-> matmul switch, BRES_FULL_TINIT / BRES_FULL_MINIT
#: run the full transpose / matmul init there instead of the short one.
SOAK: tuple = ()
#: L1 a core may spend on CBs: b depth * kt * nt + a A_DEPTH * kt (+ kt bf16 staged for transpose_a) + out 2 * nt tiles.
L1_BUDGET = 1200 * 1024
_CACHE: dict = {}


def _groups(kt, nb, ns, transpose_a, small=False):
    """(a read group, output drain) in tiles: whole rows unsplit; split (or `small`, B8's tight L1), the transpose
    group and compute's OG."""
    if ns == 1 and not small:
        return kt, nb
    return (SPLIT_GROUP if transpose_a else kt), (4 if nb % 4 == 0 else 2 if nb % 2 == 0 else 1)


def _rest_bytes(kt, nb, ns, a_bytes, transpose_a, b8=False):
    """Every CB but b's: a's row tiles (the raw transposed groups + the bf16 stage), the output and B8's b stage."""
    bf16 = MG.tile_bytes(ttnn.bfloat16)
    ag, w = _groups(kt, nb, ns, transpose_a, b8)
    return A_DEPTH * ag * a_bytes + transpose_a * kt * bf16 + 2 * w * bf16 + b8 * 2 * _bg(kt, nb) * bf16


def _bg(kt, nb):
    """BRES_B8's b staging group in tiles: compute repacks b two tiles at a time where the count allows."""
    return 2 if kt * nb % 2 == 0 else 1


def _plan(kt, nt, a_bytes, b_bytes, budget=L1_BUDGET, transpose_a=False, split=None, b8=False):
    """(column blocks ns, b units in flight 2 or 1) for the fewest blocks that fit, or (0, 0). `split` None takes
    BRES_SPLIT; False allows only ns = 1, an int forces that ns."""
    if split is None:
        from .tenstorrent import arch_name     # ttnn.get_arch_name() would open the card to plan
        split = BRES_SPLIT if BRES_SPLIT is not None else arch_name() == "wormhole_b0"
    if split is False:
        cands = (1,)
    elif split is True:
        cands = tuple(n for n in range(1, nt + 1) if nt % n == 0)
    else:
        cands = (split,) if nt % split == 0 else ()
    for ns in cands:
        nb = nt // ns
        rest = _rest_bytes(kt, nb, ns, a_bytes, transpose_a, b8)
        for depth in (2, 1):
            if depth * kt * nb * b_bytes + rest <= budget:
                return ns, depth
    return 0, 0


def cb_bytes(kt, nt, a_dtype, b_dtype, transpose_a=False, compute_kernel_config=None, budget=L1_BUDGET) -> int:
    """Static CB bytes per core a call at this shape plans, or 0 where `matmul` declines on these arguments.

    For a caller deciding whether a tensor it is about to place in L1 can sit beside this call: the budget is a
    constant, not a fit test against live L1, so an L1 buffer above `unreserved - cb_bytes` clashes at program
    creation. At 512 tokens that is 1179648 B (1212416 with transpose_a), which no L1 `a` operand fits beside.
    """
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        return 0
    a_bytes, b_bytes = MG.tile_bytes(a_dtype), MG.tile_bytes(b_dtype)
    ns, depth = _plan(kt, nt, a_bytes, b_bytes, budget, transpose_a)
    nb = nt // ns if ns else 0
    return depth and depth * kt * nb * b_bytes + _rest_bytes(kt, nb, ns, a_bytes, transpose_a)


def _build(a, b, out, C, mt, kt, nt, ta, tb, ns, depth, fidelity, max_cores, abl=0, b8=False):
    g = a.device().compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, C * ns, max_cores)
    nb = nt // ns
    ag, w = _groups(kt, nb, ns, ta, b8)

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
    bg = _bg(kt, nb) if b8 else 0
    rct = [mt, kt, nt, int(ta), int(tb), nb, ag, bg]
    for t in (a, b):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [mt, nt, nb, w] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 2,
                                   defines=[("BRES_NO_A", "1")] if abl & 4 else [], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=[("BRES_NO_WRITE", "1")] if abl & 2 else [], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=path, core_ranges=grid,
        compile_time_args=[mt, kt, nb, int(ta), int(tb), ag if ns > 1 or b8 else 0, bg], runtime_args=cp,
        defines=([("BRES_NO_MM", "1")] if abl & 1 else []) + [(d, "1") for d in SOAK],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    a_cb = cb(A_CB, kt, ttnn.bfloat16) if ta else cb(A_CB, A_DEPTH * kt, a.dtype)
    cbs = [a_cb, cb(B_CB, depth * kt * nb, ttnn.bfloat8_b if b8 else b.dtype), cb(OUT_CB, 2 * w, ttnn.bfloat16)]
    if ta:
        cbs.append(cb(A_RAW_CB, A_DEPTH * ag, a.dtype))
    if b8:
        cbs.append(cb(B_RAW_CB, 2 * bg, b.dtype))
    return [reader, writer, compute], cbs


STATS = [0, 0]   # calls served, calls declined (None returned)


def matmul(a, b, compute_kernel_config=None, transpose_a=False, transpose_b=False, budget=L1_BUDGET, max_cores=0,
           depth=None, abl=0, split=None, b8=None):
    """``a @ b`` over the channel axis as a new bf16 DRAM tensor, or None if the call is outside what the kernel does.

    It sums K in an fp32 dest at the caller's fidelity (HiFi4 without a config); a bf16-dest config declines.
    `max_cores` caps the channel split (0 = the whole grid) and `depth` forces b's channel depth; both are bench
    knobs that move scheduling, never a number, and so is `split` (see `_plan`). `b8` (None: BRES_B8) rounds a bf16 b
    to bfp8 in L1 where that plans fewer column blocks than bf16 does, and changes the result as that rounding does. `abl` is a bench-only stage mask that does change the result:
    1 skips the matmuls, 2 the output writes, 4 the reads of a (time attribution, af2_trimul_bmm_sweep --abl).
    """
    out = _matmul(a, b, compute_kernel_config, transpose_a, transpose_b, budget, max_cores, depth, abl, split,
                  BRES_B8 if b8 is None else b8)
    STATS[out is None] += 1
    return out


def _matmul(a, b, compute_kernel_config, transpose_a, transpose_b, budget, max_cores, depth, abl, split, b8):
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
    ns, fit = _plan(kt, nt, MG.tile_bytes(a.dtype), MG.tile_bytes(b.dtype), budget, transpose_a, split)
    b8 = bool(b8) and b.dtype == ttnn.bfloat16 and ns != 1
    if b8:
        ns8, fit8 = _plan(kt, nt, MG.tile_bytes(a.dtype), MG.tile_bytes(ttnn.bfloat8_b), budget, transpose_a, split, True)
        b8 = bool(fit8) and (not fit or ns8 < ns)
        if b8:
            ns, fit = ns8, fit8
    depth = min(depth or fit, fit)
    if not depth:
        return None
    dev = a.device()
    am, bn = (-1 if transpose_a else -2), (-2 if transpose_b else -1)
    shape = [int(d) for d in a.shape][:-2] + [int(a.shape[am]), int(b.shape[bn])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape),
                                         ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    key = (dev.id(), tuple(sa), tuple(sb), a.dtype, b.dtype, a.memory_config().buffer_type,
           b.memory_config().buffer_type, transpose_a, transpose_b, ns, depth, fidelity, max_cores, abl, b8, SOAK)
    if key not in _CACHE:
        _CACHE[key] = _build(a, b, out, C, mt, kt, nt, transpose_a, transpose_b, ns, depth, fidelity, max_cores, abl, b8)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    return ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
