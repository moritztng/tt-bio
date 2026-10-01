"""The trimul's in-projection and its gated channel move as one kernel (bcp-evo stage 14).

Today an untaped TriangleMultiplication forward runs ``x @ W + b`` into a ``[1, N, N, 4C]``
projection and then two gated moves, each ``permute(p * sigmoid(g), (0, 3, 1, 2))`` for one role.
This kernel computes one role's ``[1, C, N, N]`` move output straight from ``x``: the projection
never reaches DRAM and the gate is applied to the float32 accumulator, so the result is rounded
to bf16 once instead of three times. Graded against float64, not against the two-op path
(perf/bcp_evo/inproj_gated_bench.py).

K is the pair width, 128 for AF2, so one role's W^T with its bias column is 2 x Ct x (Kt + 1)
tiles (80 KB at C = 128) and every core keeps all of it in L1. That is what makes a tile-at-a-time
matmul enough here: there is no K loop worth blocking and no weight traffic worth multicasting.

Untaped calls only. Under a tape the gated move's VJP reads the projection, so the taped recompute
keeps the two-op path.
"""
from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import reblock_permute as R
from .envflags import env_flag

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "inproj_gated"

INPROJ_GATED = env_flag("TT_BIO_INPROJ_GATED", False)
# Channel-tile subgroups per (it, jt): x is read S times per role, and the grid sees S x Nt^2
# groups. 0 = pick by grid (`_pick_s`).
S_FORCE = int(os.environ.get("TT_BIO_INPROJ_GATED_S", "0"))
# Groups of x the reader may run ahead by. 2 overlaps the next group's read with this one's math;
# the CB is 32 x Kt tiles a group (256 KB at K = 128), which is what an L1 clash would hit first.
X_BUFFERS = int(os.environ.get("TT_BIO_INPROJ_GATED_XBUF", "1"))
# Diagnostic ablations, never set in production: comma list of NO_GATHER (writer consumes without
# writing), NO_SFPU (no sigmoid or multiply), ONE_K (one K step). Each makes the output wrong.
# The reblock in compute (pack-untilize then tilize) instead of the writer's face-row gather.
TILIZE_PATH = env_flag("TT_BIO_INPROJ_GATED_TILIZE", True)
# Sigmoid and multiply as one SFPU pass, and the Newton steps after the reciprocal estimate.
GATE_FUSED = env_flag("TT_BIO_INPROJ_GATED_GATE_FUSED", True)
RECIP_ITERS = int(os.environ.get("TT_BIO_INPROJ_GATED_RECIP_ITERS", "1"))
# Rows of x a reader push carries (divides 32). 32 is the whole group in one push.
X_ROWS = int(os.environ.get("TT_BIO_INPROJ_GATED_XROWS", "2"))
DIAG = tuple(f"DIAG_{d}" for d in os.environ.get("TT_BIO_INPROJ_GATED_DIAG", "").split(",") if d)
FIDELITY = getattr(ttnn.MathFidelity, os.environ.get("TT_BIO_INPROJ_GATED_FIDELITY", "HiFi4"))

W_CB, X_CB, ONES_CB, SLAB_CB = 0, 1, 2, 3
OUT_CB, STAGE_CB = R.OUT_CB, R.STAGE_CB

_CACHE: dict = {}
STATS = [0, 0]  # served, declined
REJECTS: dict = {}
# Shapes whose program clashed with live L1 buffers once; declined from then on.
CLASH: set = set()


def clash_key(x):
    return tuple(int(d) for d in x.shape)


def prepare_weights(w, b, device, memory_config=None):
    """``(wt, ones)`` for ``inproj_gated``, from torch ``w`` [K, 4C] and ``b`` [4C] or None.

    ``wt`` is ``[1, 1, 4C, K + 32]`` when there is a bias (column K holds it, the rest of that tile
    is zero) and ``[1, 1, 4C, K]`` when not; ``ones`` is a [1, 1, 32, 32] tile with column 0 = 1,
    the x-side operand that makes the bias one more K step of the same matmul.
    """
    C4 = int(w.shape[1])
    wt = ttnn.from_torch(w.detach().float().t().contiguous().reshape(1, 1, C4, -1),
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
    if b is not None:
        wt = ttnn.concat([wt, bias_column(b, device)], dim=-1)
    return wt, ones_tile(device)


def bias_column(b, device):
    """Torch ``b`` [4C] as the ``[1, 1, 4C, 32]`` bf16 tile column ``wt`` ends in: column 0 = b."""
    import torch
    pad = torch.zeros(1, 1, int(b.numel()), R.TILE_W)
    pad[..., 0] = b.detach().float().reshape(-1)
    return ttnn.from_torch(pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


_ONES: dict = {}


def ones_tile(device):
    t = _ONES.get(device.id())
    if t is None:
        import torch
        ones = torch.zeros(1, 1, R.TILE_H, R.TILE_W)
        ones[..., 0] = 1.0
        t = _ONES[device.id()] = ttnn.from_torch(
            ones, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return t


def device_weights(w, bias_col):
    """``wt`` from the live device weight ``w`` [..., K, 4C], on card, every call.

    The trimul's fused in-projection weight is cut out of the model's leaves afresh each forward,
    so nothing keyed on its buffer can be cached: a fresh buffer misses (a host round trip and a
    device sync a call) and a reused address hits another weight. A transpose (and a concat with
    the cached ``bias_column``) is two small device ops instead. bf16 only: an fp32 weight
    transposed on card and then cast came out 1 ulp off on 6.4 % of elements, while a bare cast
    matched torch's round-to-nearest-even (perf/bcp_evo/ipg_wt_check.py), so the caller declines
    any other dtype.
    """
    assert w.dtype == ttnn.bfloat16, w.dtype
    K, C4 = int(w.shape[-2]), int(w.shape[-1])
    wt = ttnn.reshape(ttnn.transpose(w, -2, -1), [1, 1, C4, K])
    if bias_col is not None:
        wt = ttnn.concat([wt, bias_col], dim=-1)
    return wt


def _defines():
    return ([(d, "1") for d in DIAG + (("TILIZE_PATH",) if TILIZE_PATH else ())
             + (("GATE_FUSED",) if GATE_FUSED else ())]
            + [("GATE_RECIP_ITERS", str(RECIP_ITERS)), ("X_ROWS", str(X_ROWS))])


def _pick_s(device, Nt, Ct):
    if S_FORCE:
        return S_FORCE
    g = device.compute_with_storage_grid_size()
    cores = g.x * g.y
    # The fewest x re-reads that still give every core a group.
    for s in range(1, Ct + 1):
        if Ct % s == 0 and Nt * Nt * s >= cores:
            return s
    return Ct


def _build(x, wt, out, device, has_bias, S, reader_ct, writer_ct):
    N = int(out.shape[2])
    Nt = (N + R.TILE_H - 1) // R.TILE_H
    Kt = int(x.shape[3]) // R.TILE_W
    Ct = int(out.shape[1]) // R.TILE_W
    Kt1 = Kt + int(has_bias)
    num_groups = Nt * Nt * S
    plan = R._split_plan(device, num_groups)
    assert plan is not None, f"no work split for {num_groups} groups"
    _, _, (num_cores, core_grid, cg1, cg2, work1, work2) = plan
    tile_bytes = R.TILE_H * R.TILE_W * R._elem()

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=R._DTYPE,
                                      page_size=tile_bytes)
        return ttnn.CBDescriptor(total_size=depth * tile_bytes, core_ranges=core_grid,
                                 format_descriptors=[fmt])

    cbs = [cb(W_CB, 2 * Ct * Kt1), cb(X_CB, X_BUFFERS * R.GROUP_TILES * Kt),
           cb(OUT_CB, 2 * R.GROUP_TILES), cb(STAGE_CB, 2)]
    if TILIZE_PATH:
        cbs.append(cb(SLAB_CB, R.GROUP_TILES))
    if has_bias:
        cbs.append(cb(ONES_CB, 1))

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first, placed, block = 0, 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    g0, gs, ghi, glo = R._walk(R.WALK, first, block, per_core, num_cores)
                    reader_rt[cx][cy] = [g0, per_core, Nt, N, S, gs, ghi, glo]
                    compute_rt[cx][cy] = [g0, per_core, S, gs, ghi, glo]
                    writer_rt[cx][cy] = [g0, per_core, Nt, N, Ct, S, gs, ghi, glo]
                    first += 1
                    block += per_core
                    placed += per_core
    assert (first, placed) == (num_cores, num_groups), (first, placed, num_cores, num_groups)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_inproj_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor(),
        defines=_defines(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_inproj_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor(),
        defines=_defines(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_inproj_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[W_CB, X_CB, ONES_CB, OUT_CB, Kt, Ct, int(has_bias), SLAB_CB],
        runtime_args=compute_rt, defines=_defines(),
        config=ttnn.ComputeConfigDescriptor(math_fidelity=FIDELITY, fp32_dest_acc_en=True),
    )
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def inproj_gated(x, wt, ones, p_slice, g_slice, slice_c, out=None, memory_config=None):
    """``permute((x @ W + b)[..., p] * sigmoid((x @ W + b)[..., g]), (0, 3, 1, 2))``.

    ``x`` is ``[1, N, N, K]`` bf16 TILE, ``wt``/``ones`` come from `prepare_weights` (``ones`` is
    ignored without a bias column) and the slice arguments are in channels. Returns
    ``[1, slice_c, N, N]``, the tensor `reblock_permute_gated` returns for the same role.
    """
    device = x.device()
    N, K = int(x.shape[2]), int(x.shape[3])
    has_bias = int(wt.shape[3]) == K + R.TILE_W
    if out is None:
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, slice_c, N, N]), R._DTYPE,
                                             ttnn.TILE_LAYOUT, device, mc)
    Nt = (N + R.TILE_H - 1) // R.TILE_H
    Ct = slice_c // R.TILE_W
    S = _pick_s(device, Nt, Ct)
    reader_ct = [W_CB, X_CB, ONES_CB, K // R.TILE_W, Ct, int(has_bias)]
    reader_ct += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    reader_ct += list(ttnn.TensorAccessorArgs(wt).get_compile_time_args())
    reader_ct += list(ttnn.TensorAccessorArgs(ones).get_compile_time_args())
    writer_ct = [R._elem(), OUT_CB, R.TILE_H, R.TILE_W, R.FACE_H, R.FACE_W, STAGE_CB]
    writer_ct += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    g = device.compute_with_storage_grid_size()
    # The shape is in the key explicitly: interleaved accessor args do not carry it, and Nt / N
    # are runtime args baked into the cached descriptor.
    key = (device.id(), g.x, g.y, N, K, slice_c, R.WALK, S, X_BUFFERS, str(FIDELITY), has_bias, DIAG, TILIZE_PATH, GATE_FUSED, RECIP_ITERS, X_ROWS,
           tuple(reader_ct), tuple(writer_ct))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, wt, out, device, has_bias, S, reader_ct, writer_ct)
    reader, writer, compute = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), wt.buffer_address(), ones.buffer_address(),
                                  p_slice // R.TILE_W, g_slice // R.TILE_W]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[],
                                cbs=entry["cbs"])
    STATS[0] += 1
    return ttnn.generic_op([x, wt, ones, out], pd)
