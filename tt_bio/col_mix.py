"""AF2's column attention over two MSA rows, from the projected q, k, v to the attended value, as one kernel.

`AF2Attention._column_depth2` writes it as thirteen programs on `[2, n, 256]` operands (five slices, two
subtracts, a multiply, the head-sum linear, the bias add, the sigmoid multiply and an add), each of which the
card finishes before the host has issued the next. `col_mix` reads the `[2, n, 3c]` projection once and writes
the `[2, n, c]` result once (`kernels/col_mix/compute.cpp`). A head is 32 channels, one tile, so the work
splits by (token tile row, head) and never leaves a tile.

Not bit-identical to the composed chain: the intermediates stay in a float32 DST instead of being rounded to
bfloat16 between programs. `perf/bc4_msafuse/col_probe.py` grades both against float64.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import genq
from . import ops as _ops
from .envflags import env_flag
from .rne_add import _split_plan

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "col_mix"
TILE = 32
IN_CB, BIAS_CB, CONST_CB, D_CB, OUT_CB, MID_CB, P_CB, G_CB = 0, 1, 2, 3, 16, 24, 25, 26

#: The lever. Default OFF until it is graded on a BindCraft 2 round.
COL_MIX = env_flag("TT_BIO_COL_MIX", False)

#: (calls served, calls declined), cumulative.
STATS = [0, 0]
_CACHE: dict = {}
_CONST: dict = {}


def _plain(t) -> bool:
    mc = t.memory_config()
    return (t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT
            and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def eligible(qkv, head_dim, bias=None) -> bool:
    """bfloat16 TILE interleaved `[2, n, 3 * heads * 32]`, heads one tile wide, and a `[1, n, 1]` bias or none.
    Under a tape only with the `col_mix` entry installed (`TT_BIO_TAPED_KERNELS`)."""
    if not COL_MIX or _ops.declines_under_tape("col_mix"):
        return False
    shape = [int(d) for d in qkv.shape]
    ok = (head_dim == TILE and len(shape) == 3 and shape[0] == 2 and shape[2] % (3 * TILE) == 0 and _plain(qkv)
          and (bias is None or (_plain(bias) and [int(d) for d in bias.shape] == [1, shape[1], 1])))
    if not ok:
        STATS[1] += 1
    return ok


def _const(device, scale):
    """Two tiles: one full of `scale` (x @ it is the scaled row sum in every column), and one whose first row
    is ones (b @ it spreads b's first column over every column)."""
    key = (device.id(), scale)
    if key not in _CONST:
        t = torch.zeros(2 * TILE, TILE)
        t[:TILE] = scale
        t[TILE] = 1.0
        _CONST[key] = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return _CONST[key]


def _cb(idx, dtype, tiles, core_grid):
    tile_bytes = TILE * TILE * (4 if dtype == ttnn.float32 else 2)
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=tile_bytes)
    return ttnn.CBDescriptor(total_size=tiles * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt])


def _build(qkv, const, bias, out, device, nt, heads, d=None):
    units = nt * heads
    num_cores, core_grid, cg1, cg2, work1, work2 = _split_plan(device, units)
    assign, placed = {}, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    assign[(cx, cy)] = (placed, per_core)
                    placed += per_core
    assert (len(assign), placed) == (num_cores, units), (len(assign), placed, units)
    plan_ct = genq.compact_plan(assign, core_grid) if genq.compact() else None
    genq_ct = [1] + plan_ct if plan_ct else [0] * 8
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if plan_ct is None:
        for (cx, cy), (fst, per_core) in assign.items():
            reader_rt[cx][cy] = [fst, per_core]
            compute_rt[cx][cy] = [per_core]
            writer_rt[cx][cy] = [fst, per_core]

    has_bias, bw = int(bias is not None), int(d is not None)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[nt, heads, has_bias, bw] + acc(qkv) + acc(const) + acc(bias if has_bias else const)
        + acc(d if bw else const) + genq_ct,
        runtime_args=reader_rt, common_runtime_args=[0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[nt, heads, bw] + acc(out) + genq_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[has_bias, bw] + genq_ct, runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True))
    cbs = [_cb(IN_CB, ttnn.bfloat16, 12, core_grid), _cb(CONST_CB, ttnn.bfloat16, 2, core_grid),
           _cb(OUT_CB, ttnn.bfloat16, 12 if bw else 4, core_grid),
           _cb(MID_CB, ttnn.float32, 4 if bw else 2, core_grid)]
    if has_bias:
        cbs.append(_cb(BIAS_CB, ttnn.bfloat16, 2, core_grid))
    if bw:
        cbs += [_cb(D_CB, ttnn.bfloat16, 4, core_grid), _cb(P_CB, ttnn.float32, 4, core_grid),
                _cb(G_CB, ttnn.float32, 2, core_grid)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def _run(qkv, scale, bias, d):
    device = qkv.device()
    two, n, c3 = (int(x) for x in qkv.shape)
    nt, heads = -(-n // TILE), c3 // (3 * TILE)
    const = _const(device, scale)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([two, n, c3 if d is not None else c3 // 3]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    g_ = device.compute_with_storage_grid_size()
    bt = lambda t: t is not None and t.memory_config().buffer_type
    key = (device.id(), nt, heads, bt(qkv), bt(bias), bt(d), g_.x, g_.y, genq.compact(), scale)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(qkv, const, bias, out, device, nt, heads, d)
    reader, writer, compute = entry["kernels"]
    addr = lambda t: t.buffer_address() if t is not None else 0
    reader.common_runtime_args = [addr(qkv), addr(const), addr(bias), addr(d)]
    writer.common_runtime_args = [addr(out)]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"])
    STATS[0] += 1
    ttnn.generic_op([qkv, const] + [t for t in (bias, d) if t is not None] + [out], pd)
    return out


@_ops.fused_kernel("col_mix")
def col_mix(qkv, scale, bias=None):
    """`v1 + sigmoid(scale * headsum(q * (k0 - k1)) + bias) * (v0 - v1)` for both MSA rows, `[2, n, c]` in DRAM."""
    return _run(qkv, scale, bias, None)


def col_mix_bw(qkv, d, scale, bias=None):
    """The cotangent of `qkv` from `d`, that of `col_mix`'s output (bfloat16, `[2, n, c]`): `[2, n, 3c]` in DRAM."""
    return _run(qkv, scale, bias, d)
