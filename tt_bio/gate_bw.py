"""The backward of a sigmoid gate, ``o * sigmoid(g)``, as one Tensix kernel.

Every gate in the AF2 Evoformer is written ``ttnn.multiply(o, g, input_tensor_b_activations=
[SIGMOID])`` and the tape backpropagates it as seven ops: the sigmoid of g for ``do``, two
multiplies, then ``ttnn.sigmoid_bw``, which is itself sigmoid, rsub and two multiplies. In the
profiled BindCraft 2 round that is 516 chains and 0.425 s of card (`perf/bcp_evo/census_blocks.py`
lineage), ~1.0 ms each at ``[288, 288, 128]``. This reads d, o, g once and writes do, dg once, with
everything between in a float32 DST (`kernels/gate_bw/compute_gate_bw.cpp`).

The program is rne_add's: genq's compact split, a descriptor cached per shape and placement, and
the tensor addresses as common runtime args. Anything it declines takes the composed path, which
is correct at every shape.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import genq
from .envflags import env_flag
from .rne_add import _NUM_CBS, _split_plan, _tile_count, _tile_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "gate_bw"

TILE = 32
D_CB, O_CB, G_CB, DO_CB, DG_CB = 0, 1, 2, 16, 17
#: Tiles per reader/writer block; compute walks one tile at a time (four DST slots a tile).
GRAN = 4
_ELEM = {ttnn.bfloat16: 2, ttnn.float32: 4}

#: The lever. Default OFF and release-gated; armed by `bindcraft2.fast_round()`. Not `FUSED`:
#: the round stamp keys levers by attribute name and `triatt_bw.FUSED` already has it.
GATE_BW_FUSED = env_flag("TT_BIO_GATE_BW_FUSED", False)

#: (calls served, calls declined), cumulative; sample at a round boundary.
STATS = [0, 0]
#: Declines by (reason, shape of d).
REJECTS: dict = {}

_CACHE: dict = {}


def _reject(reason, d):
    k = (reason, tuple(int(x) for x in d.shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return False


def eligible(d, o, g) -> bool:
    """Same tile grid on all three, TILE interleaved, o and g bfloat16, d bfloat16 or float32."""
    if not GATE_BW_FUSED:
        return False
    if o.dtype != ttnn.bfloat16 or g.dtype != ttnn.bfloat16 or d.dtype not in _ELEM:
        return _reject("dtype", d)
    if any(t.layout != ttnn.TILE_LAYOUT for t in (d, o, g)):
        return _reject("layout", d)
    if not (tuple(d.shape) == tuple(o.shape) == tuple(g.shape)):
        return _reject("shape", d)
    if any(t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
           for t in (d, o, g)):
        return _reject("sharded", d)
    return True


def _cb(idx, dtype, core_grid):
    tile_bytes = TILE * TILE * _ELEM[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=tile_bytes)
    return ttnn.CBDescriptor(total_size=2 * GRAN * tile_bytes, core_ranges=core_grid,
                             format_descriptors=[fmt])


def _build(d, o, g, outs, device):
    num_tiles = _tile_count(d)
    num_cores, core_grid, cg1, cg2, work1, work2 = _split_plan(device, num_tiles)
    assign, placed = {}, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    assign[(cx, cy)] = (placed, per_core)
                    placed += per_core
    assert (len(assign), placed) == (num_cores, num_tiles), (len(assign), placed, num_tiles)
    plan_ct = genq.compact_plan(assign, core_grid) if genq.compact() else None
    genq_ct = [1] + plan_ct if plan_ct else [0] * 8

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if plan_ct is None:
        for (cx, cy), (fst, per_core) in assign.items():
            reader_rt[cx][cy] = [fst, per_core]
            compute_rt[cx][cy] = [per_core]
            writer_rt[cx][cy] = [fst, per_core]

    reader_ct = [GRAN]
    for t in (d, o, g):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [GRAN]
    for t in outs:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=True)
    if d.dtype == ttnn.float32:
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
        modes[D_CB] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_gate_bw.cpp"), source_type=src,
        core_ranges=core_grid, compile_time_args=reader_ct + genq_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0, 0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_gate_bw.cpp"), source_type=src,
        core_ranges=core_grid, compile_time_args=writer_ct + genq_ct, runtime_args=writer_rt,
        common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_gate_bw.cpp"), source_type=src,
        core_ranges=core_grid, compile_time_args=[D_CB, O_CB, G_CB, DO_CB, DG_CB] + genq_ct,
        runtime_args=compute_rt, config=cfg)
    cbs = [_cb(D_CB, d.dtype, core_grid), _cb(O_CB, o.dtype, core_grid),
           _cb(G_CB, g.dtype, core_grid), _cb(DO_CB, outs[0].dtype, core_grid),
           _cb(DG_CB, outs[1].dtype, core_grid)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def gate_bw(d, o, g):
    """``(d * s, d * o * s * (1 - s))`` with ``s = sigmoid(g)``, both in d's dtype (the dtype
    the composed path's multiplies return)."""
    device = d.device()
    outs = [ttnn.allocate_tensor_on_device(d.shape, d.dtype, ttnn.TILE_LAYOUT, device,
                                           d.memory_config()) for _ in range(2)]
    g_ = device.compute_with_storage_grid_size()
    key = (device.id(), _tile_grid(d), d.dtype, d.memory_config().buffer_type,
           o.memory_config().buffer_type, g.memory_config().buffer_type, g_.x, g_.y,
           genq.compact())
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(d, o, g, outs, device)
    reader, writer, compute = entry["kernels"]
    # Rebuilt per call so the new addresses are in the descriptor; rne_add's ADDR_WRITE_MODE
    # probe says whether that is needed, and the rebuild is the safe answer either way.
    reader.common_runtime_args = [d.buffer_address(), o.buffer_address(), g.buffer_address()]
    writer.common_runtime_args = [outs[0].buffer_address(), outs[1].buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[],
                                cbs=entry["cbs"])
    STATS[0] += 1
    ttnn.generic_op([d, o, g, outs[0], outs[1]], pd)
    return outs[0], outs[1]
