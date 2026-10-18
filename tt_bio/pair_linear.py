"""``x [.., K] @ w [K, N]`` with the weight resident in L1 on every core, for Wormhole's pair projections.

On a Wormhole chip minimal_matmul puts N on the 9-core grid axis when M > N, so a 4-tile N leaves 5 of 9 core rows
computing padding and the projection is compute-bound on 32 of 72 cores. Here every core loads the whole weight into
L1 once and streams whole 32-row tiles of x through it (pair_swiglu_ws's machinery with one matmul, its writer
shared), so all cores work and x is read from DRAM once. Measured on a Wormhole chip at [1,224,224,K] (.107,
perf/spd/pair_linear_gate.py): N=128 0.187 -> 0.141 ms, the [..,384] x [384,128] dX 0.566 -> 0.276, N=384/512 1.1x,
bit-identical to minimal_matmul (one fp32 dest pass over K). On Blackhole minimal_matmul is as fast or faster, so
`pair_mm` asks this only on Wormhole. A bias seeds the fp32 dest with its row-replicated tiles (`PL_BIAS`), so it is
summed in fp32 with the products; AF2's pair projections mostly carry one, and without it they ran ttnn.linear.

`matmul` returns the product or None, and None means the caller keeps its own path.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_linear"
WRITER = Path(__file__).resolve().parent / "kernels" / "pair_swiglu_ws" / "writer.cpp"
X_CB, W_CB, B_CB, OUT_CB = 0, 1, 2, 16
X_DEPTH = 3            # row tiles of x in flight per core
_TILE = 32 * 32 * 2
#: L1 a core may spend on CBs; W [kt, ct] + x X_DEPTH * kt + out 2 * ct (+ bias ct) tiles must fit.
L1_BUDGET = 1000 * 1024
_CACHE: dict = {}
_BIAS: dict = {}


def _replicated(b, ct):
    """The bias as 32 identical rows of ``ct`` tiles, built once per bias tensor (kept with it, as
    `pair_mm._transposed` keeps a weight, so a recycled id cannot hand back a stale copy)."""
    hit = _BIAS.get(id(b))
    if hit is None or hit[0] is not b:
        import torch
        row = ttnn.to_torch(b).reshape(-1)[: int(b.shape[-1])].to(torch.bfloat16)
        rows = torch.zeros(32, 32 * ct, dtype=torch.bfloat16)
        rows[:, : row.numel()] = row
        hit = _BIAS[id(b)] = (b, ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                 device=b.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG))
    return hit[1]


def forget() -> None:
    """Drop every replicated bias (`pair_mm.forget` calls this on a trunk eviction)."""
    _BIAS.clear()


def _build(x, w, out, mt, kt, ct, fidelity, bias=None):
    g = x.device().compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, mt, 0)

    def cb(idx, tiles):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=_TILE)
        return ttnn.CBDescriptor(total_size=tiles * _TILE, core_ranges=grid, format_descriptors=[fmt])

    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == mt, (first, mt)
    rct = [kt, ct]
    for t in (x, w) if bias is None else (x, w, bias):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [ct] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    defines = [] if bias is None else [("PL_BIAS", "1")]
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 3,
                                   defines=defines, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(WRITER), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=[], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=path, core_ranges=grid,
        compile_time_args=[kt, ct], runtime_args=cp, defines=defines,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    cbs = [cb(X_CB, X_DEPTH * kt), cb(W_CB, kt * ct), cb(OUT_CB, 2 * ct)]
    if bias is not None:
        cbs.append(cb(B_CB, ct))
    return [reader, writer, compute], cbs


def matmul(x, w, compute_kernel_config=None, bias=None):
    """``x @ w (+ bias)`` as a new bf16 DRAM tensor, or None if the call is outside what the kernel does.

    It sums K in an fp32 dest at the caller's fidelity (HiFi4 without a config); a bf16-dest config declines.
    """
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        return None
    fidelity = ttnn.MathFidelity.HiFi4 if compute_kernel_config is None else compute_kernel_config.math_fidelity
    for t in (x, w) if bias is None else (x, w, bias):
        mc = t.memory_config()
        if (t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return None
    xs = [int(d) for d in x.padded_shape]
    mt = 1
    for d in xs[:-1]:
        mt *= d
    mt //= 32
    kt, ct = xs[-1] // 32, int(w.padded_shape[-1]) // 32
    bs = None if bias is None else [int(d) for d in bias.shape]      # ttnn.Shape does not support slicing
    if bs is not None and (bs[-1] != int(w.shape[-1]) or any(d != 1 for d in bs[:-1])):
        return None
    if (int(w.padded_shape[0]) != xs[-1]
            or (X_DEPTH * kt + kt * ct + (2 if bias is None else 3) * ct) * _TILE > L1_BUDGET):
        return None
    b = None if bias is None else _replicated(bias, ct)
    dev = x.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w.shape[-1])]),
                                         ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    key = (dev.id(), tuple(xs), ct, x.memory_config().buffer_type, w.memory_config().buffer_type, fidelity, b is None)
    if key not in _CACHE:
        _CACHE[key] = _build(x, w, out, mt, kt, ct, fidelity, b)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [x.buffer_address(), w.buffer_address(), 0 if b is None else b.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    ins = [x, w] if b is None else [x, w, b]
    return ttnn.generic_op(ins + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
