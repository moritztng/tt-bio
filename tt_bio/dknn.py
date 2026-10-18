"""RFD3's atom neighbour index (the outside-mask half of _neighbours_kdtree) as one RISC-V program on the card.

Coordinates go up as int32 fixed point at the largest power-of-two scale with |q| <= 2^30, so every squared
distance is an exact uint64 and the k-th-nearest cut is decided on integers. On six noise levels (0-160 A, 6051
atoms, k 128) that reproduces the float64 k-d tree's rows exactly; a 2^14 scale flips 1-5 rows at near-ties.
Per row the kernel finds the k-th smallest key, keeps the atoms outside the row's mask within it and, if more
than n_fill remain, the n_fill nearest. The host joins that with the sequence entries exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import ttnn

from . import core_split
from .sdpa_generic import L1_PER_CORE, PROGRAM_RESERVE

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "dknn" / "dknn.cpp")
_CB = (0, 8)   # first CB of BRISC's and NCRISC's five
_CACHE: dict = {}
_MASKS: dict = {}


def _align(n, a=8):
    return (n + a - 1) // a * a


def fixed_point(x):
    """[L, 3] float -> [L, 4] int32 at scale 2^s, s the largest with every |q| <= 2^30."""
    x = np.asarray(x, np.float64)
    span = max(float(np.abs(x).max()), 1e-6)
    q = np.zeros((len(x), 4), np.int32)
    q[:, :3] = np.rint(x * 2.0 ** np.floor(np.log2(2.0 ** 30 / span)))
    return q


def mask_words(mask, n_fill):
    """[L, L] bool mask + [L] n_fill -> [L, MW] int32 rows [n_fill, bit-packed mask...]."""
    L = mask.shape[0]
    bits = np.packbits(np.asarray(mask, bool), axis=1, bitorder="little")
    bits = np.pad(bits, ((0, 0), (0, (-bits.shape[1]) % 4))).view(np.uint32)
    out = np.zeros((L, _align(1 + bits.shape[1])), np.uint32)
    out[:, 0] = n_fill
    out[:, 1:1 + bits.shape[1]] = bits
    return out.view(np.int32)


def _cb_sizes(L, MW, WO):
    return (L * 16, L * 8, L * 16, MW * 4, WO * 4)   # coords, keys, scratch, mask row, out row


def fits(L, k):
    """True when both RISC-Vs' buffers fit one core's L1 (about 18k atoms at k 128)."""
    MW, WO = _align(1 + (L + 31) // 32), _align(k)
    return len(_CB) * sum(_cb_sizes(L, MW, WO)) + PROGRAM_RESERVE <= L1_PER_CORE


def _build(device, L, K, MW, WO, xt, mt, ot):
    g = device.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, L, 0)
    sizes = _cb_sizes(L, MW, WO)
    cbs = []
    for base in _CB:
        for i, size in enumerate(sizes):
            fmt = ttnn.CBFormatDescriptor(buffer_index=base + i, data_format=ttnn.uint32, page_size=size)
            cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=grid, format_descriptors=[fmt]))
    rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    half = (per + 1) // 2
                    rts[0][cx][cy] = [first, half]
                    rts[1][cx][cy] = [first + half, per - half]
                    first += per
    assert first == L, (first, L)
    acc = [a for t in (xt, mt, ot) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=path, core_ranges=grid,
                                     compile_time_args=[L, K, MW, WO, base] + acc, runtime_args=rt,
                                     common_runtime_args=[0, 0, 0], config=cfg)
               for base, rt, cfg in zip(_CB, rts, (ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()))]
    return kernels, cbs


def _upload(a, device):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def outside_neighbours(device, x, mask, seq, k):
    """Device [L, WO] int32: per row the kept outside-mask neighbours ascending, padded with L (not read back).

    x [L, 3] float; mask [L, L] bool and seq [L, k] (the sequence entries, L = none) are the design-level torch
    tensors of _attention_index_prefix, uploaded once per design (keyed on the mask object)."""
    L = len(x)
    hit = _MASKS.get(id(mask))
    if hit is None or hit[0] is not mask:
        _MASKS.clear()
        n_fill = (k - (seq < L).sum(1)).cpu().numpy()
        hit = _MASKS[id(mask)] = (mask, _upload(mask_words(mask.cpu().numpy(), n_fill), device),
                                  _align(int(n_fill.max()) or 1))
    _, mt, WO = hit
    xt = _upload(fixed_point(x).reshape(1, L * 4), device)
    ot = ttnn.allocate_tensor_on_device(ttnn.Shape([L, WO]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device,
                                        ttnn.DRAM_MEMORY_CONFIG)
    MW = int(mt.shape[1])
    ck = (device.id(), L, k, MW, WO)
    if ck not in _CACHE:
        _CACHE[ck] = _build(device, L, k, MW, WO, xt, mt, ot)
    kernels, cbs = _CACHE[ck]
    for kd in kernels:
        kd.common_runtime_args = [xt.buffer_address(), mt.buffer_address(), ot.buffer_address()]
    ttnn.generic_op([xt, mt, ot], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return ot


class Pending:
    """The index of each design, launched; .result() reads back and joins (the same call as a Future's)."""

    def __init__(self, device, X_L, mask, seq, k):
        self.seq, self.k = seq.cpu().numpy(), k
        self.outs = [outside_neighbours(device, x.detach().float().cpu().numpy(), mask, seq, k) for x in X_L]

    def result(self):
        L = self.seq.shape[0]
        rows = []
        for o in self.outs:
            rows.append(join(self.seq, ttnn.to_torch(o).numpy(), L, self.k))
            ttnn.deallocate(o)
        return torch.cat(rows, 0)


def join(seq, out, L, k):
    """Host side, as _neighbours_kdtree: sequence entries + kept neighbours, ascending, first k."""
    vals = np.concatenate([seq.astype(np.int32), out], 1)
    vals.sort(1)
    return torch.from_numpy(vals[:, :k].astype(np.int64)).unsqueeze(0)
