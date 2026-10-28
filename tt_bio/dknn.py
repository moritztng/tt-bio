"""RFD3's atom neighbour index (the outside-mask half of _neighbours_kdtree) as one RISC-V program on the card.

Coordinates go up as int32 fixed point at the largest power-of-two scale with |q| <= 2^30, so every squared
distance is an exact uint64 and the k-th-nearest cut is decided on integers. On six noise levels (0-160 A, 6051
atoms, k 128) that reproduces the float64 k-d tree's rows exactly; a 2^14 scale flips 1-5 rows at near-ties.
Per row the kernel finds the k-th smallest key, keeps the atoms outside the row's mask within it and, if more
than n_fill remain, the n_fill nearest. The host joins that with the sequence entries exactly as before.

In a binder or motif design most atoms do not move (the page fixture: 4651 of 6051, bit for bit over all steps).
`Static` finds each static atom's nearest static atoms once per design, and a step scores a static row against
those and the movers only (the kernel's candidate mode), under a bound that proves no other atom can be in the row.
A row the bound cannot prove is computed in full on the card, so the result is the full launch's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import ttnn

from . import core_split
from .envflags import env_flag
from .sdpa_generic import L1_PER_CORE, PROGRAM_RESERVE

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "dknn" / "dknn.cpp")
# Read the index back behind an event (read_later) instead of with a blocking to_torch at .result(). A blocking read
# waits for everything queued before it, which by then includes the token encoder; the event waits for the index only.
ASYNC_READ = env_flag("RFD3_DKNN_ASYNC_READ", True)
_CB = (0, 8)   # first CB of BRISC's and NCRISC's six
_CACHE: dict = {}
_MASKS: dict = {}


def _align(n, a=8):
    return (n + a - 1) // a * a


def scale(x):
    """The fixed point's exponent s for x: the largest with every |x * 2^s| <= 2^30."""
    return float(np.floor(np.log2(2.0 ** 30 / max(float(np.abs(x).max()), 1e-6))))


def fixed_point(x):
    """[L, 3] float -> [L, 4] int32 at scale 2^s, s the largest with every |q| <= 2^30."""
    x = np.asarray(x, np.float64)
    q = np.zeros((len(x), 4), np.int32)
    q[:, :3] = np.rint(x * 2.0 ** scale(x))
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


#: Static candidates per row beyond k. Movers only lower a row's k-th key, so the bound fails only where the k-th
#: and the (k + SLACK + 1)-th static distances lie within the fixed point's rounding (ties, a lattice).
SLACK = 8
#: Static share of the atoms before candidate mode pays for its per-design build.
MIN_STATIC = 0.25


def _cb_sizes(L, MW, WO, CW=0):
    sizes = (L * 16, L * 8, L * 16, MW * 4, WO * 4)   # coords, keys, scratch, mask row, out row
    return sizes + (CW * 4,) if CW else sizes          # candidate row


def fits(L, k):
    """True when both RISC-Vs' buffers fit one core's L1 (about 18k atoms at k 128)."""
    MW, WO, CW = _align(1 + (L + 31) // 32), _align(k), _align(4 + k + SLACK)
    return len(_CB) * sum(_cb_sizes(L, MW, WO, CW)) + PROGRAM_RESERVE <= L1_PER_CORE


def _build(device, L, K, MW, WO, xt, mt, ot, ct=None, rows=None):
    g = device.compute_with_storage_grid_size()
    CW = int(ct.shape[1]) if ct is not None else 0
    r0, r1 = rows or (0, L)
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, r1 - r0, 0)
    sizes = _cb_sizes(L, MW, WO, CW)
    cbs = []
    for base in _CB:
        for i, size in enumerate(sizes):
            fmt = ttnn.CBFormatDescriptor(buffer_index=base + i, data_format=ttnn.uint32, page_size=size)
            cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=grid, format_descriptors=[fmt]))
    rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    first = r0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    half = (per + 1) // 2
                    rts[0][cx][cy] = [first, half]
                    rts[1][cx][cy] = [first + half, per - half]
                    first += per
    assert first == r1, (first, r1)
    acc = [a for t in (xt, mt, ot, mt if ct is None else ct) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=path, core_ranges=grid,
                                     compile_time_args=[L, K, MW, WO, base, CW] + acc, runtime_args=rt,
                                     common_runtime_args=[0] * 7, config=cfg)
               for base, rt, cfg in zip(_CB, rts, (ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()))]
    return kernels, cbs


def _upload(a, device):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _launch(device, L, k, xt, mt, WO, ot, o0, rows=None, cand=None):
    """One dknn program: rows (r0, r1) in full into ot from row o0, or (cand = (ct, m0, m1)) the candidate table's
    rows, each against its candidates and the movers [m0, m1)."""
    MW = int(mt.shape[1])
    ct, m0, m1 = cand or (None, 0, 0)
    rows = (0, int(ct.shape[0])) if ct is not None else rows or (0, L)
    ck = (device.id(), L, k, MW, WO, rows, None if ct is None else int(ct.shape[1]))
    if ck not in _CACHE:
        _CACHE[ck] = _build(device, L, k, MW, WO, xt, mt, ot, ct, rows)
    kernels, cbs = _CACHE[ck]
    for kd in kernels:
        kd.common_runtime_args = [xt.buffer_address(), mt.buffer_address(), ot.buffer_address(), o0,
                                  0 if ct is None else ct.buffer_address(), m0, m1]
    ttnn.generic_op([xt, mt] + ([ct] if ct is not None else []) + [ot],
                    ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def outside_neighbours(device, x, mask, seq, k, rows=None, static=None):
    """Device [L, WO] int32: per row the kept outside-mask neighbours ascending, padded with L (not read back).

    x [L, 3] float; mask [L, L] bool and seq [L, k] (the sequence entries, L = none) are the design-level torch
    tensors of _attention_index_prefix, uploaded once per design (keyed on the mask object).
    `rows` (r0, r1): only those rows, against every atom, into a [r1 - r0, WO] result.
    `static` (a Static): when it answers x, the static rows run in candidate mode; same result."""
    L = len(x)
    hit = _MASKS.get(id(mask))
    if hit is None or hit[0] is not mask:
        _MASKS.clear()
        n_fill = (k - (seq < L).sum(1)).cpu().numpy()
        hit = _MASKS[id(mask)] = (mask, _upload(mask_words(mask.cpu().numpy(), n_fill), device),
                                  _align(int(n_fill.max()) or 1))
    _, mt, WO = hit
    xt = _upload(fixed_point(x).reshape(1, L * 4), device)
    plan = static.plan(device, x, k) if static is not None and rows is None else None
    r0, r1 = rows or (0, L)
    ot = ttnn.allocate_tensor_on_device(ttnn.Shape([r1 - r0, WO]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device,
                                        ttnn.DRAM_MEMORY_CONFIG)
    if plan is None:
        _launch(device, L, k, xt, mt, WO, ot, r0, rows)
    else:
        ct, m0, m1 = plan
        if m1 > m0:
            _launch(device, L, k, xt, mt, WO, ot, 0, (m0, m1))
        _launch(device, L, k, xt, mt, WO, ot, 0, cand=plan)
    return ot


class Static:
    """One per design stream: the static atoms (coordinates identical to the previous call's) and, per static row,
    its k + SLACK nearest static atoms and the distance D to the next one. At scale 2^s every other static atom's
    integer key is at least (2^s D - sqrt 3)^2 (each coordinate rounds by at most 1/2 on both ends), so a row whose
    k-th key among candidates and movers is below (2^s D - 2)^2 is the full row. The movers must be one id range."""

    def __init__(self):
        self.last = self.ids = None
        self.tables = {}   # s -> candidate table on the card

    def _build(self, x, ids, k):
        from scipy.spatial import cKDTree

        L, xs = len(x), x[ids]
        nc = min(k + SLACK, len(ids))
        d, j = cKDTree(xs).query(xs, k=min(nc + 1, len(ids)))
        d, j = d.reshape(len(ids), -1), j.reshape(len(ids), -1)
        self.D = d[:, nc] if d.shape[1] > nc else np.full(len(ids), np.inf)
        words = np.full((len(ids), _align(4 + nc)), L, np.uint32)
        words[:, 0] = ids
        words[:, 1:4] = 0
        words[:, 4:4 + nc] = ids[j[:, :nc]]
        for t in self.tables.values():
            ttnn.deallocate(t)
        self.ids, self.xs, self.words, self.tables = ids, xs, words, {}

    def plan(self, device, x, k):
        """(candidate table, m0, m1) when the cache answers x, else None (and the cache learns from x)."""
        x = np.asarray(x, np.float64)
        L = len(x)
        last, self.last = self.last, x
        if self.ids is None or self.ids[-1] >= L or not np.array_equal(x[self.ids], self.xs):
            self.ids = None
            if last is None or last.shape != x.shape:
                return None
            still = (x == last).all(1)
            mid = np.flatnonzero(~still)
            # a repeated call (no movers) teaches nothing; the movers must be one range
            if still.sum() < max(MIN_STATIC * L, k + 1) or not len(mid) or mid[-1] - mid[0] + 1 != len(mid):
                return None
            self._build(x, np.flatnonzero(still), k)
        s = scale(x)
        if s not in self.tables:
            t = 2.0 ** s * self.D
            b = np.minimum(np.where(t > 2, (t - 2) ** 2 * (1 - 1e-12), 0.0), 1.8e19).astype(np.uint64)
            w = self.words.copy()
            w[:, 1], w[:, 2] = (b & 0xFFFFFFFF).astype(np.uint32), (b >> np.uint64(32)).astype(np.uint32)
            self.tables[s] = _upload(w.view(np.int32), device)
        moving = np.ones(L, bool)
        moving[self.ids] = False
        mid = np.flatnonzero(moving)
        return (self.tables[s],) + ((int(mid[0]), int(mid[-1]) + 1) if len(mid) else (0, 0))


def read_later(device, ts):
    """Queue the readback of device tensors `ts` now and return a callable giving them as torch tensors. With
    ASYNC_READ the callable waits on an event recorded right after the reads, so work enqueued between this call and
    it is not waited for; the bytes are the same either way."""
    if not ASYNC_READ:
        def get():
            out = [ttnn.to_torch(t) for t in ts]
            for t in ts:
                ttnn.deallocate(t)
            return out
        return get
    hs = [ttnn.from_device(t, blocking=False) for t in ts]
    ev = ttnn.record_event(device)

    def get():
        ttnn.event_synchronize(ev)
        for t in ts:
            ttnn.deallocate(t)
        return [ttnn.to_torch(h) for h in hs]
    return get


class Pending:
    """The index of each design, launched; .result() reads back and joins (the same call as a Future's)."""

    def __init__(self, device, X_L, mask, seq, k, static=None):
        self.seq, self.k = seq.cpu().numpy(), k
        static = static if len(X_L) == 1 else None
        self.read = read_later(device, [
            outside_neighbours(device, x.detach().float().cpu().numpy(), mask, seq, k, static=static) for x in X_L])

    def result(self):
        L = self.seq.shape[0]
        return torch.cat([join(self.seq, o.numpy(), L, self.k) for o in self.read()], 0)


def join(seq, out, L, k):
    """Host side, as _neighbours_kdtree: sequence entries + kept neighbours, ascending, first k."""
    vals = np.concatenate([seq.astype(np.int32), out], 1)
    vals.sort(1)
    return torch.from_numpy(vals[:, :k].astype(np.int64)).unsqueeze(0)
