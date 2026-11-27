"""Device weights as views of upstream parameters: the gradient of a load-time fold.

A serving model folds at load. RFD4 multiplies `norm_cond.weight` into both AdaLN projections, scales
`to_bias^T` by `pair_norm.weight` and sqrt(head_dim), splits `to_qkv` and the transition input by output
column, concatenates the trimul in-projections, and keeps a second packed copy of the pair transition for
its fused kernel. A gradient taken on those device tensors names no upstream parameter, and training in
the folded space is a different model: Muon on three slices of `to_qkv` is not Muon on `to_qkv`, a product
of two parameters trains as one, and a duplicate copy goes stale for inference.

So optimizer state stays in upstream parameter space and the device weights stay derived. `Folds` builds
the model with the upstream state dict as float32 leaves that want a gradient and records the torch
source of every `ttnn.from_torch` made while it does. Each source carries the fold as a torch graph, so
the fold's VJP is torch's own backward through it, whatever shape the fold has: a closure, a raw
`from_torch` outside `Module.torch_to_tt`, or a lazy upload during the first forward (record that forward
too). Two device copies of one parameter sum. A weight uploaded from a tensor no leaf feeds is a constant
(`constants`: RFD4's trimul LayerNorm biases, which upstream builds with `bias=False`); a weight with no
recorded upload at all (made on the device, or outside the recording) is an orphan (`orphans`), and both are
named so a caller can refuse them, never zeroed.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping

import torch

from .bridge import HostParam

__all__ = ["Folds", "DeviceFolds", "make_rfd4_refold_hook"]


def _plain_transpose(s) -> bool:
    """`s` is a 2-D transpose of a leaf, or of slices of one: its VJP is pure layout, no reduction."""
    f = s.grad_fn
    if s.dim() != 2 or type(f).__name__ != "TBackward0":
        return False
    f = f.next_functions[0][0]
    while type(f).__name__ == "SliceBackward0":
        f = f.next_functions[0][0]
    return type(f).__name__ == "AccumulateGrad"


def _read_back(g, s):
    """A device gradient as a host tensor for source `s`. Most of the fold VJP is transposing gradients back
    (`W.t()` uploads); torch would do that as a strided copy on the host. For a plain transpose the card
    transposes instead and autograd gets a transposed view, so the host only accumulates contiguously. The
    bits go through as uint32: a float32 tile transpose rounds to ~TF32, a uint32 one moves data only. A
    transpose under a product (an AdaLN `norm_cond` fold) keeps the host path: its broadcast sum reduces in
    memory order, so a contiguous grad would round differently."""
    import ttnn
    if g.dtype != ttnn.float32:
        return ttnn.to_torch(g)
    u = ttnn.bitcast(g, ttnn.uint32)
    if _plain_transpose(s) and all(d == 1 for d in tuple(g.shape)[:-2]):
        t = ttnn.transpose(u, -2, -1)
        out = _untilized(t).reshape(s.shape[1], s.shape[0]).t()
        ttnn.deallocate(t)
        return out
    return _untilized(u)


#: Untilize float32 gradients on the card before reading them back (_untilized); off = host untilize.
CARD_UNTILIZE = True


def _untilized(u):
    """A uint32 TILE tensor's bits as host float32, untilized on the card: the host untilize is 2-3.5x slower
    on large gradients (`perf/r4t_perf/fold/rbrm.py`), and a uint32 untilize only moves data."""
    import ttnn
    if not CARD_UNTILIZE or u.layout != ttnn.TILE_LAYOUT:
        return ttnn.to_torch(u).view(torch.float32)
    r = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_torch(r).view(torch.float32)
    ttnn.deallocate(r)
    return out


class Folds:
    def __init__(self, state: Mapping[str, torch.Tensor]):
        self.leaves = {k: v.detach().to(torch.float32).clone().requires_grad_(True)
                       for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point()}
        self.sources = {}  # id(device handle) -> (handle, torch source); holding the handle keeps the id live
        self.consts = {}   # id(device handle) -> handle uploaded from a tensor no leaf feeds

    @contextmanager
    def recording(self, device_pack: bool = False):
        """Yields the leaf state dict to build from; every `ttnn.from_torch` of a tensor that wants a
        gradient is recorded until exit. The weight is uploaded from the detached source, so the build is
        the serving build bit for bit.

        ``device_pack`` packs bfloat8_b weights on the card (float32 upload, then typecast) instead of on
        the host, which is most of a per-step rebuild: host packing runs at ~110 Melem/s and does not
        thread. Not bit-identical: the card rounds a tie half away from zero where the host rounds it to
        even, so ~4e-6 of elements differ by one ulp and none lands further from the float32 source."""
        import ttnn
        from .. import tenstorrent as tt
        if getattr(tt, "_weight_cache", None) is not None:
            raise RuntimeError("Folds.recording under the shared weight cache: a cached tile has no torch source")
        real = ttnn.from_torch

        def upload(x, *a, **k):
            if (not a and k.get("dtype") == ttnn.bfloat16 and k.get("layout") == ttnn.TILE_LAYOUT
                    and k.get("device") is not None and x.dim() >= 2 and x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0):
                # Tilize on the card: the host tilize in from_torch is most of a rebuild (~10 ms vs ~1 ms per
                # 2.4 M elements at 2 CPUs). The bf16 rounding stays torch's, and tilize only moves data, so the
                # weight is the same bits (perf/r4t_perf/fold/tilt.py).
                r = real(x.to(torch.bfloat16), **{**k, "layout": ttnn.ROW_MAJOR_LAYOUT})
                t = ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=k.get("memory_config"))
                ttnn.deallocate(r)
                return t
            if not (device_pack and not a and k.get("dtype") == ttnn.bfloat8_b
                    and k.get("device") is not None and k.get("layout") == ttnn.TILE_LAYOUT):
                return real(x, *a, **k)
            t32 = real(x.to(torch.float32), **{**k, "dtype": ttnn.float32})
            t = ttnn.typecast(t32, ttnn.bfloat8_b)
            ttnn.deallocate(t32)
            return t

        def rec(x, *a, **k):
            if not torch.is_tensor(x):
                return real(x, *a, **k)
            t = upload(x.detach(), *a, **k)
            if x.requires_grad:
                self.sources[id(t)] = (t, x)
            else:
                self.consts[id(t)] = t
            return t

        ttnn.from_torch = rec
        try:
            yield dict(self.leaves)
        finally:
            ttnn.from_torch = real

    def source(self, handle):
        hit = self.sources.get(id(getattr(handle, "value", handle)))
        return None if hit is None or hit[0] is not getattr(handle, "value", handle) else hit[1]

    def constants(self, named: Mapping[str, object]) -> list:
        """Names among `named` (`{name: device handle or tape leaf}`) uploaded from no upstream leaf."""
        return sorted(n for n, h in named.items() if self.consts.get(id(getattr(h, "value", h))) is getattr(h, "value", h))

    def orphans(self, named: Mapping[str, object]) -> list:
        """Names among `named` with no recorded upload at all."""
        return sorted(n for n, h in named.items() if self.source(h) is None and n not in set(self.constants({n: h})))

    def unreached(self) -> list:
        """Upstream leaves no recorded source depends on: parameters the device model never reads."""
        srcs = [s for _, s in self.sources.values()]
        names = list(self.leaves)
        if not srcs:
            return sorted(names)
        got = torch.autograd.grad([s.sum() for s in srcs], [self.leaves[n] for n in names],
                                  allow_unused=True, retain_graph=True)
        return sorted(n for n, g in zip(names, got) if g is None)

    def backward(self, grads: Iterable) -> None:
        """Accumulate into the upstream leaves' `.grad` from `(device handle, gradient)` pairs, the
        gradient a torch or a device tensor. A handle with no source raises."""
        import ttnn
        srcs, gs = [], []
        for h, g in grads:
            s = self.source(h)
            if s is None:
                raise KeyError(f"no fold source for a device weight of shape {tuple(h.shape)}")
            if not torch.is_tensor(g):
                g = _read_back(g, s)
            srcs.append(s)
            gs.append(g.to(s.dtype).reshape(s.shape))
        if srcs:
            torch.autograd.backward(srcs, gs, retain_graph=True)

    def tape_backward(self) -> list:
        """`backward` from the tape: every recorded weight whose tape leaf holds a gradient. Returns the
        handles with none (a duplicate the taped forward never reads, or frozen this step).

        The device gradients are consumed: once summed into the upstream leaves they are freed and the
        tape leaves cleared, so a rebuild does not strand one float32 copy of the model per step."""
        import ttnn
        from .. import autograd as ag
        pairs, idle, used = [], [], []
        for h, _ in self.sources.values():
            leaf = ag.parameter_for(h)
            g = None if leaf is None else leaf.grad
            (idle if g is None else pairs).append(h if g is None else (h, g))
            if g is not None:
                used.append(leaf)
        self.backward(pairs)
        for leaf in used:
            g, leaf.grad = leaf.grad, None
            if g is not None and not torch.is_tensor(g):
                ttnn.deallocate(g)
        return idle

    def params(self, prefix: str = "") -> dict:
        """`{upstream name: HostParam}`, what the optimizer owns. Writing one does not move a device
        weight: record a rebuild after the step."""
        return {prefix + n: HostParam(p) for n, p in self.leaves.items()}


# =======================================================================================
# DeviceFolds: on-card fold backward + on-card refold (0 MB PCIe per step)
# =======================================================================================

_ONES_BF16_CACHE: dict[tuple[int, int], object] = {}


def _is_aligned_for_tile(shape: tuple[int, ...], dim: int, start: int, end: int) -> bool:
    """True when a slice/concat along ``dim`` on a TILE tensor lands on exact 32-element tile edges."""
    rank = len(shape)
    d = dim if dim >= 0 else rank + dim
    if d < rank - 2:
        return True
    return (start % 32 == 0) and (end % 32 == 0) and (shape[d] % 32 == 0)


def _dev_clone_f32(x):
    """Bit-exact fp32 TILE clone via uint32 bitcast (avoids TF32 truncation)."""
    import ttnn
    u = ttnn.bitcast(x, ttnn.uint32)
    c = ttnn.clone(u)
    return ttnn.bitcast(c, ttnn.float32)


def _dev_transpose_2d(x):
    """Bit-exact 2-D float32 transpose on Tenstorrent via uint32 bitcast."""
    from .device_optim import transpose_exact_f32
    return transpose_exact_f32(x)


def _to_row_major_f32(x):
    import ttnn
    if x.layout == ttnn.ROW_MAJOR_LAYOUT:
        return x, False
    u = ttnn.bitcast(x, ttnn.uint32)
    rm_u = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.bitcast(rm_u, ttnn.float32), rm_u


def _to_tile_f32(x):
    import ttnn
    if x.layout == ttnn.TILE_LAYOUT:
        return x, False
    u = ttnn.bitcast(x, ttnn.uint32)
    tl_u = ttnn.to_layout(u, ttnn.TILE_LAYOUT)
    return ttnn.bitcast(tl_u, ttnn.float32), tl_u


def _dev_slice_f32(x, dim: int, start: int, end: int):
    """Bit-exact fp32 slice ``x[..., start:end, ...]`` on card, supporting non-32-aligned axes.

    Never runs ROW_MAJOR slice along a non-32-aligned last dimension (known ttnn bug on narrow
    widths such as 12/24): for 2-D tensors sliced along dim 1, transposes to put the 32-aligned
    dimension last, slices contiguous rows along dim 0, re-tilizes, and transposes back.
    """
    import ttnn
    shape = tuple(int(s) for s in x.shape)
    rank = len(shape)
    d = dim if dim >= 0 else rank + dim
    beg = [0] * rank
    fin = list(shape)
    beg[d] = int(start)
    fin[d] = int(end)
    if _is_aligned_for_tile(shape, d, beg[d], fin[d]):
        u = ttnn.bitcast(x, ttnn.uint32)
        su = ttnn.slice(u, beg, fin)
        return ttnn.bitcast(su, ttnn.float32)
    # Non-32-aligned last dim on a 2-D tensor (e.g. bias_ws (384, 24) -> (384, 12)):
    if rank == 2 and d == 1 and shape[0] % 32 == 0:
        xt = _dev_transpose_2d(x)
        st = _dev_slice_f32(xt, 0, start, end)
        ttnn.deallocate(xt)
        out = _dev_transpose_2d(st)
        ttnn.deallocate(st)
        return out
    rm, rm_owner = _to_row_major_f32(x)
    s_rm = ttnn.slice(rm, beg, fin)
    if rm_owner is not False:
        ttnn.deallocate(rm_owner)
    tl, _ = _to_tile_f32(s_rm)
    ttnn.deallocate(s_rm)
    return tl


def _dev_concat_f32(xs: list, dim: int):
    """Bit-exact fp32 concat along ``dim`` on card, supporting non-32-aligned widths.

    For 2-D tensors concatenated along a non-32-aligned dim 1 (e.g. bias_ws (384, 12)+(384, 12)),
    transposes each piece to (12, 384), concatenates contiguous 32-aligned rows along dim 0 in
    ROW_MAJOR, re-tilizes, and transposes back to (384, 24).
    """
    import ttnn
    if len(xs) == 1:
        return _dev_clone_f32(xs[0])
    shape0 = tuple(int(s) for s in xs[0].shape)
    rank = len(shape0)
    d = dim if dim >= 0 else rank + dim
    aligned = all((d < rank - 2) or (int(t.shape[d]) % 32 == 0) for t in xs)
    if aligned:
        if len(xs) <= 24:
            us = [ttnn.bitcast(t, ttnn.uint32) for t in xs]
            cu = ttnn.concat(us, dim=d)
            return ttnn.bitcast(cu, ttnn.float32)
        chunks = [_dev_concat_f32(xs[i:i + 24], d) for i in range(0, len(xs), 24)]
        out = _dev_concat_f32(chunks, d)
        for c in chunks:
            ttnn.deallocate(c)
        return out
    if rank == 2 and d == 1 and shape0[0] % 32 == 0:
        xts = [_dev_transpose_2d(t) for t in xs]
        ct = _dev_concat_f32(xts, 0)
        for xt in xts:
            ttnn.deallocate(xt)
        out = _dev_transpose_2d(ct)
        ttnn.deallocate(ct)
        return out
    rms, owners = [], []
    for t in xs:
        rm, own = _to_row_major_f32(t)
        rms.append(rm)
        if own is not False:
            owners.append(own)
    c_rm = ttnn.concat(rms, dim=d)
    for own in owners:
        ttnn.deallocate(own)
    tl, _ = _to_tile_f32(c_rm)
    ttnn.deallocate(c_rm)
    return tl


def _dev_zeros_f32(shape: tuple[int, ...], device, layout=None):
    import ttnn
    lay = ttnn.TILE_LAYOUT if layout is None else layout
    return ttnn.zeros(list(shape), dtype=ttnn.float32, layout=lay, device=device)


def _dev_reshape_f32(x, target_shape: tuple[int, ...]):
    """Bit-exact fp32 reshape on card without ever creating a width-1 ROW_MAJOR tensor.

    ``ttnn.to_layout`` on width-1 ROW_MAJOR tensors (e.g. ``(384, 1)``) corrupts data or wedges
    on Blackhole. Any ``(M, 1)`` source or target is routed through ``(1, M)`` + bit-exact 2-D
    TILE transpose instead.
    """
    import ttnn
    cur = tuple(int(s) for s in x.shape)
    tgt = tuple(int(s) for s in target_shape)
    if cur == tgt:
        return _dev_clone_f32(x)

    # Avoid reading a width-1 (M, 1) tensor via ROW_MAJOR: transpose to (1, M) first.
    src = x
    src_owned = False
    if len(cur) == 2 and cur[-1] == 1 and cur[0] > 1:
        src = _dev_transpose_2d(x)
        src_owned = True
        cur = (1, cur[0])
        if cur == tgt:
            return src

    # Avoid producing a width-1 (M, 1) tensor via ROW_MAJOR: reshape to (1, M), then transpose.
    if len(tgt) == 2 and tgt[-1] == 1 and tgt[0] > 1:
        mid = _dev_reshape_f32(src, (1, tgt[0]))
        if src_owned:
            ttnn.deallocate(src)
        out = _dev_transpose_2d(mid)
        ttnn.deallocate(mid)
        return out

    rm, rm_owner = _to_row_major_f32(src)
    r_rm = ttnn.reshape(rm, list(tgt))
    tl, _ = _to_tile_f32(r_rm)
    if rm_owner is not False:
        ttnn.deallocate(rm_owner)
    if src_owned:
        ttnn.deallocate(src)
    return tl


def _dev_expand_f32(x, target_shape: tuple[int, ...]):
    import ttnn
    cur = tuple(int(s) for s in x.shape)
    tgt = tuple(int(s) for s in target_shape)
    if cur == tgt:
        return _dev_clone_f32(x)
    assert len(cur) == len(tgt), (cur, tgt)
    reps = [t // c for c, t in zip(cur, tgt)]
    u = ttnn.bitcast(x, ttnn.uint32)
    ru = ttnn.repeat(u, ttnn.Shape(reps))
    return ttnn.bitcast(ru, ttnn.float32)


def _dev_broadcast_shape(t, target_shape: tuple[int, ...]):
    """Reshape leading 1s onto ``t`` so its rank matches ``target_shape`` for binary ops."""
    cur = tuple(int(s) for s in t.shape)
    if len(cur) == len(target_shape):
        return t, False
    padded = (1,) * (len(target_shape) - len(cur)) + cur
    r = _dev_reshape_f32(t, padded)
    return r, True


def _dev_mul_f32(a, b, out_shape: tuple[int, ...]):
    """Broadcast float32 multiply avoiding non-32-aligned width broadcast bugs.

    For 2-D ``(M, N) * (M, 1)`` with ``M % 32 == 0`` and ``N % 32 != 0`` (e.g. ``(384, 12) *
    (384, 1)`` in ``PairBiasAttention``), transposes both operands so the 32-aligned axis ``M``
    is last, multiplies ``(N, M) * (1, M)``, and transposes back bit-exactly.
    """
    import ttnn
    a_b, a_own = _dev_broadcast_shape(a, out_shape)
    b_b, b_own = _dev_broadcast_shape(b, out_shape)
    if len(out_shape) == 2 and out_shape[-1] % 32 != 0 and out_shape[-2] % 32 == 0:
        at = _dev_transpose_2d(a_b)
        bt = _dev_transpose_2d(b_b)
        pt = ttnn.multiply(at, bt)
        ttnn.deallocate(at)
        ttnn.deallocate(bt)
        out = _dev_transpose_2d(pt)
        ttnn.deallocate(pt)
    else:
        out = ttnn.multiply(a_b, b_b)
    if a_own:
        ttnn.deallocate(a_b)
    if b_own:
        ttnn.deallocate(b_b)
    return out


def _ones_row_bf16(k: int, device):
    """Cached ``[1, k]`` bf16 TILE row of ones (cols ``k..k_pad-1`` are exact 0.0)."""
    import ttnn
    key = (id(device), int(k))
    hit = _ONES_BF16_CACHE.get(key)
    if hit is not None and hit.is_allocated():
        return hit
    t = ttnn.from_torch(
        torch.ones(1, int(k), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    _ONES_BF16_CACHE[key] = t
    return t


def _sum_dim0_exact_f32(x2d):
    """Exact fp32 sum along dim 0 of a 2-D ``[K, M]`` TILE tensor -> ``[1, M]``.

    Uses ``ones([1, K]) @ hi + ones([1, K]) @ lo`` in bf16->fp32 HiFi4+fp32_dest_acc so:
    (1) padded tile rows ``K..K_pad-1`` are multiplied by exact 0.0 (zero tail leakage), and
    (2) avoids ``ttnn.sum``'s ~2.7e-4 TF32 unpack truncation (~3e-6 accuracy instead).
    """
    import ttnn
    from .device_optim import _ckc, _split2_bf16

    k = int(x2d.shape[0])
    ones = _ones_row_bf16(k, x2d.device())
    ckc = _ckc(ttnn.MathFidelity.HiFi4)
    hi, lo = _split2_bf16(x2d)
    s_hi = ttnn.matmul(ones, hi, dtype=ttnn.float32, compute_kernel_config=ckc)
    s_lo = ttnn.matmul(ones, lo, dtype=ttnn.float32, compute_kernel_config=ckc)
    ttnn.deallocate(hi)
    ttnn.deallocate(lo)
    out = ttnn.add(s_hi, s_lo)
    ttnn.deallocate(s_hi)
    ttnn.deallocate(s_lo)
    return out


def _dev_reduce_to_f32(g, target_shape: tuple[int, ...]):
    """Exact fp32 reduction of ``g`` over broadcast / leading axes to ``target_shape``."""
    import ttnn
    gshape = tuple(int(s) for s in g.shape)
    tgt = tuple(int(s) for s in target_shape)
    if gshape == tgt:
        return _dev_clone_f32(g)

    # 2-D column reduction (M, N) -> (M, 1) or (M,) when N is non-32-aligned and M is 32-aligned:
    if (len(gshape) == 2 and gshape[-1] % 32 != 0 and gshape[-2] % 32 == 0
            and (tgt == (gshape[0], 1) or tgt == (gshape[0],))):
        gt = _dev_transpose_2d(g)              # (N, M) with M 32-aligned
        s_row = _sum_dim0_exact_f32(gt)        # (1, M)
        ttnn.deallocate(gt)
        out = _dev_reshape_f32(s_row, tgt)     # (M, 1) via transpose or (M,) via aligned RM
        ttnn.deallocate(s_row)
        return out

    pad_len = len(gshape) - len(tgt)
    padded_tgt = (1,) * pad_len + tgt
    dims = [i for i, (gs, ts) in enumerate(zip(gshape, padded_tgt)) if gs != ts and ts == 1]

    # Fast exact path: single-axis or leading-axes reduction into last dim M (32-aligned).
    if dims and dims == list(range(len(gshape) - 1)) and gshape[-1] % 32 == 0:
        k_tot = 1
        for d in dims:
            k_tot *= gshape[d]
        m = gshape[-1]
        flat = g if gshape == (k_tot, m) else _dev_reshape_f32(g, (k_tot, m))
        s_row = _sum_dim0_exact_f32(flat)
        if flat is not g:
            ttnn.deallocate(flat)
        out = _dev_reshape_f32(s_row, tgt) if tuple(int(s) for s in s_row.shape) != tgt else s_row
        if out is not s_row:
            ttnn.deallocate(s_row)
        return out

    cur = g
    owned = False
    if dims:
        nxt = ttnn.sum(cur, dim=dims, keepdim=True)
        if owned:
            ttnn.deallocate(cur)
        cur, owned = nxt, True
    if tuple(int(s) for s in cur.shape) != tgt:
        nxt = _dev_reshape_f32(cur, tgt)
        if owned:
            ttnn.deallocate(cur)
        cur, owned = nxt, True
    if not owned:
        cur = _dev_clone_f32(cur)
    return cur


class _FoldNode:
    __slots__ = ("nid", "op", "inputs", "attrs", "shape")

    def __init__(self, nid: int, op: str, inputs: tuple[int, ...], attrs: dict, shape: tuple[int, ...]):
        self.nid = nid
        self.op = op
        self.inputs = inputs
        self.attrs = attrs
        self.shape = tuple(int(s) for s in shape)


class _OutputSpec:
    __slots__ = ("nid", "handle", "dtype", "layout", "memory_config", "shape", "sites")

    def __init__(self, nid: int, handle, dtype, layout, memory_config, shape: tuple[int, ...]):
        self.nid = nid
        self.handle = handle
        self.dtype = dtype
        self.layout = layout
        self.memory_config = memory_config
        self.shape = tuple(int(s) for s in shape)
        self.sites: list = []


class DeviceFolds:
    """Graph-compiled on-card weight folds: zero-PCIe ``tape_backward`` and ``refold``.

    Records the exact PyTorch aten fold DAG under ``with dfolds.recording() as leaves:``
    via ``TorchDispatchMode``, binds every uploaded device weight handle to its live
    model slot(s) via :meth:`bind`, then executes both fold VJP (:meth:`tape_backward`)
    and post-optimizer weight refresh (:meth:`refold`) 100% on device in float32.
    """

    def __init__(self, state: Mapping[str, torch.Tensor], device):
        self.device = device
        self.leaves = {
            k: v.detach().to(torch.float32).clone().requires_grad_(True)
            for k, v in state.items()
            if torch.is_tensor(v) and v.is_floating_point()
        }
        self.nodes: list[_FoldNode] = []
        self.leaf_nid: dict[str, int] = {}
        self.specs: list[_OutputSpec] = []
        self.sources: dict[int, tuple[object, torch.Tensor]] = {}
        self.consts: dict[int, object] = {}
        self._tuple_bindings: list = []
        self._bound = False

    def _add_node(self, op: str, inputs: tuple[int, ...], attrs: dict, shape: tuple[int, ...]) -> int:
        nid = len(self.nodes)
        self.nodes.append(_FoldNode(nid, op, inputs, attrs, shape))
        return nid

    @contextmanager
    def recording(self, device_pack: bool = False):
        """Record the fold DAG + device weight uploads while building the model."""
        import ttnn
        from torch.utils._python_dispatch import TorchDispatchMode
        from .. import tenstorrent as tt

        if getattr(tt, "_weight_cache", None) is not None:
            raise RuntimeError("DeviceFolds.recording under the shared weight cache: a cached tile has no fold source")

        self.nodes.clear()
        self.leaf_nid.clear()
        self.specs.clear()
        self.sources.clear()
        self.consts.clear()
        self._tuple_bindings.clear()
        self._bound = False

        keepalive: list[torch.Tensor] = []
        node_of: dict[int, int] = {}
        zero_bufs: dict[int, tuple[int, tuple[int, ...]]] = {}
        zero_aliases: dict[int, set[int]] = {}
        slice_views: dict[int, tuple[int, int, int, int]] = {}
        scatter_writes: dict[int, list[tuple[int, int, int, int]]] = {}

        for name, leaf in self.leaves.items():
            nid = self._add_node("leaf", (), {"name": name}, tuple(leaf.shape))
            self.leaf_nid[name] = nid
            node_of[id(leaf)] = nid
            keepalive.append(leaf)

        def pin(t: torch.Tensor, nid: int):
            node_of[id(t)] = nid
            keepalive.append(t)

        def _norm_dim(dim: int, rank: int) -> int:
            return dim if dim >= 0 else rank + dim

        class _FoldTrace(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                opname = str(func)

                if opname == "aten.zeros.default" and isinstance(out, torch.Tensor):
                    tid = id(out)
                    zero_bufs[tid] = (tid, tuple(int(s) for s in out.shape))
                    zero_aliases[tid] = {tid}
                    keepalive.append(out)
                    return out

                if opname in (
                    "aten.detach.default",
                    "aten.alias.default",
                    "aten._to_copy.default",
                    "aten.to.dtype",
                    "aten.to.dtype_layout",
                    "aten.contiguous.default",
                ):
                    x = args[0]
                    if isinstance(x, torch.Tensor) and isinstance(out, torch.Tensor):
                        if id(x) in node_of:
                            pin(out, node_of[id(x)])
                        if id(x) in zero_bufs:
                            rtid, rshape = zero_bufs[id(x)]
                            zero_bufs[id(out)] = (rtid, rshape)
                            zero_aliases[rtid].add(id(out))
                            keepalive.append(out)
                    return out

                if opname == "aten.clone.default":
                    x = args[0]
                    if isinstance(x, torch.Tensor) and id(x) in node_of and isinstance(out, torch.Tensor):
                        nid = outer._add_node("clone", (node_of[id(x)],), {}, tuple(out.shape))
                        pin(out, nid)
                    return out

                if opname == "aten.t.default":
                    x = args[0]
                    if isinstance(x, torch.Tensor) and id(x) in node_of and isinstance(out, torch.Tensor):
                        nid = outer._add_node("t", (node_of[id(x)],), {}, tuple(out.shape))
                        pin(out, nid)
                    return out

                if opname == "aten.slice.Tensor":
                    x = args[0]
                    dim = _norm_dim(int(args[1]) if len(args) > 1 else 0, x.dim())
                    dim_sz = int(x.shape[dim])
                    raw_start = args[2] if len(args) > 2 else None
                    raw_end = args[3] if len(args) > 3 else None
                    step = int(args[4]) if len(args) > 4 else 1
                    start = 0 if raw_start is None else int(raw_start)
                    if start < 0:
                        start = max(0, dim_sz + start)
                    end = dim_sz if raw_end is None else int(raw_end)
                    if end < 0:
                        end = max(0, dim_sz + end)
                    end = min(end, dim_sz)
                    if isinstance(x, torch.Tensor) and isinstance(out, torch.Tensor):
                        if id(x) in zero_bufs:
                            rtid, _ = zero_bufs[id(x)]
                            slice_views[id(out)] = (rtid, dim, start, end)
                            keepalive.append(out)
                        elif id(x) in node_of:
                            if step != 1:
                                raise NotImplementedError(f"DeviceFolds: slice step={step} != 1")
                            if start == 0 and end == dim_sz:
                                pin(out, node_of[id(x)])
                            else:
                                nid = outer._add_node(
                                    "slice", (node_of[id(x)],),
                                    {"dim": dim, "start": start, "end": end},
                                    tuple(out.shape),
                                )
                                pin(out, nid)
                    return out

                if opname == "aten.copy_.default":
                    dst, src = args[0], args[1]
                    if (isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor)
                            and id(dst) in slice_views and id(src) in node_of):
                        rtid, dim, start, end = slice_views[id(dst)]
                        writes = scatter_writes.setdefault(rtid, [])
                        writes.append((dim, start, end, node_of[id(src)]))
                        _, shape = zero_bufs[rtid]
                        nid = outer._add_node(
                            "scatter_slices",
                            tuple(w[3] for w in writes),
                            {"shape": shape, "writes": list(writes)},
                            shape,
                        )
                        for atid in zero_aliases[rtid]:
                            node_of[atid] = nid
                    return out

                if opname == "aten.cat.default":
                    xs = args[0]
                    dim = _norm_dim(int(args[1]) if len(args) > 1 else 0, out.dim())
                    if any(isinstance(t, torch.Tensor) and id(t) in node_of for t in xs):
                        in_nids = []
                        for t in xs:
                            if id(t) not in node_of:
                                raise NotImplementedError(
                                    f"DeviceFolds: cat mixing tracked and untracked tensors ({tuple(t.shape)})")
                            in_nids.append(node_of[id(t)])
                        nid = outer._add_node("cat", tuple(in_nids), {"dim": dim}, tuple(out.shape))
                        pin(out, nid)
                    return out

                if opname in ("aten.mul.Tensor", "aten.mul.Scalar"):
                    a, b = args[0], args[1]
                    a_tr = isinstance(a, torch.Tensor) and id(a) in node_of
                    b_tr = isinstance(b, torch.Tensor) and id(b) in node_of
                    if a_tr and b_tr:
                        nid = outer._add_node("mul", (node_of[id(a)], node_of[id(b)]), {}, tuple(out.shape))
                        pin(out, nid)
                    elif a_tr:
                        if isinstance(b, torch.Tensor):
                            if b.numel() != 1:
                                raise NotImplementedError(f"DeviceFolds: non-scalar const mul {tuple(b.shape)}")
                            sc = float(b.detach().item())
                        else:
                            sc = float(b)
                        nid = outer._add_node("mul_scalar", (node_of[id(a)],), {"scalar": sc}, tuple(out.shape))
                        pin(out, nid)
                    elif b_tr:
                        if isinstance(a, torch.Tensor):
                            if a.numel() != 1:
                                raise NotImplementedError(f"DeviceFolds: non-scalar const mul {tuple(a.shape)}")
                            sc = float(a.detach().item())
                        else:
                            sc = float(a)
                        nid = outer._add_node("mul_scalar", (node_of[id(b)],), {"scalar": sc}, tuple(out.shape))
                        pin(out, nid)
                    return out

                if opname in ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
                              "aten.unsqueeze.default"):
                    x = args[0]
                    if isinstance(x, torch.Tensor) and id(x) in node_of and isinstance(out, torch.Tensor):
                        nid = outer._add_node(
                            "view", (node_of[id(x)],), {"shape": tuple(int(s) for s in out.shape)},
                            tuple(out.shape),
                        )
                        pin(out, nid)
                    return out

                if opname == "aten.expand.default":
                    x = args[0]
                    if isinstance(x, torch.Tensor) and id(x) in node_of and isinstance(out, torch.Tensor):
                        nid = outer._add_node(
                            "expand", (node_of[id(x)],), {"shape": tuple(int(s) for s in out.shape)},
                            tuple(out.shape),
                        )
                        pin(out, nid)
                    return out

                def _any_tracked(v):
                    if isinstance(v, torch.Tensor):
                        return id(v) in node_of
                    if isinstance(v, (list, tuple)):
                        return any(_any_tracked(u) for u in v)
                    return False

                if isinstance(out, torch.Tensor) and _any_tracked(args):
                    if opname not in (
                        "aten.zeros_like.default",
                        "aten.ones_like.default",
                        "aten.empty_like.default",
                        "aten.new_zeros.default",
                    ):
                        raise NotImplementedError(
                            f"DeviceFolds: unsupported tracked fold op {opname} -> shape {tuple(out.shape)}")
                return out

        outer = self
        real = ttnn.from_torch

        def upload(x, *a, **k):
            if (not a and k.get("dtype") == ttnn.bfloat16 and k.get("layout") == ttnn.TILE_LAYOUT
                    and k.get("device") is not None and x.dim() >= 2
                    and x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0):
                r = real(x.to(torch.bfloat16), **{**k, "layout": ttnn.ROW_MAJOR_LAYOUT})
                t = ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=k.get("memory_config"))
                ttnn.deallocate(r)
                return t
            if not (device_pack and not a and k.get("dtype") == ttnn.bfloat8_b
                    and k.get("device") is not None and k.get("layout") == ttnn.TILE_LAYOUT):
                return real(x, *a, **k)
            t32 = real(x.to(torch.float32), **{**k, "dtype": ttnn.float32})
            t = ttnn.typecast(t32, ttnn.bfloat8_b)
            ttnn.deallocate(t32)
            return t

        def rec(x, *a, **k):
            if not torch.is_tensor(x):
                return real(x, *a, **k)
            nid = node_of.get(id(x))
            t = upload(x.detach(), *a, **k)
            if nid is not None:
                self.sources[id(t)] = (t, x)
                spec = _OutputSpec(
                    nid=nid,
                    handle=t,
                    dtype=k.get("dtype", t.dtype),
                    layout=k.get("layout", t.layout),
                    memory_config=k.get("memory_config", None),
                    shape=tuple(int(s) for s in x.shape),
                )
                self.specs.append(spec)
            elif x.requires_grad:
                raise RuntimeError(
                    f"DeviceFolds: uploaded tensor with requires_grad=True was not tracked in fold DAG "
                    f"(shape={tuple(x.shape)}, grad_fn={x.grad_fn})")
            else:
                self.consts[id(t)] = t
            return t

        ttnn.from_torch = rec
        try:
            with _FoldTrace():
                yield dict(self.leaves)
        finally:
            ttnn.from_torch = real
            keepalive.clear()

    def unreached(self) -> list[str]:
        """Upstream leaves no recorded device weight spec depends on."""
        needed: set[int] = set()
        stack = [s.nid for s in self.specs]
        while stack:
            nid = stack.pop()
            if nid in needed:
                continue
            needed.add(nid)
            stack.extend(self.nodes[nid].inputs)
        return sorted(name for name, lid in self.leaf_nid.items() if lid not in needed)

    def bind(self, *roots, tensor_types=None) -> int:
        """Bind every recorded device handle to its live slot(s) on ``roots``."""
        import ttnn

        t_types = (ttnn.Tensor,) if tensor_types is None else tensor_types
        spec_by_id = {id(s.handle): s for s in self.specs}
        for s in self.specs:
            s.sites.clear()
        self._tuple_bindings.clear()

        tuple_owners: dict[int, tuple[object, object, str, tuple]] = {}
        tuple_members: dict[int, list[tuple[int, _OutputSpec]]] = {}
        seen: set[int] = set()

        def walk(obj, parent=None, pkey=None, pkind=None, depth: int = 0):
            if depth > 14 or id(obj) in seen:
                return
            if isinstance(obj, t_types):
                spec = spec_by_id.get(id(obj))
                if spec is not None:
                    if pkind == "tuple":
                        tuple_members.setdefault(id(parent), []).append((pkey, spec))
                    else:
                        spec.sites.append((pkind, parent, pkey))
                return
            seen.add(id(obj))
            if isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, obj, i, "list", depth + 1)
            elif isinstance(obj, tuple):
                tuple_owners[id(obj)] = (parent, pkey, pkind, obj)
                for i, v in enumerate(obj):
                    walk(v, obj, i, "tuple", depth + 1)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, obj, k, "dict", depth + 1)
            elif hasattr(obj, "__dict__") and not isinstance(obj, (torch.Tensor, type)):
                for k, v in vars(obj).items():
                    if k in ("weights", "state_dict", "leaves", "sources", "consts"):
                        continue
                    walk(v, obj, k, "attr", depth + 1)

        for r in roots:
            walk(r)

        for tid, members in tuple_members.items():
            parent, pkey, pkind, orig = tuple_owners[tid]
            if pkind not in ("attr", "list", "dict"):
                raise RuntimeError(f"DeviceFolds.bind: nested tuple weight container unsupported ({pkind})")
            tb = {
                "parent": parent,
                "pkey": pkey,
                "pkind": pkind,
                "items": list(orig),
                "members": members,
            }
            self._tuple_bindings.append(tb)
            for idx, spec in members:
                spec.sites.append(("tuple_slot", tb, idx))

        missing = [s for s in self.specs if not s.sites]
        if missing:
            raise RuntimeError(
                f"DeviceFolds.bind: {len(missing)}/{len(self.specs)} recorded handles unreachable from roots "
                f"(first shape={missing[0].shape}, dtype={missing[0].dtype})")
        self._bound = True
        return len(self.specs)

    def _eval_node(self, node: _FoldNode, vals: dict[int, object], dev_masters: Mapping[str, object]):
        import ttnn

        op = node.op
        if op == "leaf":
            return dev_masters[node.attrs["name"]]
        if op == "clone":
            return _dev_clone_f32(vals[node.inputs[0]])
        if op == "t":
            return _dev_transpose_2d(vals[node.inputs[0]])
        if op == "slice":
            return _dev_slice_f32(
                vals[node.inputs[0]], node.attrs["dim"], node.attrs["start"], node.attrs["end"])
        if op == "cat":
            return _dev_concat_f32([vals[i] for i in node.inputs], node.attrs["dim"])
        if op == "mul_scalar":
            return ttnn.multiply(vals[node.inputs[0]], float(node.attrs["scalar"]))
        if op == "mul":
            return _dev_mul_f32(vals[node.inputs[0]], vals[node.inputs[1]], node.shape)
        if op == "view":
            return _dev_reshape_f32(vals[node.inputs[0]], node.attrs["shape"])
        if op == "expand":
            return _dev_expand_f32(vals[node.inputs[0]], node.attrs["shape"])
        if op == "scatter_slices":
            shape = node.attrs["shape"]
            writes = sorted(node.attrs["writes"], key=lambda w: w[1])
            dim = writes[0][0]
            assert all(w[0] == dim for w in writes), "scatter_slices across multiple dims unsupported"
            parts = []
            owned_zeros = []
            cur = 0
            for _d, start, end, src_nid in writes:
                if start > cur:
                    zshape = list(shape)
                    zshape[dim] = start - cur
                    z = _dev_zeros_f32(tuple(zshape), self.device)
                    parts.append(z)
                    owned_zeros.append(z)
                parts.append(vals[src_nid])
                cur = end
            if cur < shape[dim]:
                zshape = list(shape)
                zshape[dim] = shape[dim] - cur
                z = _dev_zeros_f32(tuple(zshape), self.device)
                parts.append(z)
                owned_zeros.append(z)
            out = _dev_concat_f32(parts, dim)
            for z in owned_zeros:
                ttnn.deallocate(z)
            return out
        raise NotImplementedError(f"DeviceFolds._eval_node: unknown op {op}")

    def refold(self, dev_masters: Mapping[str, object], *, on_refold=None) -> None:
        """Recompute every bound serving weight on card from ``dev_masters`` (0 PCIe bytes)."""
        import ttnn
        from .. import autograd as ag

        if not self._bound:
            raise RuntimeError("DeviceFolds.refold called before bind()")

        ag.release_pins()
        ag.forget_parameters()

        needed: set[int] = set()
        stack = [s.nid for s in self.specs]
        while stack:
            nid = stack.pop()
            if nid in needed:
                continue
            needed.add(nid)
            stack.extend(self.nodes[nid].inputs)

        rem: dict[int, int] = {nid: 0 for nid in needed}
        for nid in needed:
            for inp in self.nodes[nid].inputs:
                rem[inp] += 1
        specs_by_nid: dict[int, list[_OutputSpec]] = {}
        for spec in self.specs:
            specs_by_nid.setdefault(spec.nid, []).append(spec)
            rem[spec.nid] += 1

        vals: dict[int, object] = {}
        owned_val: set[int] = set()

        for nid in sorted(needed):
            node = self.nodes[nid]
            v = self._eval_node(node, vals, dev_masters)
            vals[nid] = v
            if node.op != "leaf":
                owned_val.add(nid)
            for inp in node.inputs:
                rem[inp] -= 1
                if rem[inp] == 0 and inp in owned_val:
                    ttnn.deallocate(vals.pop(inp))
                    owned_val.discard(inp)

            if nid in specs_by_nid:
                sp_list = specs_by_nid[nid]
                for spec in sp_list:
                    rem[nid] -= 1
                    # Transfer ownership directly if this is the final consumer of an owned fp32 intermediate.
                    if spec.dtype == ttnn.float32 and rem[nid] == 0 and nid in owned_val:
                        new_h = vals.pop(nid)
                        owned_val.discard(nid)
                    elif spec.dtype == ttnn.float32:
                        new_h = _dev_clone_f32(v)
                    else:
                        new_h = ttnn.typecast(v, spec.dtype)
                    if spec.memory_config is not None:
                        cur_mc = new_h.memory_config()
                        if cur_mc != spec.memory_config:
                            moved = ttnn.to_memory_config(new_h, spec.memory_config)
                            ttnn.deallocate(new_h)
                            new_h = moved
                    old_h = spec.handle
                    spec.handle = new_h
                    for kind, owner, key in spec.sites:
                        if kind == "attr":
                            setattr(owner, key, new_h)
                        elif kind in ("list", "dict"):
                            owner[key] = new_h
                        elif kind == "tuple_slot":
                            owner["items"][key] = new_h
                    ttnn.deallocate(old_h)
                    ag.parameter(new_h)
                if rem[nid] == 0 and nid in owned_val:
                    ttnn.deallocate(vals.pop(nid))
                    owned_val.discard(nid)

        for tb in self._tuple_bindings:
            new_tup = tuple(tb["items"])
            p, k, pk = tb["parent"], tb["pkey"], tb["pkind"]
            if pk == "attr":
                setattr(p, k, new_tup)
            else:
                p[k] = new_tup

        self.sources = {id(s.handle): (s.handle, None) for s in self.specs}

        if on_refold is not None:
            on_refold(dev_masters)

    def tape_backward(self, dev_masters: Mapping[str, object]) -> tuple[dict[str, object], list]:
        """Reverse-mode fold VJP on card from taped weight gradients (0 PCIe bytes).

        Consumes and frees every taped ``leaf.grad``. Returns ``(dev_grads, idle_handles)``
        where ``dev_grads`` maps upstream leaf name -> fp32 ``ttnn.Tensor`` on card.
        """
        import ttnn
        from .. import autograd as ag

        grads: dict[int, object] = {}
        idle: list = []

        def add_grad(nid: int, g):
            if nid not in grads:
                grads[nid] = g
            else:
                old = grads[nid]
                grads[nid] = ttnn.add(old, g)
                ttnn.deallocate(old)
                ttnn.deallocate(g)

        for spec in self.specs:
            leaf = ag.parameter_for(spec.handle)
            g = None if leaf is None else leaf.grad
            if g is None:
                idle.append(spec.handle)
                continue
            leaf.grad = None
            g32 = g if g.dtype == ttnn.float32 else ttnn.typecast(g, ttnn.float32)
            if g32 is not g:
                ttnn.deallocate(g)
            add_grad(spec.nid, g32)

        if not grads:
            return {}, idle

        active: set[int] = set()
        stack = list(grads.keys())
        while stack:
            nid = stack.pop()
            if nid in active:
                continue
            active.add(nid)
            stack.extend(self.nodes[nid].inputs)

        need_fwd: set[int] = set()
        for nid in active:
            node = self.nodes[nid]
            if node.op == "mul":
                need_fwd.update(node.inputs)
        if need_fwd:
            fstack = list(need_fwd)
            while fstack:
                fnid = fstack.pop()
                for inp in self.nodes[fnid].inputs:
                    if inp not in need_fwd:
                        need_fwd.add(inp)
                        fstack.append(inp)

        fvals: dict[int, object] = {}
        owned_fval: set[int] = set()
        for nid in sorted(need_fwd):
            node = self.nodes[nid]
            fvals[nid] = self._eval_node(node, fvals, dev_masters)
            if node.op != "leaf":
                owned_fval.add(nid)

        slice_consumers: dict[int, list[_FoldNode]] = {}
        for nid in active:
            node = self.nodes[nid]
            if node.op == "slice":
                slice_consumers.setdefault(node.inputs[0], []).append(node)

        fused_slice_parents: dict[int, tuple[int, list[_FoldNode]]] = {}
        fused_slice_nids: set[int] = set()
        for pid, snodes in slice_consumers.items():
            dims = {sn.attrs["dim"] for sn in snodes}
            if len(dims) != 1:
                continue
            dim = next(iter(dims))
            ordered = sorted(snodes, key=lambda sn: sn.attrs["start"])
            dim_sz = self.nodes[pid].shape[dim]
            cur = 0
            ok = True
            for sn in ordered:
                if sn.attrs["start"] != cur:
                    ok = False
                    break
                cur = sn.attrs["end"]
            if ok and cur == dim_sz and len(ordered) >= 2:
                fused_slice_parents[pid] = (dim, ordered)
                for sn in ordered:
                    fused_slice_nids.add(sn.nid)

        pending_slices: dict[int, dict[int, object]] = {pid: {} for pid in fused_slice_parents}

        for nid in sorted(active, reverse=True):
            g = grads.pop(nid, None)
            if g is None:
                continue
            node = self.nodes[nid]
            op = node.op

            if op == "leaf":
                grads[nid] = g
                continue

            if op == "clone":
                add_grad(node.inputs[0], g)
                continue

            if op == "t":
                dt = _dev_transpose_2d(g)
                ttnn.deallocate(g)
                add_grad(node.inputs[0], dt)
                continue

            if op == "view":
                dv = _dev_reshape_f32(g, self.nodes[node.inputs[0]].shape)
                ttnn.deallocate(g)
                add_grad(node.inputs[0], dv)
                continue

            if op == "expand":
                de = _dev_reduce_to_f32(g, self.nodes[node.inputs[0]].shape)
                ttnn.deallocate(g)
                add_grad(node.inputs[0], de)
                continue

            if op == "mul_scalar":
                ds = ttnn.multiply(g, float(node.attrs["scalar"]))
                ttnn.deallocate(g)
                add_grad(node.inputs[0], ds)
                continue

            if op == "mul":
                aid, bid = node.inputs
                av, bv = fvals[aid], fvals[bid]
                ga_full = _dev_mul_f32(g, bv, node.shape)
                gb_full = _dev_mul_f32(g, av, node.shape)
                ttnn.deallocate(g)
                da = _dev_reduce_to_f32(ga_full, self.nodes[aid].shape)
                ttnn.deallocate(ga_full)
                db = _dev_reduce_to_f32(gb_full, self.nodes[bid].shape)
                ttnn.deallocate(gb_full)
                add_grad(aid, da)
                add_grad(bid, db)
                continue

            if op == "cat":
                dim = node.attrs["dim"]
                off = 0
                for inp in node.inputs:
                    w = self.nodes[inp].shape[dim]
                    gi = _dev_slice_f32(g, dim, off, off + w)
                    add_grad(inp, gi)
                    off += w
                ttnn.deallocate(g)
                continue

            if op == "slice":
                pid = node.inputs[0]
                if nid in fused_slice_nids:
                    bucket = pending_slices[pid]
                    bucket[nid] = g
                    dim, ordered = fused_slice_parents[pid]
                    if len(bucket) == len(ordered):
                        pieces = [bucket.pop(sn.nid) for sn in ordered]
                        joined = _dev_concat_f32(pieces, dim)
                        for p in pieces:
                            ttnn.deallocate(p)
                        add_grad(pid, joined)
                else:
                    dim = node.attrs["dim"]
                    start, end = node.attrs["start"], node.attrs["end"]
                    pshape = self.nodes[pid].shape
                    parts = []
                    owned_z = []
                    if start > 0:
                        zs = list(pshape); zs[dim] = start
                        z = _dev_zeros_f32(tuple(zs), self.device)
                        parts.append(z); owned_z.append(z)
                    parts.append(g)
                    if end < pshape[dim]:
                        zs = list(pshape); zs[dim] = pshape[dim] - end
                        z = _dev_zeros_f32(tuple(zs), self.device)
                        parts.append(z); owned_z.append(z)
                    padded = _dev_concat_f32(parts, dim)
                    ttnn.deallocate(g)
                    for z in owned_z:
                        ttnn.deallocate(z)
                    add_grad(pid, padded)
                continue

            if op == "scatter_slices":
                for dim, start, end, src_nid in node.attrs["writes"]:
                    gi = _dev_slice_f32(g, dim, start, end)
                    add_grad(src_nid, gi)
                ttnn.deallocate(g)
                continue

            raise NotImplementedError(f"DeviceFolds.tape_backward: unknown op {op}")

        for pid, bucket in pending_slices.items():
            if not bucket:
                continue
            dim, ordered = fused_slice_parents[pid]
            pshape = self.nodes[pid].shape
            parts = []
            for sn in ordered:
                if sn.nid in bucket:
                    parts.append(bucket.pop(sn.nid))
                else:
                    zs = list(pshape); zs[dim] = sn.attrs["end"] - sn.attrs["start"]
                    parts.append(_dev_zeros_f32(tuple(zs), self.device))
            joined = _dev_concat_f32(parts, dim)
            for p in parts:
                ttnn.deallocate(p)
            add_grad(pid, joined)

        for fnid in owned_fval:
            ttnn.deallocate(fvals[fnid])

        out_grads: dict[str, object] = {}
        for name, lid in self.leaf_nid.items():
            g = grads.pop(lid, None)
            if g is not None:
                out_grads[name] = g
        return out_grads, idle


def make_rfd4_refold_hook(trunk, glue=None, *, host_glue_params: Mapping[str, torch.Tensor] | None = None,
                          pair_card_ref: list | None = None, device=None):
    """Build an ``on_refold(dev_masters)`` callback refreshing lazy serving caches on RFD4.

    Refreshes:
      1. ``TriangleMultiplication._t1_w`` (K14 lazy serving weights) strictly on card (0 PCIe bytes).
      2. ``TokenAttention._bias_w`` lazy fallback cleared if ever populated.
      3. Optional host-only ``glue`` parameters (cond_factory, init_repr_factory, pair_repr_builder;
         ~1.87 M non-pair-sized params = ~7.5 MB) synced from ``dev_masters`` so host glue and
         ``PairSplit``/``PairOnCard`` stay in lockstep with ``DeviceOptimizer``.
    """
    def _hook(dev_masters: Mapping[str, object]) -> None:
        import ttnn

        # 1. Refresh lazy TriangleMultiplication._t1_w on card (0 PCIe bytes)
        for idx, pu in trunk.pair_updates.items():
            for attr, tag in (("tri_out", "tri_mult_out"), ("tri_in", "tri_mult_in")):
                tri = getattr(pu, attr, None)
                if tri is None or not getattr(tri, "_t1_w", None):
                    continue
                pfx = (f"nn.pair_update_layers.{idx}.{tag}."
                       if f"nn.pair_update_layers.{idx}.{tag}.linear_a_p.weight" in dev_masters
                       else f"pair_update_layers.{idx}.{tag}.")
                pa = dev_masters.get(pfx + "linear_a_p.weight")
                pb = dev_masters.get(pfx + "linear_b_p.weight")
                ga = dev_masters.get(pfx + "linear_a_g.weight")
                gb = dev_masters.get(pfx + "linear_b_g.weight")
                go = dev_masters.get(pfx + "linear_g.weight")
                if any(t is None for t in (pa, pb, ga, gb, go)):
                    continue
                wp_f32 = _dev_concat_f32([pa, pb], 0)
                wg_f32 = _dev_concat_f32([ga, gb], 0)
                wgo_f32 = _dev_transpose_2d(go)
                for w8, old_triple in list(tri._t1_w.items()):
                    dt = ttnn.bfloat8_b if w8 else ttnn.bfloat16
                    wp_new = ttnn.typecast(wp_f32, dt)
                    wg_new = ttnn.typecast(wg_f32, dt)
                    wgo_new = ttnn.typecast(wgo_f32, ttnn.bfloat8_b) if w8 else None
                    tri._t1_w[w8] = (wp_new, wg_new, wgo_new)
                    for old_t in old_triple:
                        if old_t is not None:
                            ttnn.deallocate(old_t)
                ttnn.deallocate(wp_f32)
                ttnn.deallocate(wg_f32)
                ttnn.deallocate(wgo_f32)

        # 2. Clear lazy TokenAttention._bias_w fallback if ever created
        for blk in getattr(trunk, "blocks", ()):
            mha = getattr(getattr(blk, "mhba", None), "mha", None)
            bw = getattr(mha, "_bias_w", None) if mha is not None else None
            if bw is not None:
                ttnn.deallocate(bw)
                mha._bias_w = None

        # 3. Sync small host-only glue params (non-pair-sized) and refresh PairSplit/PairOnCard
        if host_glue_params:
            with torch.no_grad():
                for name, p in host_glue_params.items():
                    dm = dev_masters.get(name)
                    if dm is None:
                        continue
                    arr = ttnn.to_torch(dm).float().reshape(p.shape).to(p.dtype)
                    p.copy_(arr)
            if glue is not None and pair_card_ref is not None:
                from ..rfd4.pair import PairOnCard, PairSplit
                if pair_card_ref:
                    pair_card_ref.pop().release()
                glue.pair_split = PairSplit(glue.pair_repr_builder)
                dev_use = device if device is not None else trunk.device
                pair_card_ref.append(PairOnCard(glue.pair_split, dev_use))

    return _hook
