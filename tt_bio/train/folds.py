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

__all__ = ["Folds", "DeviceFolds"]


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

def _is_aligned_for_tile(shape: tuple[int, ...], dim: int, start: int, end: int) -> bool:
    """True when a slice/concat along ``dim`` on a TILE tensor lands on exact 32-element tile edges."""
    rank = len(shape)
    d = dim if dim >= 0 else rank + dim
    if d < rank - 2:
        return True
    # Last two axes in TILE layout require 32-aligned start/end (and full-tile height/width).
    return (start % 32 == 0) and (end % 32 == 0) and (shape[d] % 32 == 0)


def _dev_transpose_2d(x):
    """Bit-exact 2-D float32 transpose on Tenstorrent via uint32 bitcast.

    Plain ``ttnn.transpose`` on float32 TILE tensors rounds to ~TF32 on Blackhole.
    Never deallocate the bitcast alias directly; only deallocate the newly created output.
    """
    import ttnn
    from .device_optim import transpose_exact_f32
    return transpose_exact_f32(x)


def _to_row_major_f32(x):
    import ttnn
    if x.layout == ttnn.ROW_MAJOR_LAYOUT:
        return x, False
    # Bit-exact uint32 untilize preserves all 32 float32 bits without rounding.
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
    """Exact fp32 slice ``x[..., start:end, ...]`` on card, supporting non-32-aligned axes."""
    import ttnn
    shape = tuple(int(s) for s in x.shape)
    rank = len(shape)
    d = dim if dim >= 0 else rank + dim
    beg = [0] * rank
    fin = list(shape)
    beg[d] = int(start)
    fin[d] = int(end)
    if _is_aligned_for_tile(shape, d, beg[d], fin[d]):
        # Bit-exact in TILE via uint32? Wait: ttnn.slice in TILE on fp32 copies tiles intact.
        return ttnn.slice(x, beg, fin)
    # Non-32-aligned axis (e.g. bias_ws dim=1 width 12/24): slice in ROW_MAJOR, re-tilize.
    rm, rm_owner = _to_row_major_f32(x)
    s_rm = ttnn.slice(rm, beg, fin)
    if rm_owner is not False:
        ttnn.deallocate(rm_owner)
    tl, _ = _to_tile_f32(s_rm)
    ttnn.deallocate(s_rm)
    return tl


def _dev_concat_f32(xs: list, dim: int):
    """Exact fp32 concat along ``dim`` on card, supporting non-32-aligned widths."""
    import ttnn
    if len(xs) == 1:
        return ttnn.clone(xs[0])
    shape0 = tuple(int(s) for s in xs[0].shape)
    rank = len(shape0)
    d = dim if dim >= 0 else rank + dim
    aligned = all(
        (d < rank - 2) or (int(t.shape[d]) % 32 == 0)
        for t in xs
    )
    if aligned:
        return ttnn.concat(xs, dim=d)
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
    import ttnn
    cur = tuple(int(s) for s in x.shape)
    tgt = tuple(int(s) for s in target_shape)
    if cur == tgt:
        return ttnn.clone(x)
    # 1-D <-> 2-D/4-D reshape in TILE can reject mismatched tile-padded dims; ROW_MAJOR is always exact.
    rm, rm_owner = _to_row_major_f32(x)
    r_rm = ttnn.reshape(rm, list(tgt))
    if rm_owner is not False:
        # r_rm may alias rm_owner if view; clone to tile before freeing rm_owner.
        tl, _ = _to_tile_f32(r_rm)
        ttnn.deallocate(rm_owner)
        return tl
    tl, _ = _to_tile_f32(r_rm)
    return tl


def _dev_expand_f32(x, target_shape: tuple[int, ...]):
    import ttnn
    cur = tuple(int(s) for s in x.shape)
    tgt = tuple(int(s) for s in target_shape)
    if cur == tgt:
        return ttnn.clone(x)
    assert len(cur) == len(tgt), (cur, tgt)
    reps = [t // c for c, t in zip(cur, tgt)]
    return ttnn.repeat(x, ttnn.Shape(reps))


def _dev_broadcast_shape(t, target_shape: tuple[int, ...]):
    """Reshape leading 1s onto ``t`` so its rank matches ``target_shape`` for binary ops."""
    cur = tuple(int(s) for s in t.shape)
    if len(cur) == len(target_shape):
        return t, False
    padded = (1,) * (len(target_shape) - len(cur)) + cur
    r = _dev_reshape_f32(t, padded)
    return r, True


def _dev_reduce_to_f32(g, target_shape: tuple[int, ...]):
    """Sum ``g`` over any broadcast / leading axes to match ``target_shape`` exactly."""
    import ttnn
    gshape = tuple(int(s) for s in g.shape)
    tgt = tuple(int(s) for s in target_shape)
    if gshape == tgt:
        return ttnn.clone(g)
    pad_len = len(gshape) - len(tgt)
    padded_tgt = (1,) * pad_len + tgt
    cur = g
    owned = False
    # Sum leading axes + axes where padded_tgt == 1 and gshape > 1
    dims = [i for i, (gs, ts) in enumerate(zip(gshape, padded_tgt)) if gs != ts and ts == 1]
    if dims:
        # Sum one axis at a time from highest dim down or all at once via keepdim=True.
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
        cur = ttnn.clone(cur)
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

        # Pin every traced tensor for the duration of recording so Python never reuses an id().
        keepalive: list[torch.Tensor] = []
        node_of: dict[int, int] = {}
        # zero_bufs[tid] = (root_zero_tid, shape); alias_group[root_zero_tid] = set of all tids sharing that zero buf
        zero_bufs: dict[int, tuple[int, tuple[int, ...]]] = {}
        zero_aliases: dict[int, set[int]] = {}
        # slice_views[id(view)] = (root_zero_tid, dim, start, end)
        slice_views: dict[int, tuple[int, int, int, int]] = {}
        # scatter_writes[root_zero_tid] = list of (dim, start, end, src_nid)
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

                # Track fresh zero buffers (for dit_pair_bias.rms_weight copy_ scatters)
                if opname == "aten.zeros.default" and isinstance(out, torch.Tensor):
                    tid = id(out)
                    zero_bufs[tid] = (tid, tuple(int(s) for s in out.shape))
                    zero_aliases[tid] = {tid}
                    keepalive.append(out)
                    return out

                # Track pass-through / alias / detach / dtype cast ops
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

                if opname in ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"):
                    x = args[0]
                    if isinstance(x, torch.Tensor) and id(x) in node_of and isinstance(out, torch.Tensor):
                        nid = outer._add_node(
                            "view", (node_of[id(x)],), {"shape": tuple(int(s) for s in out.shape)},
                            tuple(out.shape),
                        )
                        pin(out, nid)
                    return out

                if opname == "aten.unsqueeze.default":
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

                # Catch any unexpected tracked op immediately rather than silently corrupting the DAG.
                def _any_tracked(v):
                    if isinstance(v, torch.Tensor):
                        return id(v) in node_of
                    if isinstance(v, (list, tuple)):
                        return any(_any_tracked(u) for u in v)
                    return False

                if isinstance(out, torch.Tensor) and _any_tracked(args):
                    # Ignore pure inspection/constant builders like zeros_like/empty_like
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

    def bind(self, *roots, tensor_types=None) -> int:
        """Bind every recorded device handle to its live slot(s) on ``roots``.

        Handles object attributes, lists, dicts, and tuples (by replacing the tuple on its
        owning attribute/list/dict). Raises if any recorded handle is unreachable.
        """
        import ttnn

        t_types = (ttnn.Tensor,) if tensor_types is None else tensor_types
        spec_by_id = {id(s.handle): s for s in self.specs}
        for s in self.specs:
            s.sites.clear()
        self._tuple_bindings.clear()

        # Map id(tuple) -> (parent, pkey, kind) so tuples containing weights can be rebuilt cleanly.
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
            return ttnn.clone(vals[node.inputs[0]])
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
            a = vals[node.inputs[0]]
            b = vals[node.inputs[1]]
            a_b, a_own = _dev_broadcast_shape(a, node.shape)
            b_b, b_own = _dev_broadcast_shape(b, node.shape)
            out = ttnn.multiply(a_b, b_b)
            if a_own:
                ttnn.deallocate(a_b)
            if b_own:
                ttnn.deallocate(b_b)
            return out
        if op == "view":
            return _dev_reshape_f32(vals[node.inputs[0]], node.attrs["shape"])
        if op == "expand":
            return _dev_expand_f32(vals[node.inputs[0]], node.attrs["shape"])
        if op == "scatter_slices":
            # Build zero-padded tensor along dim from non-overlapping slice writes (e.g. rms_weight).
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

        # Determine needed nodes and ref-counts for immediate deallocation of intermediates.
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
        owned_val: set[int] = set()  # True for intermediates we created (never dev_masters!)

        for nid in sorted(needed):
            node = self.nodes[nid]
            v = self._eval_node(node, vals, dev_masters)
            vals[nid] = v
            if node.op != "leaf":
                owned_val.add(nid)
            # Decrement input refcounts and free consumed intermediates immediately.
            for inp in node.inputs:
                rem[inp] -= 1
                if rem[inp] == 0 and inp in owned_val:
                    ttnn.deallocate(vals.pop(inp))
                    owned_val.discard(inp)

            if nid in specs_by_nid:
                for spec in specs_by_nid[nid]:
                    if spec.dtype == ttnn.float32:
                        new_h = ttnn.clone(v)
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
                rem[nid] -= 1
                if rem[nid] == 0 and nid in owned_val:
                    ttnn.deallocate(vals.pop(nid))
                    owned_val.discard(nid)

        # Write back any updated tuples atomically.
        for tb in self._tuple_bindings:
            new_tup = tuple(tb["items"])
            p, k, pk = tb["parent"], tb["pkey"], tb["pkind"]
            if pk == "attr":
                setattr(p, k, new_tup)
            else:
                p[k] = new_tup

        # Refresh self.sources keys to current handles
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

        # Find all nodes reachable backwards from active gradients.
        active: set[int] = set()
        stack = list(grads.keys())
        while stack:
            nid = stack.pop()
            if nid in active:
                continue
            active.add(nid)
            stack.extend(self.nodes[nid].inputs)

        # Forward-evaluate ONLY the operands needed by active `mul` nodes (AdaLN / pair_norm).
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

        # Group contiguous full-cover slice consumers per parent node so QKV / SwiGLU /
        # PairUpdate / trimul chunks reassemble in a single ttnn.concat instead of N pad+adds.
        slice_consumers: dict[int, list[_FoldNode]] = {}
        all_consumers: dict[int, list[_FoldNode]] = {}
        for nid in active:
            node = self.nodes[nid]
            for inp in node.inputs:
                all_consumers.setdefault(inp, []).append(node)
            if node.op == "slice":
                slice_consumers.setdefault(node.inputs[0], []).append(node)

        fused_slice_parents: dict[int, tuple[int, list[_FoldNode]]] = {}
        fused_slice_nids: set[int] = set()
        for pid, snodes in slice_consumers.items():
            # Only fuse when all active slice consumers share dim and partition [0, dim_sz) cleanly.
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

        # Pending slice gradients waiting for their sibling chunks before one concat.
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
                ab, a_own = _dev_broadcast_shape(av, node.shape)
                bb, b_own = _dev_broadcast_shape(bv, node.shape)
                ga_full = ttnn.multiply(g, bb)
                gb_full = ttnn.multiply(g, ab)
                ttnn.deallocate(g)
                if a_own:
                    ttnn.deallocate(ab)
                if b_own:
                    ttnn.deallocate(bb)
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

        # Flush any partial fused-slice buckets (in case a sibling slice had no grad this step).
        for pid, bucket in pending_slices.items():
            if not bucket:
                continue
            dim, ordered = fused_slice_parents[pid]
            pshape = self.nodes[pid].shape
            parts, owned_z = [], []
            for sn in ordered:
                if sn.nid in bucket:
                    parts.append(bucket.pop(sn.nid))
                else:
                    zs = list(pshape); zs[dim] = sn.attrs["end"] - sn.attrs["start"]
                    z = _dev_zeros_f32(tuple(zs), self.device)
                    parts.append(z); owned_z.append(z)
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
