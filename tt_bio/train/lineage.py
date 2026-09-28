"""Which checkpoint keys each device weight of a built model was made from.

A training step trains what reaches a registered leaf. ``check_registered`` asks that of the
device tensors a model holds, and a module whose weights never become device tensors, because
the forward applies them on the host straight from the checkpoint, leaves it nothing to ask:
nothing is late and nothing exists. D261 (the diffusion encoder's ref-atom embedder) and D263
(the input embedder's whole atom encoder, 93 tensors) were both that. So the question is also
asked from the checkpoint's side: every key the step claims to train must have been uploaded
into a registered leaf.

The answer is recorded rather than matched. While the model is built, every checkpoint tensor
carries the set of keys it came from through every torch op (``_Traced``), and
``ttnn.from_torch`` files that set against the device tensor it returns. Fusing, transposing,
padding and scaling keep the set; a value matcher would need to know each rule.

Recorded ``differentiable``, the same build also answers the question in reverse: which
checkpoint values a moved device weight corresponds to. Every checkpoint tensor becomes an
autograd leaf, each upload keeps the host tensor it was made from, and ``fold_back`` pulls a
device-side movement back through the build's own ops. That is how a trained model is written
out as the checkpoint ``tt-bio predict`` loads, without a per-module rule for how each weight
was transposed, fused or padded on its way to the card.
"""

from __future__ import annotations

import contextlib
import weakref

import torch

__all__ = ["recording", "Lineage", "fold_back"]

#: Reads that leave the tensor world. Autograd refuses them on a tensor that requires grad, and
#: they end the trace anyway, so a differentiable build detaches first.
_READS = (torch.Tensor.numpy, torch.Tensor.tolist, torch.Tensor.item)


class _Traced(torch.Tensor):
    """A tensor that remembers which checkpoint keys its value was computed from."""

    keys: frozenset = frozenset()
    _live: "weakref.WeakSet" = weakref.WeakSet()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        keys = frozenset().union(*(t.keys for t in _tensors((args, kwargs))))
        if func in _READS:
            args = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args)
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*args, **kwargs)
        return _mark(out, keys)


def _tensors(x):
    if isinstance(x, _Traced):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


def _mark(x, keys):
    if isinstance(x, torch.Tensor):
        t = x.as_subclass(_Traced)
        t.keys = keys
        _Traced._live.add(t)
        return t
    if isinstance(x, (list, tuple)):
        return type(x)(_mark(y, keys) for y in x)
    return x


def _handle(t):
    """What makes two ttnn tensors the same weight: the device buffer, else the object.

    The buffer, because an in-place op returns a new wrapper around the same one: a pair-bias
    projection is uploaded and then ``ttnn.multiply_``-scaled, and the model keeps the result.
    """
    try:
        return ("buffer", t.buffer_address())
    except RuntimeError:
        return ("object", id(t))


class Lineage:
    """``made``: device-tensor handle -> (tensor, keys, host) for every upload during the build.

    ``host`` is the torch tensor the upload was made from, kept only by a differentiable
    recording, and ``leaves`` is then ``{checkpoint key: autograd leaf}``.
    """

    def __init__(self):
        self.made = {}
        self.leaves = {}

    def add(self, t, keys, host=None):
        self.made[_handle(t)] = (t, keys, host)

    def by_path(self, walked, *, hosts: bool = False) -> dict:
        """``{path: keys}`` over ``walk_device_weights`` output, then forget the uploads.

        ``hosts=True`` maps each path to ``(keys, host)`` instead, for ``fold_back``.

        The uploads are held until here so neither a buffer nor an id can be reused by a later
        tensor (ttnn tensors take no weak reference); a buffer deallocated during the build
        does not count.
        """
        out = {}
        for p, _o, _k, t in walked:
            h = _handle(t)
            got = self.made.get(h)
            if got and (got[0] is t if h[0] == "object" else got[0].is_allocated()):
                out[p] = (got[1], got[2]) if hosts else got[1]
        self.made = {}
        return out


@contextlib.contextmanager
def recording(state_dict: dict, canonical=lambda k: k, *, differentiable: bool = False):
    """Yield ``(traced_state_dict, Lineage)``; build the model from the traced dict inside.

    ``canonical`` names the parameter a key is a copy of (a checkpoint may store one module
    under two prefixes). On exit ``ttnn.from_torch`` is the shipped function again and every
    traced tensor the model kept is demoted to a plain ``torch.Tensor``, so nothing after the
    build pays for the tracing.

    ``differentiable`` makes every floating-point checkpoint tensor an autograd leaf
    (``Lineage.leaves``) and keeps each upload's host tensor, for ``fold_back``. The model
    handed each key a clone of its leaf, so an in-place op in a build is recorded rather than
    refused.
    """
    import ttnn

    lin = Lineage()
    shipped = ttnn.from_torch

    def from_torch(tensor, *args, **kwargs):
        keys = tensor.keys if isinstance(tensor, _Traced) else None
        host = None
        if keys is not None:
            tensor = tensor.as_subclass(torch.Tensor)
            if tensor.requires_grad:
                host, tensor = tensor, tensor.detach()
        out = shipped(tensor, *args, **kwargs)
        if keys:
            lin.add(out, keys, host)
        return out

    def trace(k, v):
        if differentiable and v.is_floating_point():
            lin.leaves[k] = v.detach().clone().requires_grad_(True)
            v = lin.leaves[k].clone()
        return _mark(v, frozenset({canonical(k)}))

    traced = {k: trace(k, v) if torch.is_tensor(v) else v for k, v in state_dict.items()}
    ttnn.from_torch = from_torch
    try:
        yield traced, lin
    finally:
        ttnn.from_torch = shipped
        for t in list(_Traced._live):
            t.__class__ = torch.Tensor
        _Traced._live = weakref.WeakSet()


def fold_back(state_dict: dict, lineage: Lineage, uploads: dict, moved: dict,
              canonical=lambda k: k, *, tol: float = 1e-3) -> dict:
    """``state_dict`` with each upload's movement pulled back into the keys it was built from.

    ``uploads`` is ``Lineage.by_path(walked, hosts=True)`` from a differentiable recording and
    ``moved`` is ``{path: new - old}`` in device coordinates, the shape the device tensor reads
    back as. The build is linear in the checkpoint for a weight (transposes, concatenations,
    padding, casts, constant scales), so a key moves by the least-squares preimage
    ``J^T d / J^T J 1`` of what its uploads moved, which is exact when each value reaches the
    card once. ``J^T J 1`` comes from a double backward, so no rule about any one transform is
    written down anywhere.

    Keys that are copies of one parameter (``canonical``) move together. Every moved upload is
    then checked forward: rebuilding it from the returned dict must reproduce its movement to
    ``tol`` relative, else this raises naming it. That catches a value that reached the card
    twice and trained apart, which one checkpoint cannot represent, and an upload with no path
    back to its keys at all. An upload that did not move is not checked: it got no gradient, so
    it is a view the trained forward never read, and it follows the one it did. The returned tensors keep their stored dtype; untouched keys are
    the same objects.
    """
    orphan = sorted(p for p in moved if p not in uploads and torch.as_tensor(moved[p]).any())
    if orphan:
        raise ValueError(f"{len(orphan)} device weights moved but were not uploaded from the "
                         f"checkpoint, so nothing can carry their change, e.g. {orphan[:4]}")
    leaves = lineage.leaves
    members = {}
    for k in leaves:
        members.setdefault(canonical(k), []).append(k)

    def sources(keys):
        return [k for c in sorted(keys) for k in members.get(c, ())]

    def pullback(host, xs, d):
        return torch.autograd.grad(host, xs, d, retain_graph=True, allow_unused=True)

    def forward(host, xs, v):
        """``J v``: differentiate the (linear-in-u) ``J^T u`` with respect to u."""
        u = torch.zeros_like(host, requires_grad=True)
        jt = torch.autograd.grad(host, xs, u, create_graph=True, allow_unused=True)
        pairs = [(g, x) for g, x in zip(jt, v) if g is not None and g.requires_grad]
        if not pairs:
            return torch.zeros_like(host)
        (out,) = torch.autograd.grad([g for g, _ in pairs], u, [x for _, x in pairs],
                                     retain_graph=True)
        return out

    num, den = {}, {}
    for path, (keys, host) in uploads.items():
        d = moved.get(path)
        if d is None or not torch.as_tensor(d).any():
            continue
        xs_k = sources(keys)
        if host is None or host.grad_fn is None or not xs_k:
            raise ValueError(f"{path} moved but has no differentiable path back to "
                             f"{sorted(keys)}; it cannot be written as a checkpoint")
        xs = [leaves[k] for k in xs_k]
        d = torch.as_tensor(d).reshape(host.shape).to(host.dtype)
        g = pullback(host, xs, d)
        n = pullback(host, xs, forward(host, xs, [torch.ones_like(x) for x in xs]))
        for k, gk, nk in zip(xs_k, g, n):
            if gk is None:
                continue
            c = canonical(k)
            num[c] = num.get(c, 0) + gk.double()
            den[c] = den.get(c, 0) + nk.double()

    step = {c: torch.where(den[c] > 0, num[c] / den[c].clamp_min(1e-300), 0.0)
            for c in num}
    out = dict(state_dict)
    for c, s in step.items():
        for k in members[c]:
            out[k] = (state_dict[k].double() + s).to(state_dict[k].dtype)

    worst = []
    for path, (keys, host) in uploads.items():
        # An upload the run never moved had no gradient: the forward that trained never read
        # it. It is a second view of weights another upload carries (triangle attention holds
        # q/k/v/g as `qkv`+`g`, fused `qkvg` and `qkvgb`, and uses one per call), and rebuilt
        # from the result it carries the trained values like the view that was read.
        if path not in moved:
            continue
        xs_k = [k for k in sources(keys) if canonical(k) in step]
        d = torch.as_tensor(moved[path], dtype=torch.float64)
        if not xs_k:
            worst.append((path, 1.0))
            continue
        got = forward(host, [leaves[k] for k in xs_k],
                      [step[canonical(k)].to(leaves[k].dtype) for k in xs_k]).double()
        d = d.reshape(got.shape)
        err = float((got - d).norm() / max(float(d.norm()), float(got.norm()), 1e-30))
        # A build that rounds on the host before uploading rounds the movement with it.
        if err > max(tol, 4 * torch.finfo(host.dtype).eps):
            worst.append((path, err))
    if worst:
        worst.sort(key=lambda w: -w[1])
        raise ValueError(f"{len(worst)} device weights cannot be written back to the checkpoint "
                         f"they were built from (relative residual above {tol}), e.g. "
                         f"{worst[:4]}")
    return out
