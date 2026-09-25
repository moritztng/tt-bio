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
"""

from __future__ import annotations

import contextlib
import weakref

import torch

__all__ = ["recording", "Lineage"]


class _Traced(torch.Tensor):
    """A tensor that remembers which checkpoint keys its value was computed from."""

    keys: frozenset = frozenset()
    _live: "weakref.WeakSet" = weakref.WeakSet()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        keys = frozenset().union(*(t.keys for t in _tensors((args, kwargs))))
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
    """``made``: device-tensor handle -> (tensor, keys) for every upload during the build."""

    def __init__(self):
        self.made = {}

    def add(self, t, keys):
        self.made[_handle(t)] = (t, keys)

    def by_path(self, walked) -> dict:
        """``{path: keys}`` over ``walk_device_weights`` output, then forget the uploads.

        The uploads are held until here so neither a buffer nor an id can be reused by a later
        tensor (ttnn tensors take no weak reference); a buffer deallocated during the build
        does not count.
        """
        out = {}
        for p, _o, _k, t in walked:
            h = _handle(t)
            got = self.made.get(h)
            if got and (got[0] is t if h[0] == "object" else got[0].is_allocated()):
                out[p] = got[1]
        self.made = {}
        return out


@contextlib.contextmanager
def recording(state_dict: dict, canonical=lambda k: k):
    """Yield ``(traced_state_dict, Lineage)``; build the model from the traced dict inside.

    ``canonical`` names the parameter a key is a copy of (a checkpoint may store one module
    under two prefixes). On exit ``ttnn.from_torch`` is the shipped function again and every
    traced tensor the model kept is demoted to a plain ``torch.Tensor``, so nothing after the
    build pays for the tracing.
    """
    import ttnn

    lin = Lineage()
    shipped = ttnn.from_torch

    def from_torch(tensor, *args, **kwargs):
        keys = tensor.keys if isinstance(tensor, _Traced) else None
        if keys is not None:
            tensor = tensor.as_subclass(torch.Tensor)
        out = shipped(tensor, *args, **kwargs)
        if keys:
            lin.add(out, keys)
        return out

    traced = {k: _mark(v, frozenset({canonical(k)})) if torch.is_tensor(v) else v
              for k, v in state_dict.items()}
    ttnn.from_torch = from_torch
    try:
        yield traced, lin
    finally:
        ttnn.from_torch = shipped
        for t in list(_Traced._live):
            t.__class__ = torch.Tensor
        _Traced._live = weakref.WeakSet()
