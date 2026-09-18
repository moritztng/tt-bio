"""One dispatch slot, and the decorator that uses it. Imports nothing.

Two op surfaces hold a slot today -- `tt_bio/ops.py` for the linear and layer norm every
device module makes, and `tt_bio/abodybuilder3_ops.py` for the matmul, eltwise, reduction and
shape set a geometry model needs -- and `train-a1-defork` and `train-b2-abb3-port` built the
same nineteen lines of machinery independently before either could see the other. A3 would
have written them a third time. This is those lines, once.

The SLOT stays per surface and that is deliberate, not an oversight: one global hook would
mean installing a tape for ABodyBuilder3 also tapes Protenix. What is shared is the mechanism,
not the state.

The hook signature is `hook(name, shipped, args, kwargs)`, returning `None` to decline. Two
properties follow from it and both are load-bearing.

* It scales with the op set instead of against it. A hook exposing named methods needs a new
  method in every implementation for every new op; this one needs none.
* It hands `shipped` over, so a taped op computes its forward BY CALLING the production
  function. That makes "there is one forward" structural rather than conventional, and it is
  what produces a bit-exact grad-off-against-grad-on column instead of a nearly-equal one.

Declining is how a hook handles only the operands it tracks, and also how a taped module runs
the shipped op for a tensor that is on the tape but not being differentiated -- a frozen
block, or anything under `no_grad`.
"""

from __future__ import annotations

import functools

__all__ = ["OpSurface"]


class OpSurface:
    """The grad-hook slot for one module's worth of ops."""

    __slots__ = ("name", "_hook")

    def __init__(self, name: str):
        self.name = name
        self._hook = None

    def set_grad_hook(self, hook):
        """Install a differentiable implementation, or clear it with ``None``.

        Returns the previous hook, so a caller can restore it.
        """
        prev, self._hook = self._hook, hook
        return prev

    def grad_hook(self):
        return self._hook

    def dispatching(self, fn):
        """Offer the call to the hook first, then run ``fn``, which is the production op.

        With no hook installed -- every inference path, always -- the decorated op is the
        ``ttnn`` call it was written as plus one ``is None`` test. The undecorated function
        stays reachable as ``<op>.shipped`` for a caller that must bypass the hook.
        """
        name = fn.__name__

        @functools.wraps(fn)
        def call(*args, **kwargs):
            hook = self._hook
            if hook is not None:
                out = hook(name, fn, args, kwargs)
                if out is not None:
                    return out
            return fn(*args, **kwargs)

        call.shipped = fn
        return call
