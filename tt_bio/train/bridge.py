"""Host torch <-> device tape, for a model whose glue runs on the host around a taped device trunk.

RFD4 serves its conditioning, initial representations, pair builder and heads as host torch
(`rfd4/network.py`, ~3.2 M of 175 M parameters, run once a pass) around the device trunk. Training
keeps that split rather than porting the glue: the host half is the served forward and torch's own
autograd differentiates it, the device half is the served forward and `tt_bio.autograd` tapes it. A
`Bridge` joins the two graphs at the tensors that cross:

* `up(h)`   a host tensor entering the trunk becomes a device leaf; its gradient flows back into
            `h`'s torch graph.
* `down(t)` a trunk output read by the host becomes a torch leaf; its `.grad` seeds the tape.
* `backward(loss)` runs the three halves in order: host loss to the `down` leaves, one tape replay
            seeded by all of them, then the `up` gradients into the host graph.

A head that reads `s.detach()` upstream reads `bridge.down(s).detach()` here, so its loss stops at
the host exactly as it does there. `HostParam` makes a host `torch.nn.Parameter` an optimizer leaf, so
one `AdamW` owns both halves and the global gradient norm it clips is the whole model's.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["Bridge", "HostParam", "host_params"]


class HostParam:
    """A host torch parameter as an `optim.AdamW` leaf: `.value` and `.grad` read the parameter, and the
    optimizer writes the stepped master back in place through `write`, so the module keeps its tensor."""

    def __init__(self, p: torch.Tensor):
        self.p = p

    @property
    def value(self):
        return self.p.detach()

    @property
    def grad(self):
        return self.p.grad

    @grad.setter
    def grad(self, g):
        if g is not None and not isinstance(g, torch.Tensor):
            g = torch.from_numpy(np.asarray(g, np.float32)).reshape(self.p.shape)
        self.p.grad = None if g is None else g.to(self.p.dtype)

    def write(self, theta: np.ndarray) -> None:
        with torch.no_grad():
            self.p.copy_(torch.from_numpy(theta).reshape(self.p.shape))


def host_params(module: torch.nn.Module, prefix: str = "") -> dict:
    """`{name: HostParam}` for every trainable parameter of a host module."""
    return {prefix + n: HostParam(p) for n, p in module.named_parameters() if p.requires_grad}


class Bridge:
    """The crossings of one taped pass. Build one per pass; `backward` consumes it."""

    def __init__(self, device, dtype=None):
        import ttnn
        self.device, self.dtype = device, (ttnn.bfloat16 if dtype is None else dtype)
        self.ups, self.downs = [], []

    def up(self, h: torch.Tensor, dtype=None):
        """`h` on the device as a tape leaf. Wants a gradient exactly when `h` does."""
        import ttnn
        from .. import autograd as ag
        v = ttnn.from_torch(h.detach().to(torch.float32), dtype=dtype or self.dtype,
                            layout=ttnn.TILE_LAYOUT, device=self.device)
        leaf = ag.Tensor(v, requires_grad=h.requires_grad)
        if h.requires_grad:
            self.ups.append((h, leaf))
        return leaf

    def down(self, t) -> torch.Tensor:
        """A trunk output on the host as a float32 torch leaf whose gradient will seed the tape."""
        import ttnn
        h = ttnn.to_torch(t.value).to(torch.float32).requires_grad_(t.requires_grad)
        if t.requires_grad:
            self.downs.append((t, h))
        return h

    def backward(self, loss: torch.Tensor) -> None:
        import ttnn
        from .. import autograd as ag
        # retain_graph: a host tensor can feed both the loss and an `up`, and the third half replays it.
        loss.backward(retain_graph=bool(self.ups))
        seeded = [(t, h.grad) for t, h in self.downs if h.grad is not None]
        if seeded:
            ag.backward([t for t, _ in seeded],
                        [ttnn.from_torch(g.contiguous(), dtype=t.value.dtype, layout=ttnn.TILE_LAYOUT,
                                         device=self.device) for t, g in seeded])
        back = [(h, ttnn.to_torch(leaf.grad).to(h.dtype).reshape(h.shape))
                for h, leaf in self.ups if leaf.grad is not None]
        if back:
            torch.autograd.backward([h for h, _ in back], [g for _, g in back])
        self.ups, self.downs = [], []
