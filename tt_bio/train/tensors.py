"""Host <-> device transfer for the training stack. One PCIe read or write per call.

Tier 2. They are public because a Tier-2 user owns the ``for`` statement and therefore owns
when a tensor crosses the bus; a training loop that cannot say "read this back now" has to
guess, and the guess is usually a read per step it did not need.

``ttnn`` is imported inside the calls, not at module scope. That is what keeps every Tier-2
name that does not touch a device -- ``plan``, ``batches``, ``Mesh``, ``AdamW`` and its
refusals -- importable on a machine with no wheel and no card, which is Tier 0's cut line
holding one level down.
"""

from __future__ import annotations

__all__ = ["to_host", "to_device"]


def to_host(t, *, dtype=None):
    """Read a ttnn tensor to a float32 numpy array. One PCIe read."""
    import numpy as np
    import torch
    import ttnn
    out = ttnn.to_torch(t).to(torch.float32).numpy()
    return out.astype(dtype) if dtype is not None else np.ascontiguousarray(out)


def to_device(arr, device, *, dtype=None, layout=None):
    """Push a numpy array to the device, rounding to ``dtype``. One PCIe write.

    ``dtype`` and ``layout`` default to ``ttnn.bfloat16`` and ``ttnn.TILE_LAYOUT``, resolved
    here rather than in the signature so importing this module does not need the wheel.
    """
    import torch
    import ttnn
    return ttnn.from_torch(torch.from_numpy(arr).to(torch.float32),
                           dtype=ttnn.bfloat16 if dtype is None else dtype,
                           layout=ttnn.TILE_LAYOUT if layout is None else layout,
                           device=device)

