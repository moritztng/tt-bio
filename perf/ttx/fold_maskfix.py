#!/usr/bin/env python3
"""Fold the 298 aa control on a post-0.70.1 stack with the old attn_mask convention restored.

ttnn 0.70.1 changed what `ttnn.transformer.scaled_dot_product_attention` does with `attn_mask`:
0.69.0 and earlier compute `softmax((QK^T + mask) * scale) @ V`, 0.70.1 and later compute
`softmax(QK^T * scale + mask) @ V`, which is torch's convention. Measured on the fold's own call
346 operands (`sdpa_mask_convention.py`): each arm sits within 2e-2 relative L2 of its own
convention and 3.6e-1 from the other.

tt-bio pre-multiplies the pair bias by `_bias_scale = head_dim**0.5` (`tenstorrent.py:7578`)
exactly to cancel the old convention's scaling of the mask, so on 0.70.1 the bias arrives
sqrt(32) = 5.657x too large. Scaling the mask by `scale` before the call restores the old
identity, `softmax(QK^T*scale + mask*scale) == softmax((QK^T + mask)*scale)`, without touching
model code: the padding sentinel goes from -9.98e8 to -1.76e8, still saturating the softmax.

Blanket, so only valid where every SDPA site pre-scales its bias. Boltz-2 does
(`scale_pair_bias=True`); an OF3 site with `_bias_scale = 1.0` would need the per-site form.

    fold_maskfix.py <same arguments as perf/roof_shared/fold_shared.py>
"""
import runpy
import sys
from pathlib import Path

import ttnn

REPO = Path(__file__).resolve().parents[2]
_orig = ttnn.transformer.scaled_dot_product_attention


def _shim(*a, **kw):
    mask, scale = kw.get("attn_mask"), kw.get("scale")
    if mask is None or scale is None:
        return _orig(*a, **kw)
    scaled = ttnn.multiply(mask, scale)
    kw["attn_mask"] = scaled
    try:
        return _orig(*a, **kw)
    finally:
        ttnn.deallocate(scaled)


ttnn.transformer.scaled_dot_product_attention = _shim
sys.argv = [str(REPO / "perf/roof_shared/fold_shared.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
