"""Does `ttnn.scale_mask_softmax` take RFD3's per-row attention bias, on this wheel?

Memory (`ttnn-scale-mask-softmax-and-widen-in-binary-op`, from OpenFold3, and RFD3 p30) records that
the whole family asserts `mask.padded_shape()[-2] in {1, tile_height}`, i.e. a row-broadcast mask
only. p35's first probe called it positionally and got a *signature* error, which proves nothing
about the mask. This settles it: print the signature, then call it the way the signature wants, at
the real shape, and check the value against the four-op chain it would replace.

Worth 3 minutes because if it did accept the bias it would collapse typecast + multiply + add +
softmax (4.028 ms measured) into one ~0.83 ms pass, 9 times a step, with no kernel.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ttnn  # noqa: E402

DEV = ttnn.open_device(device_id=0)
H, L, N, K = 4, 3359, 3360, 128
scale = 32 ** -0.5
print("signature:", getattr(ttnn.scale_mask_softmax, "__doc__", "")[:900], flush=True)

g = torch.Generator().manual_seed(0)
sc = ttnn.from_torch(torch.randn(1, H, L, N, generator=g), dtype=ttnn.float32,
                     layout=ttnn.TILE_LAYOUT, device=DEV)
bias_full = ttnn.from_torch(torch.randn(1, H, L, N, generator=g), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=DEV)
bias_row = ttnn.from_torch(torch.randn(1, 1, 1, N, generator=g), dtype=ttnn.float32,
                           layout=ttnn.TILE_LAYOUT, device=DEV)
ref = ttnn.to_torch(ttnn.softmax(ttnn.add(ttnn.multiply(sc, scale), bias_full), dim=-1))

for name, mask in (("full [1,4,3359,3360]", bias_full), ("row-broadcast [1,1,1,3360]", bias_row)):
    for form in ("kw", "pos_scale_kw_mask"):
        try:
            ttnn.synchronize_device(DEV)
            t0 = time.perf_counter()
            out = (ttnn.scale_mask_softmax(sc, scale=scale, mask=mask) if form == "kw"
                   else ttnn.scale_mask_softmax(sc, scale, mask=mask))
            ttnn.synchronize_device(DEV)
            ms = (time.perf_counter() - t0) * 1e3
            eq = torch.equal(ttnn.to_torch(out), ref) if mask is bias_full else "n/a"
            print(f"  {name:28s} {form:18s} ACCEPTED {ms:7.3f} ms  bit_exact={eq}", flush=True)
        except Exception as e:
            print(f"  {name:28s} {form:18s} REJECTED {type(e).__name__}: "
                  f"{' '.join(str(e).split())[:220]}", flush=True)

ttnn.close_device(DEV)
