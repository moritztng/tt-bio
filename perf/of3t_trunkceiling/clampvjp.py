"""Validate the clamp tape entry against a float64 torch VJP on the same values.

A tape rule is arithmetic, so it is checked against float64 and not against another
approximation. Two cases the -60 floor actually produces: a live row where nothing clips, and a
fully-masked row where everything clips.
"""
import json
import sys

import torch
import ttnn
import tt_bio.taped_ttnn as tt
from tt_bio import autograd as ag

from tt_bio.tenstorrent import get_device
dev = get_device()
out = {}
try:
    torch.manual_seed(0)
    for name, x_t in (("live_row", torch.randn(1, 1, 32, 128, dtype=torch.float32) * 3.0),
                      ("clips_hard", torch.randn(1, 1, 32, 128, dtype=torch.float32) * 3.0 - 70.0),
                      ("straddles", torch.linspace(-90, 10, 32 * 128).reshape(1, 1, 32, 128)
                       .to(torch.float32))):
        g_t = torch.randn_like(x_t)
        ref = x_t.to(torch.float64).clone().requires_grad_(True)
        ref.clamp(min=-60.0).backward(g_t.to(torch.float64))
        want = ref.grad

        with ag.tape():
            xv = ttnn.from_torch(x_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            x = ag.Tensor(xv, requires_grad=True)
            y = tt._VERBS["clamp"](ttnn.clamp, (x, -60.0, None), {})
            gv = ttnn.from_torch(g_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            y.backward(gv)
        got = ttnn.to_torch(x.grad).to(torch.float64)
        out[name] = {
            "clipped_elements": int((x_t < -60.0).sum()),
            "max_absdiff_vs_float64": float((got - want).abs().max()),
            "bit_identical": bool(torch.equal(got, want)),
            "ref_grad_norm": float(want.norm()),
        }
finally:
    pass
print("CLAMPVJP " + json.dumps(out))
bad = [k for k, v in out.items() if v["max_absdiff_vs_float64"] != 0.0]
sys.exit(1 if bad else 0)
