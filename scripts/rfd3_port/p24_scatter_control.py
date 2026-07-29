"""Positive control: make ttnn.scatter's output land on KNOWN-dirty DRAM.

p23 measured, end to end in a real fold, that a scatter output's tile padding held +/-3e38
and inf while the same call in a fresh process held finite values -- i.e. the op does not
write its output padding. p23's own op-level probe never actually landed a plant on the
output buffer ("scatter landed on dirtied address: False", 0x40 vs 0x17340), so the op-level
claim was never isolated. Without a working positive control, a clean-heap-vs-primed-heap
survey of other ops proves nothing.

Strategy: allocate the inputs first, then dirty a whole run of consecutive slots of the exact
output footprint, free them, run the op, and report which slot the output actually took.
"""
from __future__ import annotations

import sys

import torch

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()


def padstats(t):
    full = t.cpu().to_torch_with_padded_shape().float()
    box = tuple(slice(0, s) for s in t.shape)
    rest = full.clone()
    rest[box] = 0.0
    fin = rest[torch.isfinite(rest)]
    return "absmax=%-11.5g nonfinite=%-4d nonzero=%-7d uniq3=%s" % (
        float(fin.abs().max()) if fin.numel() else 0.0,
        int((~torch.isfinite(rest)).sum()), int((fin != 0).sum()),
        sorted(set(fin[fin != 0].flatten().tolist()))[:3])


def trial(L, K, H, dtype, plant_value, nslots):
    tmpl = ttnn.full((1, H, L, L), -1e4, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    idx = ttnn.from_torch(torch.arange(K).view(1, 1, 1, K).expand(1, H, L, K)
                          .to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT,
                          device=dev)
    src = ttnn.from_torch(torch.full((1, H, L, K), 0.5), dtype=dtype,
                          layout=ttnn.TILE_LAYOUT, device=dev)
    pshape = tuple(tmpl.padded_shape)
    dirty = [ttnn.full(pshape, plant_value, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                       device=dev) for _ in range(nslots)]
    addrs = [d.buffer_address() for d in dirty]
    for d in reversed(dirty):
        ttnn.deallocate(d)
    out = ttnn.scatter(tmpl, 3, idx, src)
    a = out.buffer_address()
    print("  L=%d %s plant=%-10.4g slots=[%s..%s] out=%s landed=%s" % (
        L, dtype, plant_value, hex(addrs[0]), hex(addrs[-1]), hex(a), a in addrs), flush=True)
    print("    template pad : %s" % padstats(tmpl), flush=True)
    print("    scatter  pad : %s" % padstats(out), flush=True)
    for t in (out, src, idx, tmpl):
        ttnn.deallocate(t)


print("=== ttnn.scatter (bfloat16), output forced onto dirtied DRAM, vs design size ===",
      flush=True)
for L in (130, 419, 1023, 1959, 2702, 3359):
    for plant in (1.2345e38, float("inf")):
        trial(L, 8, 4, ttnn.bfloat16, plant, 32 if L < 1024 else 6)
print("=== fp32 tiled scatter (what the production path would need) ===", flush=True)
try:
    trial(419, 8, 4, ttnn.float32, 1.2345e38, 8)
except Exception as e:  # noqa: BLE001
    print("  fp32 TILE scatter unsupported: %s" % str(e).splitlines()[0], flush=True)
