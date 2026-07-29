"""Op-level semantics behind the p23 fix.

1. `ttnn.scatter` does not write its output's tile padding -- it leaves whatever the freshly
   allocated DRAM buffer happened to hold. Planted by pre-dirtying the buffer.
2. `ttnn.softmax(dim=-1)` reads those pad columns (already shown end to end: re-uploading the
   same logical scores with zero padding reproduces the isolated answer).
3. `ttnn.pad` extending the key axis to a tile multiple writes the requested value over that
   region, leaving a tensor with no tile padding at all -- the basis of the fix.
"""
from __future__ import annotations

import sys

import torch

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()
L, K, H = 130, 8, 4          # logical 130 -> padded 160; small but same class


def padded(t):
    return t.cpu().to_torch_with_padded_shape().float()


def padstats(t):
    full = padded(t)
    box = tuple(slice(0, s) for s in t.shape)
    rest = full.clone()
    rest[box] = 0.0
    fin = rest[torch.isfinite(rest)]
    return (tuple(t.shape), tuple(t.padded_shape), float(fin.abs().max()) if fin.numel() else 0.0,
            int(torch.isinf(rest).sum()), sorted(set(fin.flatten().tolist()))[:4])


print("=== 1. does ttnn.scatter write its output's tile padding? ===", flush=True)
dirty = ttnn.full((1, H, 160, 160), 1.2345e38, dtype=ttnn.bfloat16,
                  layout=ttnn.TILE_LAYOUT, device=dev)
addr = dirty.buffer_address()
ttnn.deallocate(dirty)
tmpl = ttnn.full((1, H, L, L), -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
print("  template          shape/padded/pad_absmax/inf/uniq:", padstats(tmpl), flush=True)
idx = ttnn.from_torch(torch.arange(K).view(1, 1, 1, K).expand(1, H, L, K).to(torch.int32),
                      dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=dev)
src = ttnn.from_torch(torch.full((1, H, L, K), 0.5), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=dev)
out = ttnn.scatter(tmpl, 3, idx, src)
print("  scatter out       shape/padded/pad_absmax/inf/uniq:", padstats(out), flush=True)
print("  scatter landed on dirtied address:", hex(out.buffer_address()) == hex(addr),
      hex(out.buffer_address()), hex(addr), flush=True)

print("=== 3. does ttnn.pad write a defined value into the extended key axis? ===", flush=True)
src2 = ttnn.from_torch(torch.randn(1, H, L, 32), dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=dev)
try:
    p = ttnn.pad(src2, [(0, 0), (0, 0), (0, 160 - L), (0, 0)], 0.0)
    full = padded(p)
    print("  pad out           shape/padded:", tuple(p.shape), tuple(p.padded_shape), flush=True)
    print("  new rows all zero:", bool((full[..., L:, :] == 0).all()), flush=True)
    print("  original rows preserved:",
          bool(torch.equal(full[..., :L, :], padded(src2)[..., :L, :])), flush=True)
except Exception as e:  # noqa: BLE001
    print("  ttnn.pad FAILED:", type(e).__name__, e, flush=True)

print("=== 3b. pad a tensor whose tile padding is garbage (scatter output) ===", flush=True)
try:
    q = ttnn.pad(out, [(0, 0), (0, 0), (0, 0), (0, 160 - L)], -1e4)
    print("  pad out           shape/padded/pad_absmax/inf/uniq:", padstats(q), flush=True)
    fullq = padded(q)
    print("  new columns == -1e4:",
          sorted(set(fullq[..., :, L:].flatten().tolist()))[:4], flush=True)
except Exception as e:  # noqa: BLE001
    print("  ttnn.pad FAILED:", type(e).__name__, e, flush=True)
