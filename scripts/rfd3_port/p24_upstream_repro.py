"""Minimal, self-contained repro of two ttnn 0.68.0 correctness defects.

Defect A -- `ttnn.softmax(dim=-1)` reads the tile padding of its reduction axis.
    Two tensors with bit-identical LOGICAL contents but different tile padding give
    different logical softmax outputs. Tile padding is not part of a tensor's value, so
    this makes the op's result depend on memory the user cannot see or control.

Defect B -- `ttnn.scatter` does not write its output's tile padding.
    The output buffer keeps whatever the freshly allocated DRAM happened to hold. Combined
    with A, a scatter feeding a softmax silently returns different numbers depending on what
    ran earlier in the same process.

Together they cost a 0.335 A coordinate error in a production protein-design model, which is
how they were found: the same design folded to two different structures depending only on
which other designs had been folded before it in the same process.

Run: python3 p24_upstream_repro.py        (needs a single Tenstorrent device; no other deps)
"""
from __future__ import annotations

import torch

import ttnn

DEV = ttnn.open_device(device_id=0)

# The bug needs the reduction axis to be a non-tile-multiple, and enough pad columns for the
# stale values to matter. 2702 = 84 tiles + 14, so 18 pad columns of every 2720-wide row.
B, H, L, K = 1, 4, 2702, 8
PAD = -1e4


def padded(t):
    return t.cpu().to_torch_with_padded_shape().float()


def pad_cols(t):
    return padded(t)[..., : t.shape[-2], t.shape[-1]:]


def summarise(name, t):
    c = pad_cols(t)
    fin = c[torch.isfinite(c)]
    print("  %-22s logical=%s padded=%s  pad cols: absmax=%.6g nonfinite=%d" % (
        name, tuple(t.shape), tuple(t.padded_shape),
        float(fin.abs().max()) if fin.numel() else 0.0,
        int((~torch.isfinite(c)).sum())), flush=True)


def dirty_dram(value, chunk=(1, 32, 2048, 2048)):
    """Fill all free DRAM with `value` and release it, so the next buffer lands on it."""
    held = []
    while True:
        try:
            held.append(ttnn.full(chunk, value, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=DEV))
        except Exception:  # noqa: BLE001  DRAM full
            break
    for t in reversed(held):
        ttnn.deallocate(t)
    return len(held)


print("=== Defect B: ttnn.scatter leaves its output's tile padding undefined ===", flush=True)
print("  dirtied %d x 256 MB of DRAM with +inf" % dirty_dram(float("inf")), flush=True)

tmpl = ttnn.full((B, H, L, L), PAD, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)
idx = ttnn.from_torch(torch.arange(K).view(1, 1, 1, K).expand(B, H, L, K).to(torch.int32),
                      dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=DEV)
src = ttnn.from_torch(torch.full((B, H, L, K), 0.5), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=DEV)
summarise("ttnn.full template", tmpl)          # ttnn.full DOES write its padding: all -1e4
out = ttnn.scatter(tmpl, 3, idx, src)
summarise("ttnn.scatter output", out)          # ... the scatter output's padding does not
print("  EXPECTED: scatter output padding is defined (e.g. inherited -1e4, or zero).", flush=True)
print("  ACTUAL:   it is the +inf this script wrote into free DRAM before allocating it.",
      flush=True)

print("=== Defect A: ttnn.softmax(dim=-1) reads that padding ===", flush=True)
logical = ttnn.to_torch(out)                   # the tensor's value, padding excluded
clean = ttnn.from_torch(logical, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)
summarise("re-uploaded copy", clean)           # host upload zero-fills the padding
print("  logical contents identical:", bool(torch.equal(logical, ttnn.to_torch(clean))),
      flush=True)

a = ttnn.to_torch(ttnn.softmax(out, dim=-1)).float()
b = ttnn.to_torch(ttnn.softmax(clean, dim=-1)).float()
delta = (a - b).abs().max().item()
print("  softmax(dirty-padding) sum = %.12f" % a.double().sum().item(), flush=True)
print("  softmax(zero-padding)  sum = %.12f" % b.double().sum().item(), flush=True)
print("  max |difference| over the LOGICAL region = %.6g" % delta, flush=True)
print("  EXPECTED: 0 -- the two inputs have the same value.", flush=True)
print("  ACTUAL:   %s" % ("0 (not reproduced on this run)" if delta == 0 else
                          "%.6g -- softmax's result depends on invisible memory" % delta),
      flush=True)

ttnn.close_device(DEV)
