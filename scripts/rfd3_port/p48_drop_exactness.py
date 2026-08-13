"""The single assumption S1's bit-exactness rests on, tested directly.

The argument on record: a dropped tile is fully masked, its post-exp contribution is exactly 0.0 in
fp32, and adding exact zeros to a float sum in any grouping is exact -- so computing softmax over
only the surviving tiles gives bit-identical weights, and `torch.equal` is the gate rather than a
PCC.

That holds if ttnn reduces the softmax denominator by scanning tiles into one accumulator: adding a
zero tile leaves the accumulator bit-identical. It does NOT obviously hold if the reduction is a
tree across tiles, because dropping tiles re-pairs the survivors.

So this is an empirical question and it is cheap. Build one score row set at full width with the
dropped positions masked to -1e4, and the same values compacted to the surviving tiles, and compare
the surviving columns of softmax(full) against softmax(compact) with torch.equal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

TILE = 32
MASK = -1e4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64, help="query rows (2 row blocks)")
    ap.add_argument("--full", type=int, default=6080)
    ap.add_argument("--keep", type=int, default=32, help="surviving column tiles")
    ap.add_argument("--heads", type=int, default=4)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    g = torch.Generator().manual_seed(7)
    n_full_t = a.full // TILE
    keep = torch.randperm(n_full_t, generator=g)[:a.keep].sort().values   # ORIGINAL ORDER kept

    s = torch.randn(1, a.heads, a.rows, a.full, generator=g) * 3.0
    full = torch.full_like(s, MASK)
    for t in keep:
        lo = int(t) * TILE
        full[..., lo:lo + TILE] = s[..., lo:lo + TILE]
    comp = torch.cat([s[..., int(t) * TILE:int(t) * TILE + TILE] for t in keep], dim=-1)

    def sm(x):
        d = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.to_torch(ttnn.softmax(d, dim=-1))
        ttnn.deallocate(d)
        return o

    o_full, o_comp = sm(full), sm(comp)
    o_full_kept = torch.cat([o_full[..., int(t) * TILE:int(t) * TILE + TILE] for t in keep], dim=-1)

    dropped_max = float(o_full[..., [i for i in range(a.full)
                                     if i // TILE not in set(int(t) for t in keep)]].abs().max())
    eq = torch.equal(o_full_kept, o_comp)
    d = (o_full_kept.double() - o_comp.double()).abs()
    print(f"full width {a.full} ({n_full_t} tiles), keeping {a.keep} tiles "
          f"({a.keep * TILE} cols, density {a.keep / n_full_t:.4f})")
    print(f"dropped columns of softmax(full):  max |value| = {dropped_max:.3e}  "
          f"{'exactly zero' if dropped_max == 0.0 else 'NOT ZERO'}")
    print(f"surviving columns, softmax(full) vs softmax(compact):")
    print(f"  torch.equal = {eq}")
    print(f"  max |diff|  = {float(d.max()):.3e}   max rel = "
          f"{float(d.max() / o_comp.abs().max()):.3e}")
    print(f"\nVERDICT: dropping fully-masked tiles is "
          f"{'BIT-EXACT -- torch.equal is a valid gate' if eq else 'NOT bit-exact; the softmax reduction re-pairs'}")
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
