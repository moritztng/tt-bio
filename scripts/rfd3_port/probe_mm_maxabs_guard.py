"""Is `_mm_maxabs` a real global reduce? Plant one outlier and see if it is found.

p15's calibration guard accepted a program config that differed everywhere except row 0,
because `ttnn.max(d)` with no `dim` was read as if it were a global reduce. The fix takes the
host-side max of whatever `ttnn.max` returns, which only works if `ttnn.max` reduces every
element into its result. This probe plants a single 1.0 in an otherwise-zero tensor at the far
corner and checks each candidate reduction against the known answer.

Usage: TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/probe_mm_maxabs_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402

SHAPES = [
    (8, 250, 250, 128),   # pair tensor, ragged I
    (1, 8, 250, 768),     # DiT token tensor
    (8, 40, 40, 512),     # small pair tensor
]


def main():
    dev = get_device()
    bad = 0
    for shape in SHAPES:
        t = torch.zeros(shape)
        # far corner, and a second one mid-tensor, both outside row 0 of every leading dim
        t[(-1,) * (len(shape) - 1) + (-1,)] = 3.5
        want = 3.5
        d = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        got_nodim = float(ttnn.to_torch(ttnn.max(d)).float().abs().max())
        got_lastdim = float(ttnn.to_torch(ttnn.max(d, dim=-1)).float().abs().max())
        print(f"{shape}  want={want}  ttnn.max(d)={got_nodim}  "
              f"ttnn.max(d,dim=-1)->host={got_lastdim}", flush=True)
        bad += got_nodim != want or got_lastdim != want
        ttnn.deallocate(d)

    print("GUARD PROBE", "PASS" if bad == 0 else f"FAIL ({bad} shapes)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
