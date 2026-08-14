#!/usr/bin/env python3
"""S4 -- the two host screens that decide levers D and B before either is built. No device.

LEVER D. Stage 1 computes all (2*box/32)^2 tiles of the padded plane per component, but a slice only
samples a disc. With padding factor 2 the source coordinate of an output sample runs out to r_max*pad
~ box in a plane of width 2*box, so the sampled set is the INSCRIBED disc, and it is the same disc for
every in-plane rotation because a slice passes through the plane centre. The union over psi is
therefore the disc itself and the answer does not depend on the direction. What the screen has to add
is the dilation the reader's window imposes: 64 elements to the right of the leftmost sample of a row,
and the 3-tap +-1 rows. GO if the tile-granular fraction is <= 0.85.

LEVER B. A 2-pass separable rotation by theta forces A = cos(theta) for pass 1 and D = 1/cos(theta)
for pass 2, so one pass always decimates and pass 2's r window advances 32*D per output tile, not 32.
The reuse ratio over a tile-column run of T output tiles is n(T, D) / (2T) with
n = (D*(32T - 1) + 2)/32 + 1. Fire the kill gate if the mean r over the psi grid is above 0.78.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PSI = 96
WIN = 64            # elements the reader pulls per row
TAPS = 1            # +-1 row around the pass-2 sample
T_RUN = 25.133 / 5.0   # output tiles per tile-column run: 25.13 disc tiles over 5 column positions


def lever_d(box, pad=2):
    """Fraction of the padded plane's 32x32 tiles a slice can touch, at tile granularity."""
    n = pad * box
    g = n // 32
    r = box                       # sampled radius in source elements = r_max * pad
    c = n / 2.0
    yy, xx = np.mgrid[0:g, 0:g]
    # A tile [32x, 32x+32) x [32y, 32y+32) is touched if the dilated disc meets it. The dilation is
    # anisotropic: the read window runs WIN elements to the right of the leftmost sample of a row and
    # the pass-2 tap reaches TAPS rows out, so shrink the tile box by the dilation on those sides and
    # test the closest point of the shrunk box against the disc.
    x0, x1 = 32 * xx - WIN, 32 * xx + 32
    y0, y1 = 32 * yy - TAPS, 32 * yy + 32 + TAPS
    dx = np.maximum(np.maximum(x0 - c, c - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - c, c - y1), 0.0)
    hit = (dx * dx + dy * dy) <= r * r
    # the undilated disc, for reference
    x0u, x1u, y0u, y1u = 32 * xx, 32 * xx + 32, 32 * yy, 32 * yy + 32
    dxu = np.maximum(np.maximum(x0u - c, c - x1u), 0.0)
    dyu = np.maximum(np.maximum(y0u - c, c - y1u), 0.0)
    hitu = (dxu * dxu + dyu * dyu) <= r * r
    return {"box": box, "grid": int(g), "tiles": int(g * g),
            "frac_tiles_dilated": float(hit.mean()), "frac_tiles_raw_disc": float(hitu.mean()),
            "frac_area": math.pi * r * r / (n * n)}


def reuse_ratio(t, d):
    return ((d * (32 * t - 1) + 2) / 32.0 + 1.0) / (2.0 * t)


def lever_b(t_run=T_RUN, psi=PSI):
    """r over the psi grid, with theta reduced into [-45, 45] degrees as mode 13's transpose allows."""
    ang = np.arange(psi) * (2 * math.pi / psi)
    th = (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4     # the |theta| <= 45 deg reduction
    d = 1.0 / np.cos(th)
    r = reuse_ratio(t_run, d)
    return {"t_run": t_run, "psi": psi, "theta_max_deg": float(np.abs(th).max() * 180 / math.pi),
            "D_mean": float(d.mean()), "D_max": float(d.max()),
            "r_mean": float(r.mean()), "r_min": float(r.min()), "r_max": float(r.max()),
            "assemblies_speedup_mean": float(1.0 / r.mean())}


def main():
    out = {"lever_d": [lever_d(b) for b in (256, 384, 512)], "lever_b": lever_b()}
    for e in out["lever_d"]:
        print(f"lever D  box {e['box']:4d}  {e['grid']}x{e['grid']} tiles  "
              f"disc {e['frac_tiles_raw_disc']:.4f}  dilated {e['frac_tiles_dilated']:.4f}  "
              f"(area {e['frac_area']:.4f})")
    b = out["lever_b"]
    print(f"lever B  T = {b['t_run']:.3f}  D mean {b['D_mean']:.4f} max {b['D_max']:.4f}  "
          f"r mean {b['r_mean']:.4f} (min {b['r_min']:.4f} max {b['r_max']:.4f})  "
          f"assemblies {b['assemblies_speedup_mean']:.3f}x")
    print(f"\nGATE lever D (<= 0.85): "
          f"{'GO' if out['lever_d'][0]['frac_tiles_dilated'] <= 0.85 else 'NO-GO'}")
    print(f"GATE lever B (mean r <= 0.78): {'GO' if b['r_mean'] <= 0.78 else 'NO-GO -- defer'}")
    json.dump(out, open(HERE / "s4_geom.json", "w"), indent=1)


main()
