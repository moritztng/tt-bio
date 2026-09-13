#!/usr/bin/env python3
"""Score a confidence-head change the way a confidence-head change has to be scored.

The confidence head runs after the sampler and its outputs are scores, so at one diffusion sample
it cannot move an atom: no confidence-ranked selection is live (the multi-sample ``model_rank``
ordering and the affinity path's ``argsort(iptm)`` both degenerate at n = 1). The coordinate claim
is therefore an equality, not a bar -- and the quantity that DOES move, the per-atom pLDDT the CIF
carries in its B-factor column, is reported on its own.

That is also why a CIF sha256 is the wrong instrument here: it will differ whenever a pLDDT digit
moves, with every coordinate identical. Both readings are printed.

    score_conf.py <ref_dir> <arm_dir> [<arm_dir> ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def read_cif(path: Path):
    """``(coords, plddt)`` from the ATOM records: x/y/z and the B-factor column."""
    coords, plddt = [], []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM ", "HETATM ")):
            continue
        f = line.split()
        coords.append((float(f[10]), float(f[11]), float(f[12])))
        plddt.append(float(f[17]))
    return coords, plddt


def compare(ref: Path, arm: Path) -> dict:
    name = sorted(p.name for p in ref.glob("*.cif"))[0]
    rc, rp = read_cif(ref / name)
    ac, ap = read_cif(arm / name)
    assert len(rc) == len(ac), f"atom count {len(rc)} vs {len(ac)}"
    dmax = max(max(abs(a - b) for a, b in zip(u, v)) for u, v in zip(rc, ac))
    sq = sum(sum((a - b) ** 2 for a, b in zip(u, v)) for u, v in zip(rc, ac))
    dp = [abs(a - b) for a, b in zip(rp, ap)]
    return {
        "file": name,
        "atoms": len(rc),
        "coord_max_abs_delta_A": round(dmax, 6),
        "coord_rmsd_A": round((sq / len(rc)) ** 0.5, 6),
        "coords_identical": dmax == 0.0,
        "plddt_max_abs_delta": round(max(dp), 6),
        "plddt_mean_abs_delta": round(sum(dp) / len(dp), 6),
        "plddt_mean_ref": round(sum(rp) / len(rp), 4),
        "plddt_mean_arm": round(sum(ap) / len(ap), 4),
        "sha_ref": hashlib.sha256((ref / name).read_bytes()).hexdigest()[:16],
        "sha_arm": hashlib.sha256((arm / name).read_bytes()).hexdigest()[:16],
    }


def main() -> int:
    ref, arms = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    out = {"ref": ref.name, "arms": {}}
    for arm in arms:
        r = compare(ref, arm)
        out["arms"][arm.name] = r
        print(f"{arm.name:20s} coords {'IDENTICAL' if r['coords_identical'] else 'MOVED'}  "
              f"max {r['coord_max_abs_delta_A']:.6f} A  rmsd {r['coord_rmsd_A']:.6f} A  "
              f"| plddt max {r['plddt_max_abs_delta']:.3f} mean {r['plddt_mean_abs_delta']:.4f}  "
              f"| sha {r['sha_ref']} -> {r['sha_arm']}")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
