#!/usr/bin/env python3
"""One arm x one model x one size of the W6 fold parity gate: fold it, keep the coordinates.

Deliberately does NOT implement a fold harness. It imports ``build_fold`` from
``scripts/gpu_vs_tt/tt_baseline.py``, which already opens the card, loads the model once, seeds
the MSA cache from a committed a3m so no search ever runs, and folds at the production config
(10 recycles / 200 sampling steps / 1 sample / seed 0).

What it adds on top: it captures the rank-0 coordinate array at full float precision by wrapping
``tt_bio.main._write_protenix_structure`` (both the protenix-v2 and the opendde emit paths import
it at call time, so one patch covers both). A CIF carries 3 decimals; a bit-exactness claim needs
better than 1e-3 A.

Results land in one JSON + one .npy + the CIFs per (arm, model, size), and an existing JSON makes
the run a no-op, so a relaunch resumes instead of repeating device time.

    python3 perf/w6_gate/fold_ab.py --arm BASE --model protenix-v2 --size 298 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

OUT = Path(__file__).resolve().parent / "out"

SIZES = {
    "117": (REPO / "examples/prot.yaml", REPO / "scripts/gpu_vs_tt/fixtures/prot117.a3m"),
    "298": (REPO / "examples/prot300.yaml", REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--repeat", type=int, default=3, help="warm folds after the cold one")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for the result name, for paired rounds")
    args = ap.parse_args()

    tag = f"{args.arm}_{args.model}_{args.size}{args.tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / f"{tag}.json"
    if js.exists() and not args.force:
        print(f"SKIP {tag}: {js} already exists")
        return 0

    # The arm must already be materialised in the worktree. Check it, rather than trusting the
    # caller's ordering: a mislabelled arm would silently poison every downstream verdict.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import arm as armmod
    want = armmod.build(args.arm)
    have = armmod.TARGET.read_text()
    if want != have:
        sys.exit(f"tt_bio/tenstorrent.py is not arm {args.arm} -- run arm.py --arm {args.arm} first")

    target, a3m = SIZES[args.size]
    msa_dir = Path.home() / "w6_gate_msa"

    coords_seen: list = []
    import numpy as np
    import tt_bio.main as _main
    _orig_write = _main._write_protenix_structure

    def _capture(coords, feats, *a, **k):
        coords_seen.append(np.asarray(coords, dtype=np.float64))
        return _orig_write(coords, feats, *a, **k)

    _main._write_protenix_structure = _capture

    from tt_baseline import build_fold
    one_fold, meta, state = build_fold(args.model, msa_dir, target, a3m)

    t0 = time.perf_counter()
    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"

    warm = []
    for _ in range(args.repeat):
        t, metrics = one_fold()
        warm.append(t)
    if not warm:
        metrics = cold_metrics

    # coords_seen holds one entry per written model per fold; 1 sample means 1 per fold.
    final = coords_seen[-1]
    # Same-seed folds inside one process must already agree, or the A/B has no resolution.
    self_max = float(np.abs(coords_seen[-1] - coords_seen[0]).max()) if len(coords_seen) > 1 else 0.0

    dst = OUT / tag
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for cif in sorted(Path(meta["struct_dir"]).glob("*.cif")):
        shutil.copy2(cif, dst / cif.name)
    np.save(dst / "coords.npy", final)

    rec = {
        "arm": args.arm, "model": args.model, "size": args.size,
        "n_tokens": metrics.get("n_tokens"), "n_atoms": metrics.get("n_atoms"),
        "n_residues": metrics.get("n_residues"),
        "cold_s": round(cold_s, 3), "warm_s": [round(t, 3) for t in warm],
        "warm_median_s": round(sorted(warm)[len(warm) // 2], 3) if warm else None,
        "warm_min_s": round(min(warm), 3) if warm else None,
        "confidence": {k: v for k, v in metrics.items()
                       if k in ("plddt", "complex_plddt", "ptm", "iptm", "confidence_score")},
        "coords_shape": list(final.shape),
        "intra_run_max_abs_delta_A": self_max,
        "n_folds": 1 + len(warm),
        "load_s": meta["load_s"], "n_msa": meta["n_msa"], "hardware": meta["hardware"],
        "card_type": meta.get("card_type"), "aiclk_mhz": meta.get("aiclk_mhz"),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    js.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))

    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
