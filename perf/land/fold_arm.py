#!/usr/bin/env python3
"""Fold one arm of the qb1 landing stack and keep the coordinates.

An arm here is a (commit, TT_BIO_TRIMUL_OUT_FUSED) pair, not a patched file: every
candidate is a real commit on wk/perfwar-qb1-rebaseline-and-land, checked out into its own
git worktree under arms/. This script never edits a tree, so two arms can never race the
way a string-substituting arm builder can.

It does not implement a fold harness. `build_fold` from scripts/gpu_vs_tt/tt_baseline.py
opens the card, loads the model once, seeds the MSA cache from a committed a3m so no search
runs, and folds at the production config (10 recycles / 200 sampling steps / 1 sample /
seed 0) -- the same entry point W11's gate used, so the numbers are comparable to it.

    python3 perf/land/fold_arm.py --tag L0 --tree arms/L0 --expect <sha> \
        --model protenix-v2 --size 298 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def _git(tree: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} in {tree} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm name, e.g. L0 or L1F")
    ap.add_argument("--tree", required=True, help="worktree holding this arm's commit")
    ap.add_argument("--expect", required=True, help="commit the tree must be at")
    ap.add_argument("--fused", default="0", choices=["0", "1"],
                    help="TT_BIO_TRIMUL_OUT_FUSED for this arm")
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--size", required=True, choices=["117", "298"])
    ap.add_argument("--repeat", type=int, default=3, help="warm folds after the cold one")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tree = Path(args.tree).resolve()
    tag = f"{args.tag}_{args.model}_{args.size}"
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / f"{tag}.json"
    if js.exists() and not args.force:
        print(f"SKIP {tag}: {js} already exists")
        return 0

    # The arm is the tree's state, so check it rather than trusting the caller's ordering.
    head = _git(tree, "rev-parse", "HEAD")
    if not head.startswith(args.expect) and not args.expect.startswith(head[:8]):
        sys.exit(f"{tree} is at {head}, arm {args.tag} wants {args.expect}")
    dirty = _git(tree, "status", "--porcelain", "--", "tt_bio")
    if dirty:
        sys.exit(f"{tree}/tt_bio is dirty, refusing to label a modified tree as {args.tag}:\n{dirty}")

    # Read at tt_bio.tenstorrent import time, so it has to be set before the import below.
    os.environ["TT_BIO_TRIMUL_OUT_FUSED"] = args.fused
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))
    sys.path.insert(0, str(tree))

    target = tree / ("examples/prot300.yaml" if args.size == "298" else "examples/prot.yaml")
    a3m = tree / ("scripts/gpu_vs_tt/fixtures/prot300.a3m" if args.size == "298"
                  else "scripts/gpu_vs_tt/fixtures/prot117.a3m")
    msa_dir = Path.home() / "w6_gate_msa"   # shared with W11: same target, same alignment

    import numpy as np
    import tt_bio.main as _main
    import tt_bio.tenstorrent as _tt
    if str(Path(_tt.__file__).resolve().parent.parent) != str(tree):
        sys.exit(f"imported tt_bio from {_tt.__file__}, not from {tree}")
    if args.fused == "1" and not _tt._TRIMUL_OUT_FUSED:
        sys.exit("arm asks for the fused output op but this commit does not carry the flag")

    coords_seen: list = []
    _orig_write = _main._write_protenix_structure

    def _capture(coords, feats, *a, **k):
        # A CIF carries 3 decimals; a bit-exactness claim needs better than 1e-3 A.
        coords_seen.append(np.asarray(coords, dtype=np.float64))
        return _orig_write(coords, feats, *a, **k)

    _main._write_protenix_structure = _capture

    from tt_baseline import build_fold
    one_fold, meta, _state = build_fold(args.model, msa_dir, target, a3m)

    t0 = time.perf_counter()
    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"
    warm, metrics = [], cold_metrics
    for _ in range(args.repeat):
        t, metrics = one_fold()
        warm.append(t)

    final = coords_seen[-1]
    self_max = float(np.abs(coords_seen[-1] - coords_seen[0]).max()) if len(coords_seen) > 1 else 0.0

    dst = OUT / tag
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for cif in sorted(Path(meta["struct_dir"]).glob("*.cif")):
        shutil.copy2(cif, dst / cif.name)
    np.save(dst / "coords.npy", final)

    rec = {
        "arm": args.tag, "commit": head, "fused": args.fused,
        "model": args.model, "size": args.size,
        "n_tokens": metrics.get("n_tokens"), "n_atoms": metrics.get("n_atoms"),
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

    _tt.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
