#!/usr/bin/env python3
"""Per-cell completeness census for the AbAg-XM N=512 panel over the FULL 164-target set.

The panel is 164 targets x 4 models x 8 chunks = 5248 chunk-folds, which pool into
164 x 4 = 656 cells. Every earlier census in this campaign ran against a reduced
denominator (160 targets, or a per-model target list carrying the Wormhole DRAM
exclusions), so a missing cell and an excluded cell were indistinguishable. This script
takes the denominator from the yaml set itself and reports what is absent, by name.

A chunk-fold counts as complete only if all of these hold, which is the campaign's own
`verify()` contract:

  * the harvest dir exists                     <root>/<model_dir>/<target>_n512_c<j>
  * it holds one results dir                   */<model>_results_<target>
  * results.json parses, status == "ok", len(all_runs) == 64
  * the results dir holds 64 .cif files
  * labels.json exists beside the harvest dir  (DockQ labels, needed by the panel read)

Usage (on the analysis host, qb1):

    python3 census_panel_164.py --yaml-dir <repo>/examples/abag_xm
    python3 census_panel_164.py --targets "21av 21du ..."      # explicit list
    python3 census_panel_164.py --root ~/abag_xm/deepn/galaxy --json census.json

Exits 0 when all 5248 chunk-folds are complete, 1 otherwise, so it can gate a window.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

# harvest dir name -> the model name used in results.jsonl / the panel artifacts
MODEL_DIRS = {
    "boltz2": "boltz2",
    "esmfold2": "esmfold2",
    "opendde": "opendde-abag",
    "protenix": "protenix-v2",
}
CHUNKS = 8
SAMPLES_PER_CHUNK = 64


def cell_state(root: str, model_dir: str, target: str, chunk: int) -> str | None:
    """Return None when the chunk-fold is complete, else a short reason."""
    d = os.path.join(root, model_dir, f"{target}_n512_c{chunk}")
    if not os.path.isdir(d):
        return "no dir"
    rds = glob.glob(os.path.join(d, f"*results_{target}"))
    if not rds:
        return "no results dir"
    rj = os.path.join(rds[0], "results.json")
    if not os.path.exists(rj):
        return "no results.json"
    try:
        r = json.load(open(rj))
    except Exception as exc:                                   # truncated / mid-write
        return f"unparseable results.json ({exc.__class__.__name__})"
    r0 = r[0] if isinstance(r, list) else r
    status = r0.get("status")
    n_runs = len(r0.get("all_runs") or [])
    n_cif = len(glob.glob(os.path.join(rds[0], "structures", "**", "*.cif"), recursive=True))
    if status != "ok" or n_runs != SAMPLES_PER_CHUNK or n_cif != SAMPLES_PER_CHUNK:
        return f"status={status} all_runs={n_runs} cifs={n_cif}"
    if not os.path.exists(os.path.join(d, "labels.json")):
        return "no labels.json"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/abag_xm/deepn/galaxy"))
    ap.add_argument("--yaml-dir", default=None,
                    help="directory of <target>.yaml; sets the denominator")
    ap.add_argument("--targets", default=None, help="explicit space-separated target list")
    ap.add_argument("--expect", type=int, default=164, help="expected target count")
    ap.add_argument("--json", default=None, help="write the machine-readable census here")
    args = ap.parse_args()

    if args.targets:
        targets = sorted(args.targets.split())
    elif args.yaml_dir:
        targets = sorted(os.path.basename(p)[:-5]
                         for p in glob.glob(os.path.join(args.yaml_dir, "*.yaml")))
    else:
        ap.error("pass --yaml-dir or --targets")
    if len(targets) != args.expect:
        print(f"CENSUS ABORT: {len(targets)} targets, expected {args.expect}. "
              "The denominator is the point of this script; fix the target set first.")
        return 2

    missing: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
    for model_dir, model in MODEL_DIRS.items():
        for t in targets:
            for j in range(CHUNKS):
                why = cell_state(args.root, model_dir, t, j)
                if why:
                    missing[model].append((t, j, why))

    n_t = len(targets)
    per_model_slots = n_t * CHUNKS
    print(f"AbAg-XM N=512 panel census   root={args.root}")
    print(f"denominator: {n_t} targets x {len(MODEL_DIRS)} models x {CHUNKS} chunks "
          f"= {n_t * len(MODEL_DIRS) * CHUNKS} chunk-folds, "
          f"{n_t * len(MODEL_DIRS)} cells\n")

    total_bad = 0
    cells_ok_total = 0
    print(f"{'model':14s} {'chunk-folds':>16s} {'cells (8/8 chunks)':>20s}")
    for model in MODEL_DIRS.values():
        bad = missing[model]
        total_bad += len(bad)
        short_targets = {t for t, _, _ in bad}
        cells_ok = n_t - len(short_targets)
        cells_ok_total += cells_ok
        print(f"{model:14s} {per_model_slots - len(bad):8d}/{per_model_slots:<7d} "
              f"{cells_ok:12d}/{n_t:<7d}")
    print(f"\nchunk-folds complete: {n_t * len(MODEL_DIRS) * CHUNKS - total_bad}"
          f"/{n_t * len(MODEL_DIRS) * CHUNKS}")
    print(f"cells complete:       {cells_ok_total}/{n_t * len(MODEL_DIRS)}")

    if total_bad:
        print("\nincomplete chunk-folds, by model:")
        for model in MODEL_DIRS.values():
            bad = missing[model]
            if not bad:
                print(f"  {model}: none")
                continue
            by_t = collections.defaultdict(list)
            for t, j, why in bad:
                by_t[t].append((j, why))
            print(f"  {model}  ({len(bad)} chunk-folds over {len(by_t)} targets)")
            for t in sorted(by_t):
                cs = by_t[t]
                reasons = sorted({w for _, w in cs})
                print(f"    {t}: chunks {sorted(j for j, _ in cs)}  {reasons}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "root": args.root,
                "targets": targets,
                "models": list(MODEL_DIRS.values()),
                "chunk_folds_total": n_t * len(MODEL_DIRS) * CHUNKS,
                "chunk_folds_complete": n_t * len(MODEL_DIRS) * CHUNKS - total_bad,
                "cells_total": n_t * len(MODEL_DIRS),
                "cells_complete": cells_ok_total,
                "missing": {m: [{"target": t, "chunk": j, "reason": w}
                                for t, j, w in v] for m, v in missing.items()},
            }, fh, indent=1)
        print(f"\nwrote {args.json}")

    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
