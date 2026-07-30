#!/usr/bin/env python3
"""Stage the coordinate half of the release: gzipped mmCIF + PAE, in the published layout.

The parquet tables cover the tabular half. This is the other half, and it is the expensive
one: 164 targets x 3 generators x 50 samples is ~24,600 CIFs and ~24,600 PAE arrays. Both
the gzip pass and the upload are slow enough that discovering them at publish time is how a
release slips a day.

    structures/<generator>/<target>/<target>_model_<k>.cif.gz
    pae/<generator>/<target>/<target>_model_<k>_pae.npz

Idempotent: a file already staged with a newer mtime than its source is skipped, so this can
be run repeatedly as Tier A completes and only does the new work. Reads `tier_a`, writes only
under --out_dir.

    python3 scripts/abag_xm_stage_release.py --out_dir ~/abag_xm/release [--limit N] [--dry_run]
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

TIERA = Path.home() / "abag_xm" / "tier_a"
PROGRESS = TIERA / "progress.jsonl"
GEN_DIR = {"protenix-v2": ("protenix_v2", "protenix"),
           "opendde-abag": ("opendde_abag", "opendde"),
           "boltz2": ("boltz2", "boltz2")}


def ok_folds():
    seen = {}
    if PROGRESS.exists():
        for line in PROGRESS.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                seen[(r["target"], r["model"])] = r
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(Path.home() / "abag_xm" / "release"))
    ap.add_argument("--limit", type=int, default=0, help="stage at most N folds (for a trial run)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)

    folds = sorted(ok_folds())
    if a.limit:
        folds = folds[:a.limit]
    n_cif = n_pae = 0
    raw = comp = 0
    skipped = 0
    for target, gen in folds:
        sub, prefix = GEN_DIR[gen]
        src = TIERA / sub / f"{prefix}_results_{target}" / "structures"
        if not src.is_dir():
            print(f"  {target}/{gen}: no structures dir", file=sys.stderr)
            continue
        sdir = out / "structures" / gen / target
        pdir = out / "pae" / gen / target
        if not a.dry_run:
            sdir.mkdir(parents=True, exist_ok=True)
            pdir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.cif")):
            # Sample 0's coordinates are written as "<target>.cif", every other sample as
            # "<target>_model_<k>.cif" -- verified against the label JSON, whose rank-0 entry
            # points at the unnumbered file, and there is no _model_0.cif to collide with.
            # Publishing that asymmetry would mean a user looping k in range(50) over the
            # documented "<target>_model_<k>.cif.gz" pattern silently misses one sample per
            # fold. Stage it as _model_0 so all 50 match, and so CIF naming agrees with the
            # PAE files, which already use _model_0.
            name = f"{target}_model_0.cif" if f.stem == target else f.name
            dst = sdir / (name + ".gz")
            if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
                skipped += 1
                continue
            raw += f.stat().st_size
            if not a.dry_run:
                with f.open("rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
                    shutil.copyfileobj(fi, fo)
                comp += dst.stat().st_size
            n_cif += 1
        # PAE is left exactly as produced. Each fold has _model_0.._model_49 plus an
        # unnumbered "<target>_pae.npz", and that extra file is NOT byte-identical to
        # _model_0's, so what it represents is unresolved -- possibly the top-ranked rather
        # than the first sample. Neither dropping nor renaming it is safe without knowing,
        # so it ships as-is and the question is recorded for the publish step.
        for f in sorted(src.glob("*_pae.npz")):
            dst = pdir / f.name
            if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
                skipped += 1
                continue
            if not a.dry_run:
                shutil.copy2(f, dst)
                comp += dst.stat().st_size
            n_pae += 1

    print(f"folds staged      {len(folds)}")
    print(f"cif gzipped       {n_cif}")
    print(f"pae copied        {n_pae}")
    print(f"already current   {skipped}")
    if raw and comp and not a.dry_run:
        print(f"cif raw           {raw / 1e6:.1f} MB")
        print(f"staged total      {comp / 1e6:.1f} MB")
        per_fold = comp / max(1, len(folds))
        print(f"per fold          {per_fold / 1e6:.1f} MB  -> 492 folds ~= "
              f"{per_fold * 492 / 1e9:.1f} GB, ~{492 * 100:,} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
